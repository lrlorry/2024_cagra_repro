#include "engineered/engineered_search.cuh"

#include "common/cagra_build_kernels.cuh"
#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"

#include <math_constants.h>

namespace cagra_repro::engineered {

// ── MSB parent flag ───────────────────────────────────────────────────────────
/*
 * top-M 列表中每个条目的节点 id 用最高位（bit 31）标记"是否已作为父节点展开过"。
 *
 * 动机：CAGRA 的 beam search 是贪心的——每次迭代从 top-M 中选最靠前且未展开的节点，
 *   读取其全部出边，把这些邻居作为候选插入 top-M。一旦某节点被展开，后续不再重复展开，
 *   以避免无限循环并保证搜索单调收敛。
 *
 * 选择 MSB 而非独立 bool 数组的原因：
 *   top-M 存于 shared memory，减少一个 kInternalTopM 大小的 bool 数组可节省 smem 压力。
 *   MSB 不影响任何合法节点 id（节点数 n < 2^31），可安全打包。
 *
 * 对应 cuVS：search_plan.cuh → HashMapT 中的 parent flag 机制。
 */
__device__ __forceinline__ unsigned strip_parent(unsigned x) { return x & kIdMask; }
__device__ __forceinline__ unsigned mark_parent(unsigned x)  { return x | kParentMask; }
__device__ __forceinline__ bool     is_parented(unsigned x)  { return (x & kParentMask) != 0; }

// ── Block-cooperative L2 distance (float4 vectorised) ────────────────────────
/*
 * 计算 dataset[point] 与 queries[query] 之间的 L2² 距离（平方，省去 sqrt）。
 *
 * 并行策略：
 *   block 内所有 kBlockThreads 个线程共同完成一次距离计算，每线程负责 dim/blockDim.x 个维度。
 *   partial[threadIdx.x] 收集各线程的局部平方和，最后通过树形归约累加到 partial[0]，
 *   由 thread-0 读出并返回（所有线程返回相同值，因为 partial[0] 对全 block 可见）。
 *
 * float4 加速（dim 为 4 的倍数时）：
 *   每次 ldg 加载 4 个 float（128-bit），减少内存事务数约 4×，
 *   同时编译器可以 issue 更大的向量化读指令，提升 memory-level parallelism。
 *   非对齐维度（dim % 4 ≠ 0）退化为逐维 scalar 路径，正确性保证。
 *
 * 同步要求：
 *   写 partial[] 后必须 __syncthreads()，再做树形归约；
 *   每轮归约后必须 __syncthreads()，确保下一轮读写不越界。
 *   调用方（search_kernel）在每次 block_l2 调用前已通过 __syncthreads() 对齐。
 *
 * 对应 cuVS：search_single_cta.cuh → compute_distance_to_child。
 */
__device__ float block_l2(const float* dataset, const float* queries,
                           int point, int query, int dim)
{
  __shared__ float partial[kBlockThreads];
  float sum = 0.0f;
  const float* q = queries  + (long long)query * dim;
  const float* x = dataset  + (long long)point * dim;

  if ((dim & 3) == 0) {
    // float4 向量化路径：每次处理 4 个维度，减少内存事务。
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < dim / 4; i += blockDim.x) {
      float4 a = q4[i], b = x4[i];
      float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z, dw = a.w-b.w;
      sum += dx*dx + dy*dy + dz*dz + dw*dw;
    }
  } else {
    // Scalar fallback，处理任意维度。
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float diff = q[d] - x[d]; sum += diff * diff;
    }
  }

  // 树形归约：将所有线程的局部和累积到 partial[0]。
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  return partial[0];  // 全 block 均可读；调用方只在 thread-0 使用。
}

// ── Shared-memory open-addressing hash (visited set) ─────────────────────────
/*
 * 基于共享内存的开放定址哈希表，用作 beam search 的"已访问集合"。
 *
 * 目的：
 *   beam search 展开邻居时，同一节点可能被多条路径重复发现。
 *   用哈希表过滤重复候选，避免重复计算 L2 距离（距离计算占主要开销）。
 *
 * 设计选择（开放定址 vs. 链式）：
 *   - 共享内存不支持动态分配，必须静态大小。开放定址只需一块连续数组，无指针。
 *   - 线性探测（linear probe）对缓存友好；表大小 kHashSize 为 2 的幂，
 *     取模用位与 & (kHashSize-1) 代替除法。
 *
 * 表满时的策略（return true）：
 *   当所有槽都被占用时，把候选视为"已见过"（直接跳过）。
 *   这是一个 conservative approximation：可能漏掉真正新节点，
 *   但 kHashSize ≫ kInternalTopM + kGraphDegree × kMaxSearchIters（实践中不会满），
 *   且即使少量漏失也只影响 recall，不影响正确性。
 *   对应 cuVS 中 hashmap_bitlen 参数的设计理念。
 *
 * 并发安全：
 *   hash_clear 由全 block 执行（分段清零），但 hash_contains_or_insert 仅由 thread-0
 *   调用（见 search_kernel 中的 if (threadIdx.x == 0) 保护），不存在并发写冲突。
 */
__device__ void hash_clear(int* table)
{
  // 全 block 并行清零，避免 thread-0 单独清零的串行瓶颈。
  for (int i = threadIdx.x; i < kHashSize; i += blockDim.x) table[i] = -1;
  __syncthreads();
}

__device__ bool hash_contains_or_insert(int* table, int value)
{
  int h = (int)(hash_u32((unsigned)value)) & (kHashSize - 1);
  for (int p = 0; p < kHashSize; ++p) {
    int slot = (h + p) & (kHashSize - 1);
    int old  = table[slot];
    if (old == value) return true;   // 已存在
    if (old == -1)  { table[slot] = value; return false; }  // 新插入
  }
  return true;  // 表满 → 保守地视为已见
}

// ── Sorted top-M insert (thread-0 only) ──────────────────────────────────────
/*
 * 将候选节点 (cand, dist) 插入 top-M 排序列表（按距离升序）。
 *
 * 调用约束：仅由 thread-0 调用，无需同步。
 *
 * MSB 感知去重：
 *   ids 存储带 MSB parent-flag 的 packed id，strip_parent 取低 31 位后与 cand 比较。
 *   若列表中已有同 id（无论是否已展开），跳过插入——已展开节点不需重新排序。
 *
 * 插入位置：
 *   从尾部线性扫描，找到第一个 dist < dists[pos-1] 的位置向右移动。
 *   时间复杂度 O(kInternalTopM)，因为 kInternalTopM 是编译期常量（通常≤64），
 *   寄存器展开后不存在循环开销。
 *
 * 插入时 MSB 清零（ids[pos] = (unsigned)cand 而非 mark_parent）：
 *   新插入的节点尚未展开，MSB 应为 0；mark_parent 由 pick_parent 负责。
 */
__device__ void topm_insert(unsigned* ids, float* dists, int cand, float dist)
{
  // 去重：忽略已在列表中的节点（含已展开的）。
  for (int i = 0; i < kInternalTopM; ++i)
    if ((int)strip_parent(ids[i]) == cand) return;
  if (dist >= dists[kInternalTopM - 1]) return;  // 距离不优于最差项，跳过。

  int pos = kInternalTopM - 1;
  while (pos > 0 && dist < dists[pos - 1]) {
    ids[pos] = ids[pos - 1]; dists[pos] = dists[pos - 1]; --pos;
  }
  ids[pos] = (unsigned)cand; dists[pos] = dist;  // MSB=0，标记为未展开。
}

/*
 * 选取下一个待展开的父节点：扫描 top-M 找到第一个 MSB=0 的条目，
 * 标记其 MSB（防止重复展开），返回其 id；列表中无未展开节点时返回 -1。
 *
 * 收敛判据：当 pick_parent 返回 -1 时，外层迭代立即退出。
 * 这对应 CAGRA 论文中"无可扩展邻居"的终止条件，
 * 而非 top-M 列表稳定不变（两者等价，但前者更早退出）。
 * 对应 audit report SEARCH-05 的讨论。
 */
__device__ int pick_parent(unsigned* ids)
{
  for (int i = 0; i < kInternalTopM; ++i) {
    if (!is_parented(ids[i])) {
      int p = (int)strip_parent(ids[i]);
      ids[i] = mark_parent(ids[i]);
      return p;
    }
  }
  return -1;
}

// ── Unified search kernel (single-CTA and multi-CTA) ─────────────────────────
/*
 * kMultiCta=false（SingleCta 模式）：
 *   一个 block 负责一个 query，直接写 out_ids/out_dists[q * kTopK]。
 *   适合大批量查询（num_queries 足够大时 SM 占用率高）。
 *
 * kMultiCta=true（MultiCta 模式）：
 *   kMultiCtaPerQuery 个 block 共同服务一个 query，各自独立 beam search 后
 *   输出 partial result 到 out_ids[(q * kMultiCtaPerQuery + lane) * kTopK]。
 *   多路搜索后由 merge_partial_kernel 归并最终 top-k。
 *   适合小批量查询（block 少时 SM 空闲，多 CTA 提升并行度）。
 *
 * 两种模式的三处编译期差异：
 *   ① query 索引与 lane：
 *        SingleCta: q = blockIdx.x,         lane = 0
 *        MultiCta:  q = blockIdx.x / kMCPQ, lane = blockIdx.x % kMCPQ
 *   ② 随机种子混入：
 *        SingleCta: seed = q * 1009 + s * 9176
 *        MultiCta:  seed = q * 1009 + lane * 65537 + s * 9176
 *        lane 混入保证不同 CTA 对同一 query 探索不同的初始节点，提升召回率。
 *        65537 = 2^16 + 1 为质数，与其他系数不共约数，哈希分布均匀。
 *   ③ 输出偏移：
 *        SingleCta: base = q * kTopK
 *        MultiCta:  base = (q * kMultiCtaPerQuery + lane) * kTopK
 *
 * 对应 cuVS：search_single_cta.cuh / search_multi_cta.cuh → 此处统一为模板。
 *
 * 搜索过程（beam search）：
 *   1. 用 kInternalTopM 个随机种子初始化 top-M（已去重，通过哈希表过滤）。
 *   2. 每次迭代：最多选 sw 个未展开节点作为父节点，展开其全部出边（共 kGraphDegree 条）。
 *   3. 每条出边先查哈希表去重，再由全 block 并行计算 L2 距离，thread-0 负责插入 top-M。
 *   4. 迭代至 pick_parent 返回 -1（无可展开节点）或达到 max_iters。
 *   5. thread-0 将 top-k 写回输出数组（strip_parent 去除 MSB 标志）。
 */
template <bool kMultiCta>
__global__ void search_kernel(const float* dataset, const int* graph, const float* queries,
                               int n, int dim, int num_queries, int max_iters, int sw,
                               int* out_ids, float* out_dists)
{
  const int q        = kMultiCta ? blockIdx.x / kMultiCtaPerQuery : blockIdx.x;
  const int cta_lane = kMultiCta ? blockIdx.x % kMultiCtaPerQuery : 0;
  if (q >= num_queries) return;

  __shared__ unsigned top_ids[kInternalTopM];
  __shared__ float    top_dists[kInternalTopM];
  __shared__ int      htable[kHashSize];
  __shared__ int      cand_shared;    // thread-0 写，全 block 读（经 __syncthreads 对齐）
  __shared__ int      parent_shared;  // 同上

  hash_clear(htable);
  if (threadIdx.x == 0)
    for (int i = 0; i < kInternalTopM; ++i) { top_ids[i] = (unsigned)-1; top_dists[i] = CUDART_INF_F; }
  __syncthreads();

  // 阶段 1：随机种子填充 top-M，lane 混入保证多 CTA 多样性。
  for (int s = 0; s < kInternalTopM; ++s) {
    if (threadIdx.x == 0) {
      unsigned seed = q * 1009U + (kMultiCta ? cta_lane * 65537U : 0U) + s * 9176U;
      int c = (int)(hash_u32(seed) % (unsigned)n);
      // 哈希表去重：同一 query 的不同种子可能碰撞到同一节点。
      cand_shared = hash_contains_or_insert(htable, c) ? -1 : c;
    }
    __syncthreads();
    // 全 block 并行计算 L2；cand_shared < 0 时返回 INF（thread-0 不插入）。
    float dist = cand_shared >= 0 ? block_l2(dataset, queries, cand_shared, q, dim) : CUDART_INF_F;
    if (threadIdx.x == 0 && cand_shared >= 0) topm_insert(top_ids, top_dists, cand_shared, dist);
    __syncthreads();
  }

  // 阶段 2：Beam search 主循环。
  for (int iter = 0; iter < max_iters; ++iter) {
    int expanded = 0;
    // 每次迭代展开最多 sw 个父节点（search width）。
    for (int w = 0; w < sw; ++w) {
      if (threadIdx.x == 0) parent_shared = pick_parent(top_ids);
      __syncthreads();
      if (parent_shared < 0) break;  // top-M 中无未展开节点，提前退出。
      ++expanded;

      // 展开父节点的全部 kGraphDegree 条出边。
      for (int e = 0; e < kGraphDegree; ++e) {
        if (threadIdx.x == 0) {
          int c = graph[parent_shared * kGraphDegree + e];
          // 合法性检查 + 哈希去重：-1 或越界节点跳过，已访问节点跳过。
          cand_shared = (c < 0 || c >= n || hash_contains_or_insert(htable, c)) ? -1 : c;
        }
        __syncthreads();
        float dist = cand_shared >= 0 ? block_l2(dataset, queries, cand_shared, q, dim) : CUDART_INF_F;
        if (threadIdx.x == 0 && cand_shared >= 0) topm_insert(top_ids, top_dists, cand_shared, dist);
        __syncthreads();
      }
    }
    if (expanded == 0) break;  // 本轮 sw 个槽全部返回 -1，搜索收敛。
  }

  // 阶段 3：写回结果（仅 thread-0），去除 MSB parent 标志。
  if (threadIdx.x == 0) {
    int base = kMultiCta ? (q * kMultiCtaPerQuery + cta_lane) * kTopK : q * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      out_ids  [base + k] = (int)strip_parent(top_ids[k]);
      out_dists[base + k] = top_dists[k];
    }
  }
}

// ── Merge partial results from kMultiCtaPerQuery CTAs into final top-k ────────
/*
 * 将 MultiCta 模式下每个 query 的 kMultiCtaPerQuery 份 partial top-k 归并为最终 top-k。
 *
 * 算法：对 query q 的每一个 lane（0..kMultiCtaPerQuery-1），读取其 kTopK 个结果，
 *   逐一尝试插入全局 top-k 排序列表。插入前去重（同一节点可能被多 lane 发现）。
 *
 * 时间复杂度：O(kMultiCtaPerQuery × kTopK²），因为每次插入需要去重扫描。
 *   kTopK 和 kMultiCtaPerQuery 均为编译期小常量，展开后无显著开销。
 *
 * 一线程一 query 的并行策略：
 *   每个线程独立处理一个 query，不存在 block 内协作——此 kernel 本身是简单的 CPU 归并
 *   搬到 GPU 上执行，避免一次 device→host 传输。
 */
__global__ void merge_partial_kernel(const int* partial_ids, const float* partial_dists,
                                     int num_queries, int* out_ids, float* out_dists)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;

  int   ids[kTopK];
  float dists[kTopK];
  for (int k = 0; k < kTopK; ++k) { ids[k] = -1; dists[k] = CUDART_INF_F; }

  for (int lane = 0; lane < kMultiCtaPerQuery; ++lane) {
    int base = (q * kMultiCtaPerQuery + lane) * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      int c = partial_ids[base + k]; float d = partial_dists[base + k];
      // 去重：不同 lane 可能搜到同一节点。
      bool dup = false;
      for (int j = 0; j < kTopK; ++j) if (ids[j] == c) { dup = true; break; }
      if (dup || d >= dists[kTopK - 1]) continue;
      int pos = kTopK - 1;
      while (pos > 0 && d < dists[pos - 1]) { ids[pos] = ids[pos-1]; dists[pos] = dists[pos-1]; --pos; }
      ids[pos] = c; dists[pos] = d;
    }
  }

  for (int k = 0; k < kTopK; ++k) {
    out_ids  [q * kTopK + k] = ids[k];
    out_dists[q * kTopK + k] = dists[k];
  }
}

// ─────────────────────────────────────────────────────────────────────────────

void search_engineered(const float* d_dataset, const int* d_graph, const float* d_queries,
                       int n, int dim, int num_queries, SearchPlan plan,
                       int* d_out_ids, float* d_out_dists)
{
  SearchAlgo algo    = choose_algo(plan.algo, num_queries);
  int max_iters      = plan.max_iterations > 0 ? plan.max_iterations : kMaxSearchIters;
  int sw             = plan.search_width   > 0 ? plan.search_width   : kSingleCtaSearchWidth;

  if (algo == SearchAlgo::SingleCta) {
    search_kernel<false><<<num_queries, kBlockThreads>>>(
      d_dataset, d_graph, d_queries, n, dim, num_queries,
      max_iters, sw, d_out_ids, d_out_dists);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // MultiCta：为每个 query 分配 kMultiCtaPerQuery 个 block，需要临时 partial 缓冲区。
  int partial_count = num_queries * kMultiCtaPerQuery * kTopK;
  int*   d_partial_ids   = nullptr;
  float* d_partial_dists = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partial_ids,   sizeof(int)   * partial_count));
  CUDA_CHECK(cudaMalloc(&d_partial_dists, sizeof(float) * partial_count));

  search_kernel<true><<<num_queries * kMultiCtaPerQuery, kBlockThreads>>>(
    d_dataset, d_graph, d_queries, n, dim, num_queries,
    max_iters, sw, d_partial_ids, d_partial_dists);
  CUDA_CHECK(cudaGetLastError());

  dim3 block(kBlockThreads), grid((num_queries + kBlockThreads - 1) / kBlockThreads);
  merge_partial_kernel<<<grid, block>>>(
    d_partial_ids, d_partial_dists, num_queries, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_partial_ids));
  CUDA_CHECK(cudaFree(d_partial_dists));
}

}  // namespace cagra_repro::engineered
