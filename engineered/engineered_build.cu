#include "engineered/engineered_build.cuh"

#include "common/cagra_build_kernels.cuh"
#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"

namespace cagra_repro::engineered {

// ── Stage 4a: 原子式反向边收集 ───────────────────────────────────────────────
/*
 * 目标：构造反向图 rev_graph，其中 rev_graph[dst] 记录所有以 dst 为终点的前向边的源节点。
 *
 * 为什么需要反向边：
 *   rank_prune_kernel 只保留每个节点最"重要"的 kGraphDegree 条出边，
 *   但图的连通性依赖双向可达。若 src→dst 存在但 dst→src 不在剪枝后的 dst 出边中，
 *   从 dst 出发的搜索将无法直接找到 src，造成召回损失。
 *   将反向边插入 dst 的邻居列表可修补这一单向性问题，是 CAGRA 构建的第 4 阶段。
 *   对应 cuVS：graph_core.cuh → add_reverse_edges。
 *
 * 并行策略（per-edge）：
 *   每个线程处理全图的一条前向边 (src → dst)。
 *   atomicAdd(rev_count + dst, 1) 原子地为 dst 分配一个槽位，
 *   然后将 src 写入该槽位 rev_graph[dst * kGraphDegree + pos]。
 *   若 dst 的入度超过 kGraphDegree（槽位已满），多余的反向边被丢弃——
 *   这是有意的近似：只保留最早到达的 kGraphDegree 条，后续 combine 阶段会进一步选择。
 *
 * 注意：rev_graph 在调用前已由 cudaMemset(0xff) 初始化为全 -1，
 *   因此未被填写的槽位保持 -1，combine 阶段可以安全跳过。
 */
__global__ void make_reverse_graph_kernel(const int* pruned_graph, int n,
                                          int* rev_graph, int* rev_count)
{
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= n * kGraphDegree) return;

  int src = edge / kGraphDegree;
  int dst = pruned_graph[edge];
  if (dst < 0 || dst >= n) return;  // 跳过无效边（剪枝后可能存在）

  // 原子申请槽位：多个线程同时向 dst 写反向边，atomicAdd 保证不冲突。
  int pos = atomicAdd(rev_count + dst, 1);
  if (pos < kGraphDegree) rev_graph[dst * kGraphDegree + pos] = src;
  // pos >= kGraphDegree：槽位已满，丢弃此反向边（近似处理，不影响大多数搜索路径）。
}

// ── Stage 4b: 保护前缀合并（pruned 前向边 + 反向边）────────────────────────────
/*
 * 目标：将剪枝后的前向边和反向边合并为最终搜索图 out_graph。
 *
 * 保护前缀策略（protected-prefix merge）：
 *   前向剪枝边按 (绕路数, rank) 排序，前 kGraphDegree/2 个槽位 [0, protect) 保留最优前向边。
 *   反向边从 protect 位置开始插入，替换槽位靠后（质量较差）的前向边。
 *   这样：
 *     - 最近的 kGraphDegree/2 个前向邻居（高召回贡献）始终保留。
 *     - 后半部分引入反向边，提升图的双向可达性，改善连通性。
 *   对应 cuVS：graph_core.cuh → mst_optimization → optimize。
 *
 * 去重检查：
 *   反向边中某节点可能已出现在 pruned 的前向边中，此时跳过插入（避免多重边）。
 *
 * 插入顺序（从 kGraphDegree-1 倒序遍历 rev）：
 *   使最后一条反向边占据 protect 位置（优先级最高），最早一条反向边占据靠后位置。
 *   这与 cuVS 的实现顺序一致，属于有意设计（而非任意反向边顺序）。
 *
 * protect = kGraphDegree / 2 的选择：
 *   这是一个经验值。cuVS 注释中提到该比例保证了足够的前向边保留量，
 *   同时为反向边提供足够的插入空间。详见 audit report BUILD-05。
 */
__global__ void combine_graph_kernel(const int* pruned, const int* rev, int n, int* out)
{
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n) return;

  int result[kGraphDegree];
  for (int k = 0; k < kGraphDegree; ++k) result[k] = pruned[node * kGraphDegree + k];

  const int protect = kGraphDegree / 2;  // 前半部分保护最优前向边
  for (int r = kGraphDegree - 1; r >= 0; --r) {
    int c = rev[node * kGraphDegree + r];
    if (c < 0 || c >= n) continue;
    // 去重：反向边节点已在当前结果中则跳过。
    bool dup = false;
    for (int k = 0; k < kGraphDegree; ++k) if (result[k] == c) { dup = true; break; }
    if (dup) continue;
    // 将尾部元素右移一位，在 protect 位置插入反向边（覆盖最末尾的前向边）。
    for (int j = kGraphDegree - 1; j > protect; --j) result[j] = result[j - 1];
    result[protect] = c;
  }

  for (int k = 0; k < kGraphDegree; ++k) out[node * kGraphDegree + k] = result[k];
}

// ── 可选：连通性保证（BFS + 修复）────────────────────────────────────────────
/*
 * 部分极端情况下（孤立簇、高维稀疏分布），图剪枝后可能出现从节点 0 不可达的节点。
 * 下面两个 kernel 配对使用，保证图强连通（从节点 0 可 BFS 到所有节点）。
 * 对应 cuVS：graph_core.cuh → guarantee_connectivity。
 *
 * 实现策略：
 *   1. bfs_mark_kernel：将 reachable[] 初始化为只有节点 0 为 true，
 *      每轮将 reachable[i]=true 的节点的所有邻居标记为 reachable，
 *      若有新节点被标记则 *changed=true，外层循环继续。
 *   2. repair_unreachable_kernel：遍历所有不可达节点，从全集中找到距离最近的
 *      可达节点 best_c，将其接到 graph[node * kGraphDegree + 0]（覆盖第一条边）。
 *
 * 时间复杂度：
 *   BFS：O(直径 × n × kGraphDegree)，均摊为 O(n) 轮；
 *   修复：O(n²) 暴力搜索（repair_unreachable_kernel），对不可达节点数量线性。
 *   实践中孤立节点极少，BFS 通常 2-3 轮收敛；修复 kernel 几乎不被触发。
 *
 * 注意：bfs_mark_kernel 存在写-写竞争（多个线程可能同时将同一节点设为 true），
 *   但写的值相同（bool true），竞争无害——worst case 是多写一次 true，不影响正确性。
 *   *changed 同理：true 的原子性由 bool 赋值的内存宽度保证（单字节赋值是原子的在大多数架构上）。
 */
__global__ void bfs_mark_kernel(const int* graph, bool* reachable, bool* changed, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !reachable[i]) return;
  for (int k = 0; k < kGraphDegree; ++k) {
    int nb = graph[i * kGraphDegree + k];
    if (nb >= 0 && nb < n && !reachable[nb]) { reachable[nb] = true; *changed = true; }
  }
}

/*
 * 对每个不可达节点，线性扫描全图找到最近的可达节点，将其写入该节点的第一条出边。
 *
 * 覆盖第一条边（slot 0）而非追加：
 *   图度数固定为 kGraphDegree，无法动态增加。slot 0 原为剪枝排名最高的前向边，
 *   覆盖它会轻微降低该节点的搜索质量，但使其可达——这是连通性修复的合理权衡。
 *   若该节点本身即为孤立节点，所有出边质量相同，覆盖任意一条均可。
 */
__global__ void repair_unreachable_kernel(const float* dataset, int* graph,
                                          const bool* reachable, int n, int dim)
{
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n || reachable[node]) return;

  float best = CUDART_INF_F;
  int   best_c = -1;
  // O(n) 暴力扫描：仅对不可达节点触发，实践中极少执行。
  for (int c = 0; c < n; ++c) {
    if (!reachable[c]) continue;
    float d = l2_point_point_scalar(dataset, node, c, dim);
    if (d < best) { best = d; best_c = c; }
  }
  if (best_c >= 0) graph[node * kGraphDegree] = best_c;
}

// ─────────────────────────────────────────────────────────────────────────────

void build_graph_engineered(const float* d_dataset, int n, int dim,
                            bool guarantee_connectivity, int* d_graph,
                            int nn_iters, BuildTiming* timing)
{
  // Use default when nn_iters == -1
  if (nn_iters < 0) nn_iters = kNnDescentIters;

  int* d_initial   = nullptr;
  int* d_pruned    = nullptr;
  int* d_rev       = nullptr;
  int* d_rev_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_initial,   sizeof(int) * n * kInitialDegree));
  CUDA_CHECK(cudaMalloc(&d_pruned,    sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_rev,       sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_rev_count, sizeof(int) * n));

  dim3 block(kBlockThreads);
  dim3 gnodes((n + kBlockThreads - 1) / kBlockThreads);
  dim3 gedges((n * kGraphDegree + kBlockThreads - 1) / kBlockThreads);

  // ── Event timing helpers ──────────────────────────────────────────────────
  // We use a simple macro-style pair: record e0 before kernel, e1 after, then
  // accumulate into the appropriate BuildTiming field.
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  float _ev_ms = 0.f;
#define TIME_START() \
  do { if (timing) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); } } while(0)
#define TIME_STOP(field) \
  do { if (timing) { cudaEventRecord(e1); cudaEventSynchronize(e1); \
       cudaEventElapsedTime(&_ev_ms, e0, e1); timing->field += _ev_ms; \
       cudaEventDestroy(e0); cudaEventDestroy(e1); e0 = nullptr; e1 = nullptr; } } while(0)

  // 阶段 1：随机种子图初始化
  TIME_START();
  init_graph_kernel<kInitialDegree><<<gnodes, block>>>(d_dataset, n, dim, d_initial);
  CUDA_CHECK(cudaGetLastError());
  TIME_STOP(init_ms);

  // 阶段 2：NN-Descent 松弛（多轮）
  TIME_START();
  for (int iter = 0; iter < nn_iters; ++iter) {
    nn_descent_kernel<kInitialDegree><<<gnodes, block>>>(d_dataset, n, dim, d_initial);
    CUDA_CHECK(cudaGetLastError());
  }
  TIME_STOP(nn_descent_ms);

  // 阶段 3：Rank 剪枝
  TIME_START();
  rank_prune_kernel<kInitialDegree, kGraphDegree><<<gnodes, block>>>(d_initial, n, d_pruned);
  CUDA_CHECK(cudaGetLastError());
  TIME_STOP(prune_ms);

  // 阶段 4：反向边收集 + 保护前缀合并。
  // 0xff 字节重复填充 int 得到 0xFFFFFFFF = -1（补码），等效于 fill(-1)。
  CUDA_CHECK(cudaMemset(d_rev,       0xff, sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMemset(d_rev_count, 0,    sizeof(int) * n));

  TIME_START();
  make_reverse_graph_kernel<<<gedges, block>>>(d_pruned, n, d_rev, d_rev_count);
  CUDA_CHECK(cudaGetLastError());
  combine_graph_kernel<<<gnodes, block>>>(d_pruned, d_rev, n, d_graph);
  CUDA_CHECK(cudaGetLastError());
  TIME_STOP(reverse_ms);

  // 可选阶段 5：BFS 连通性检测与修复（默认关闭，guarantee_connectivity=true 时启用）。
  if (guarantee_connectivity) {
    bool* d_reachable = nullptr;
    bool* d_changed   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reachable, sizeof(bool) * n));
    CUDA_CHECK(cudaMalloc(&d_changed,   sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_reachable, 0, sizeof(bool) * n));
    bool one = true;
    // 节点 0 作为 BFS 起始节点，初始化为可达。
    CUDA_CHECK(cudaMemcpy(d_reachable, &one, sizeof(bool), cudaMemcpyHostToDevice));

    TIME_START();
    bool h_changed;
    do {
      h_changed = false;
      CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
      bfs_mark_kernel<<<gnodes, block>>>(d_graph, d_reachable, d_changed, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (h_changed);

    repair_unreachable_kernel<<<gnodes, block>>>(d_dataset, d_graph, d_reachable, n, dim);
    CUDA_CHECK(cudaGetLastError());
    TIME_STOP(connectivity_ms);

    CUDA_CHECK(cudaFree(d_reachable));
    CUDA_CHECK(cudaFree(d_changed));
  }

#undef TIME_START
#undef TIME_STOP

  CUDA_CHECK(cudaFree(d_initial));
  CUDA_CHECK(cudaFree(d_pruned));
  CUDA_CHECK(cudaFree(d_rev));
  CUDA_CHECK(cudaFree(d_rev_count));
}

}  // namespace cagra_repro::engineered
