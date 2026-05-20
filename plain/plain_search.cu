#include "plain/plain_search.cuh"

#include "common/cagra_common.cuh"
#include "common/cuda_utils.cuh"
#include "plain/config.cuh"

#include <math_constants.h>

namespace cagra_repro::plain {

// ── 单线程每 query 的标量 beam search ───────────────────────────────────────
/*
 * 实现 CAGRA beam search 的 plain reference 版本：一个线程处理一个 query。
 *
 * 与 engineered/ 版本的对比：
 *   engineered：一个 block（kBlockThreads 个线程）处理一个 query，
 *               float4 向量化 L2，shared-memory 哈希去重，MSB parent-flag。
 *   plain：     一个线程处理一个 query，
 *               scalar L2，线性扫描去重（O(kInternalTopM)），bool expanded[] 数组。
 *   前者适合高吞吐量生产，后者适合调试和正确性验证。
 *
 * top-M 数据结构：
 *   ids[kInternalTopM]：节点 id（-1 表示空）
 *   dists[kInternalTopM]：对应 L2² 距离（INF 表示空）
 *   expanded[kInternalTopM]：是否已作为父节点展开（对应 engineered 的 MSB 标志）
 *   三个数组同步维护，按距离升序排列。
 *
 * 初始化（随机种子）：
 *   kInternalTopM 个随机节点作为搜索起点，hash 函数以 (q, s) 为输入保证不同 query 不同起点。
 *   去重：线性扫描 ids[] 检查重复（kInternalTopM 较小，O(kInternalTopM²) 可接受）。
 *   插入逻辑：内联 sorted insert（与 common 中的 insert_sorted_unique 等价，
 *             但需额外维护 expanded[]，故在此处展开而不调用公共函数）。
 *
 * Beam search 主循环：
 *   每次迭代选最多 kSearchWidth 个未展开节点作为父节点（按距离优先）。
 *   对每个父节点，展开其全部 kGraphDegree 条出边：
 *     - 检查合法性（c >= 0 && c < n）
 *     - 线性扫描去重（无哈希表）
 *     - 若距离优于当前 top-M 最末项则尝试插入
 *   当一轮迭代中 done==0（无父节点可展开）时退出，等价于 engineered 的 pick_parent 返回 -1。
 *   或者达到 kMaxSearchIters 上限。
 *
 * 输出：
 *   写 out_ids/out_dists 的前 kTopK 项（top-M 的前 kTopK 即为全局最近邻结果）。
 *
 * 对应 cuVS：search_single_cta.cuh → compute_distance → sorted insert 路径（scalar 子集）。
 */
__global__ void search_plain_kernel(const float* dataset, const int* graph,
                                    const float* queries,
                                    int n, int dim, int num_queries,
                                    int* out_ids, float* out_dists)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;

  int   ids[kInternalTopM];
  float dists[kInternalTopM];
  bool  expanded[kInternalTopM];
  for (int i = 0; i < kInternalTopM; ++i) { ids[i] = -1; dists[i] = CUDART_INF_F; expanded[i] = false; }

  // 阶段 1：随机种子填充 top-M。
  for (int s = 0; s < kInternalTopM; ++s) {
    int c = (int)(hash_u32(q * 1009U + s * 9176U) % (unsigned)n);
    float d = l2_query_point_scalar(dataset, queries, c, q, dim);
    // 线性去重：plain 版本不使用哈希表。
    bool dup = false;
    for (int i = 0; i < kInternalTopM; ++i) if (ids[i] == c) { dup = true; break; }
    if (dup || d >= dists[kInternalTopM - 1]) continue;
    // 内联 sorted insert（需同步维护 expanded[]，故不调用 insert_sorted_unique）。
    int pos = kInternalTopM - 1;
    while (pos > 0 && d < dists[pos - 1]) {
      ids[pos] = ids[pos-1]; dists[pos] = dists[pos-1]; expanded[pos] = expanded[pos-1]; --pos;
    }
    ids[pos] = c; dists[pos] = d; expanded[pos] = false;
  }

  // 阶段 2：Beam search 主循环。
  for (int iter = 0; iter < kMaxSearchIters; ++iter) {
    int done = 0;
    // 从 top-M 前端选最多 kSearchWidth 个未展开节点作为父节点。
    for (int m = 0; m < kInternalTopM && done < kSearchWidth; ++m) {
      if (ids[m] < 0 || expanded[m]) continue;
      expanded[m] = true; ++done;

      int parent = ids[m];
      for (int e = 0; e < kGraphDegree; ++e) {
        int c = graph[parent * kGraphDegree + e];
        if (c < 0 || c >= n) continue;
        float d = l2_query_point_scalar(dataset, queries, c, q, dim);
        // 线性去重 + 距离剪枝。
        bool dup = false;
        for (int i = 0; i < kInternalTopM; ++i) if (ids[i] == c) { dup = true; break; }
        if (dup || d >= dists[kInternalTopM - 1]) continue;
        // 插入 top-M，同步移位 expanded[]。
        int pos = kInternalTopM - 1;
        while (pos > 0 && d < dists[pos - 1]) {
          ids[pos] = ids[pos-1]; dists[pos] = dists[pos-1]; expanded[pos] = expanded[pos-1]; --pos;
        }
        ids[pos] = c; dists[pos] = d; expanded[pos] = false;
      }
    }
    if (done == 0) break;  // 本轮无父节点可展开，搜索收敛。
  }

  for (int k = 0; k < kTopK; ++k) {
    out_ids  [q * kTopK + k] = ids[k];
    out_dists[q * kTopK + k] = dists[k];
  }
}

void search_plain(const float* d_dataset, const int* d_graph, const float* d_queries,
                  int n, int dim, int num_queries, int* d_out_ids, float* d_out_dists)
{
  dim3 block(kThreads), grid((num_queries + kThreads - 1) / kThreads);
  search_plain_kernel<<<grid, block>>>(
    d_dataset, d_graph, d_queries, n, dim, num_queries, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cagra_repro::plain
