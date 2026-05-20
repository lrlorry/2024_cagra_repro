#include "plain/plain_build.cuh"

#include "common/cagra_build_kernels.cuh"
#include "common/cuda_utils.cuh"
#include "plain/config.cuh"

namespace cagra_repro::plain {

// ── Stage 4 (plain): O(n²) 反向边合并 ──────────────────────────────────────
/*
 * 目标：将剪枝后的前向图与其反向边合并，得到最终搜索图。
 *
 * 与 engineered/ 版本的对比：
 *   engineered 版本（make_reverse_graph_kernel + combine_graph_kernel）：
 *     - per-edge 并行，原子写 rev_count，时间复杂度 O(n × kGraphDegree)
 *     - 需要额外的 rev_graph 和 rev_count 缓冲区
 *   plain 版本（此 kernel）：
 *     - per-node 并行，每个线程扫描全图所有节点的出边，时间复杂度 O(n²)
 *     - 无需额外缓冲区，直接在 output_graph 上操作
 *     - 简单易读，是 reference 实现；不适合大规模数据集
 *
 * 保护前缀策略（与 engineered 相同）：
 *   slots [0, protect) 保留剪枝后的最优前向边，
 *   新发现的反向边从 protect 位置插入，覆盖质量较差的尾部槽位。
 *   这确保了最高质量的前向边不被反向边覆盖，同时提升图的双向连通性。
 *
 * 实现细节：
 *   对于当前目标节点 dst，扫描所有 src（0..n-1）及其所有 kGraphDegree 条出边，
 *   找到所有满足 pruned_graph[src][k] == dst 的 src（即以 dst 为终点的前向边的源节点），
 *   将 src 插入 dst 的输出列表（跳过已存在的节点，contains_id 用于去重）。
 *
 * 注意：同一个 src 只要有一条出边指向 dst，就在 dst 的列表中插入一次 src。
 *   inner for-k 循环找到第一条匹配边后不 break，但 contains_id 保证只插入一次。
 *   （若 src 有多条边指向 dst，这在剪枝后不会发生；此处是保守处理。）
 */
__global__ void slow_reverse_merge_kernel(const int* pruned_graph, int n, int* output_graph)
{
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= n) return;

  int out[kGraphDegree];
  for (int k = 0; k < kGraphDegree; ++k) out[k] = pruned_graph[dst * kGraphDegree + k];

  const int protect = kGraphDegree / 2;  // 前半部分保护最优剪枝前向边

  // 扫描全图，收集所有以 dst 为终点的前向边的源节点作为反向边。
  for (int src = 0; src < n; ++src) {
    for (int k = 0; k < kGraphDegree; ++k) {
      if (pruned_graph[src * kGraphDegree + k] != dst) continue;
      if (contains_id<kGraphDegree>(out, src)) continue;  // 已存在则跳过
      // 将尾部元素右移，在 protect 位置插入反向边（覆盖最末尾的前向边）。
      for (int j = kGraphDegree - 1; j > protect; --j) out[j] = out[j - 1];
      out[protect] = src;
    }
  }

  for (int k = 0; k < kGraphDegree; ++k) output_graph[dst * kGraphDegree + k] = out[k];
}

// ─────────────────────────────────────────────────────────────────────────────

void build_graph_plain(const float* d_dataset, int n, int dim, int* d_graph)
{
  int* d_initial = nullptr;
  int* d_pruned  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_initial, sizeof(int) * n * kInitialDegree));
  CUDA_CHECK(cudaMalloc(&d_pruned,  sizeof(int) * n * kGraphDegree));

  dim3 block(kThreads), grid((n + kThreads - 1) / kThreads);

  // 阶段 1-3：随机初始图 → NN-Descent 松弛 → Rank 剪枝（共享 kernel，见 common/）。
  init_graph_kernel<kInitialDegree><<<grid, block>>>(d_dataset, n, dim, d_initial);
  CUDA_CHECK(cudaGetLastError());

  for (int iter = 0; iter < kNnDescentIters; ++iter) {
    nn_descent_kernel<kInitialDegree><<<grid, block>>>(d_dataset, n, dim, d_initial);
    CUDA_CHECK(cudaGetLastError());
  }

  rank_prune_kernel<kInitialDegree, kGraphDegree><<<grid, block>>>(d_initial, n, d_pruned);
  CUDA_CHECK(cudaGetLastError());

  // 阶段 4：O(n²) 反向边合并（plain reference 实现）。
  slow_reverse_merge_kernel<<<grid, block>>>(d_pruned, n, d_graph);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_initial));
  CUDA_CHECK(cudaFree(d_pruned));
}

}  // namespace cagra_repro::plain
