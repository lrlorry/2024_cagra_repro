#pragma once
/*
 * cagra_build_kernels.cuh — 图构建阶段的共享 GPU kernel
 *
 * 这四个 kernel 对应 CAGRA 论文（Ootomo et al., ICDE 2024）中图构建的
 * 前三个阶段，被 plain/ 和 engineered/ 两个实现共享，通过模板参数
 * IDEG（初始图度数）和 GDEG（输出图度数）适配各自的 config.cuh 常量，
 * 不重复定义。
 *
 * 调用方式：
 *   #include "common/cagra_build_kernels.cuh"
 *   init_graph_kernel<kInitialDegree><<<grid, block>>>(...);
 *   nn_descent_kernel<kInitialDegree><<<grid, block>>>(...);
 *   rank_prune_kernel<kInitialDegree, kGraphDegree><<<grid, block>>>(...);
 */

#include "cagra_common.cuh"
#include <climits>
#include <math_constants.h>

namespace cagra_repro {

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1 — 随机种子图初始化
// ═══════════════════════════════════════════════════════════════════════════════
/*
 * 目标：为每个节点 src 构造一个长度为 IDEG 的候选邻居列表，按 L2 距离排序。
 *
 * 策略：
 *   ① 通过 hash 散列生成 IDEG×8 个随机候选节点，保留距离最小的 IDEG 个。
 *      乘以 8 是为了让初始图质量足够高，同时保持 O(IDEG) 的工作量。
 *      hash 函数以 (src, t) 为输入，保证不同节点采样路径不同，避免系统性偏差。
 *   ② 若随机采样后仍有空槽（极小数据集），线性顺序填充保证列表满度。
 *
 * 对应 cuVS：cagra_build.cuh → nn_descent 路径的初始图创建。
 *
 * 内存布局：graph[src * IDEG + k] = 第 k 近邻的节点 id。
 */
template <int IDEG>
__global__ void init_graph_kernel(const float* dataset, int n, int dim, int* graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  // 寄存器内维护排序列表，避免反复写全局内存。
  int   ids[IDEG];
  float dists[IDEG];
  for (int i = 0; i < IDEG; ++i) { ids[i] = -1; dists[i] = CUDART_INF_F; }

  // ① 随机采样：hash(src * 4099 + t * 131) 两个质数乘子使散列分布均匀。
  for (int t = 0; t < IDEG * 8; ++t) {
    int c = (int)(hash_u32(src * 4099U + t * 131U) % (unsigned)n);
    if (c != src)
      insert_sorted_unique<IDEG>(ids, dists, c, l2_point_point_scalar(dataset, src, c, dim));
  }

  // ② 线性回退：当数据集极小或 hash 大量碰撞时保证列表满度。
  for (int t = 0; t < n && ids[IDEG - 1] < 0; ++t) {
    int c = (src + t + 1) % n;
    if (c != src)
      insert_sorted_unique<IDEG>(ids, dists, c, l2_point_point_scalar(dataset, src, c, dim));
  }

  for (int k = 0; k < IDEG; ++k) graph[src * IDEG + k] = ids[k];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 2 — NN-Descent 松弛迭代
// ═══════════════════════════════════════════════════════════════════════════════
/*
 * 核心思想（Dong et al., 2011）：
 *   "如果 B 是 A 的近邻，那么 B 的近邻也很可能是 A 的近邻。"
 *   每次迭代，对 src 当前每条邻居边 (src→nb)，把 nb 的所有邻居作为
 *   src 的候选，计算距离后尝试插入 src 的列表。
 *
 * 与标准 NN-Descent 的差异：
 *   - 标准版本维护 new/old 两套邻居集合并区分正向/逆向检查。
 *   - 此版本简化为单向扫描（src 的邻居 → 邻居的邻居），是论文在 GPU 上
 *     的近似实现，牺牲收敛精度换取 kernel 的简洁性和并行度。
 *   - 无 delta-stopping：迭代次数固定为 kNnDescentIters，由调用方控制。
 *
 * 数据竞争说明：
 *   多个线程可能同时读写同一节点的邻居行（当两个节点互为邻居时）。
 *   此处不加锁，是 CAGRA 论文认可的 GPU 近似 NN-Descent 做法：
 *   竞态写入最多使某次更新丢失，不影响正确性，只影响收敛速度。
 *
 * 对应 cuVS：graph_core.cuh → nn_descent_build / relax。
 */
template <int IDEG>
__global__ void nn_descent_kernel(const float* dataset, int n, int dim, int* graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  // 从全局内存加载当前邻居列表到寄存器，计算好距离备用。
  int   ids[IDEG];
  float dists[IDEG];
  for (int k = 0; k < IDEG; ++k) {
    int c    = graph[src * IDEG + k];
    ids[k]   = c;
    dists[k] = (c >= 0) ? l2_point_point_scalar(dataset, src, c, dim) : CUDART_INF_F;
  }

  // 遍历每条当前邻边 src→nb，把 nb 的全部邻居作为候选。
  for (int nb = 0; nb < IDEG; ++nb) {
    int d = graph[src * IDEG + nb];  // 邻居节点 D
    if (d < 0 || d >= n) continue;
    for (int sr = 0; sr < IDEG; ++sr) {
      int c = graph[d * IDEG + sr];  // D 的邻居 C，作为 src 的候选
      if (c < 0 || c == src) continue;
      insert_sorted_unique<IDEG>(ids, dists, c, l2_point_point_scalar(dataset, src, c, dim));
    }
  }

  // 将更新后的列表写回全局内存。
  for (int k = 0; k < IDEG; ++k) graph[src * IDEG + k] = ids[k];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3 辅助 — 单条边的二跳绕路计数
// ═══════════════════════════════════════════════════════════════════════════════
/*
 * 定义：对于节点 src 的第 cand_rank 条边（指向 cand），"绕路数"是指
 *   有多少个排名更靠前的邻居 D（rank < cand_rank），其邻居列表中包含 cand。
 *   即存在路径 src→D→cand，而 D 比 cand 离 src 更近。
 *
 * 直觉：若 cand 已经可以通过更近的邻居间接到达，则直连边 src→cand 冗余度高，
 *   可以优先被裁剪。绕路数越多，该边越不重要。
 *
 * 与 HNSW/NSG 的 RNG 剪枝的区别（参见 audit report BUILD-03）：
 *   - RNG 剪枝：判断 d(D,cand) < d(src,cand) 是否成立（几何距离判据）。
 *   - 本实现：仅检查 D 的邻居列表是否包含 cand（拓扑判据，不计算距离）。
 *   - CAGRA 论文明确选择 rank-based 方案以避免额外的距离计算开销，属于有意设计。
 *
 * 关于 kDB < kAB 的注释（audit report BUILD-04）：
 *   cuVS 源码 graph_core.cuh:201 有被注释掉的条件 `if (kDB < kAB)`，
 *   若启用则要求 D→cand 边的 rank 也小于 cand_rank（双向 rank 约束）。
 *   当前实现与 cuVS GPU kernel 一致：D 的邻居列表中任意 rank 包含 cand 即计数。
 *
 * 对应 cuVS：graph_core.cuh → kern_prune 内的内层循环逻辑。
 */
template <int IDEG>
__device__ int count_detours(const int* graph, int src, int cand_rank, int cand)
{
  int count = 0;
  // 遍历所有排名更靠前（更近）的邻居 D。
  for (int p = 0; p < cand_rank; ++p) {
    int d = graph[src * IDEG + p];
    if (d < 0) continue;
    // 检查 D 的邻居列表中是否包含 cand。
    for (int k = 0; k < IDEG; ++k)
      if (graph[d * IDEG + k] == cand) { ++count; break; }
  }
  return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3 — Rank-based 图剪枝
// ═══════════════════════════════════════════════════════════════════════════════
/*
 * 目标：将 IDEG 度的初始 kNN 图压缩为 GDEG 度的输出图，保留最"重要"的边。
 *
 * 算法（对每个节点 src）：
 *   1. 遍历初始图中 src 的全部 IDEG 条边，计算每条边的绕路数 d。
 *   2. 维护一个长度为 GDEG 的排序列表，主键为 (d, rank)，越小越优先。
 *      即：绕路数少 → 直连价值高；绕路数相同时，距离更近（rank 更小）的优先。
 *   3. 每遇到一条"比当前列表最末项更好"的边，插入并保持排序。
 *   4. 若剪枝后某槽仍为空（数据集过小或自环过多），用顺序回退节点填充，
 *      确保每个节点始终拥有满度的出边（对搜索连通性至关重要）。
 *
 * 内存布局：
 *   输入  in_graph [n × IDEG]  — 已排序的初始 kNN 图（距离从近到远）
 *   输出 out_graph [n × GDEG]  — 剪枝后的搜索图
 *
 * 对应 cuVS：graph_core.cuh → kern_prune + 后续 CPU 端的 optimize 函数。
 */
template <int IDEG, int GDEG>
__global__ void rank_prune_kernel(const int* in_graph, int n, int* out_graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  // 输出列表：ids[k]、det[k]（绕路数）、rnk[k]（原始 rank）。
  // 初始化为"最差"，以便任何真实边都能插入。
  int ids[GDEG], det[GDEG], rnk[GDEG];
  for (int i = 0; i < GDEG; ++i) { ids[i] = -1; det[i] = INT_MAX; rnk[i] = INT_MAX; }

  for (int rank = 0; rank < IDEG; ++rank) {
    int c = in_graph[src * IDEG + rank];
    if (c < 0 || c == src) continue;  // 跳过无效节点和自环

    int d = count_detours<IDEG>(in_graph, src, rank, c);

    // 找到第一个"比候选边更差"的位置，将候选插入该位置。
    int pos = -1;
    for (int i = 0; i < GDEG; ++i)
      if (d < det[i] || (d == det[i] && rank < rnk[i])) { pos = i; break; }
    if (pos < 0) continue;  // 候选比列表中最差项还差，跳过

    // 向右移动 pos 之后的项，为候选腾出空间。
    for (int j = GDEG - 1; j > pos; --j) {
      ids[j] = ids[j - 1]; det[j] = det[j - 1]; rnk[j] = rnk[j - 1];
    }
    ids[pos] = c; det[pos] = d; rnk[pos] = rank;
  }

  // 写回输出图；空槽用顺序回退节点填充，保证图度数满。
  for (int k = 0; k < GDEG; ++k) {
    if (ids[k] < 0) ids[k] = (src + k + 1) % n;
    out_graph[src * GDEG + k] = ids[k];
  }
}

}  // namespace cagra_repro
