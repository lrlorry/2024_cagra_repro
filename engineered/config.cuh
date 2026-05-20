#pragma once

namespace cagra_repro::engineered {

// cuVS nn_descent_params(128) 实际工作宽度：
//   graph_degree=128 → extended = roundUp32(128×1.3) = 192
//   intermediate_degree=192 → extended = roundUp32(192×1.3) = 256
//   max_iterations=20（cuVS nn_descent::index_params 默认值）
constexpr int kInitialDegree = 192;  // cuVS: extended intermediate_graph_degree≈192
constexpr int kGraphDegree = 64;     // cuVS: graph_degree=64（最终剪枝目标）
constexpr int kNnDescentIters = 20;  // cuVS: max_iterations=20
constexpr int kTopK = 10;
constexpr int kInternalTopM = 64;
constexpr int kSingleCtaSearchWidth = 2;
constexpr int kMaxSearchIters = 64;
constexpr int kBlockThreads = 128;
// sm_89 shared memory 上限 48KB，kHashSize=16384(64KB) 超限。
// 8192 覆盖到 max_iters=32（64+2×64×32=4160），足够 Pareto 实验。
constexpr int kHashSize = 8192;
constexpr int kMultiCtaPerQuery = 4;

constexpr unsigned kParentMask = 0x80000000U;
constexpr unsigned kIdMask = 0x7fffffffU;

}  // namespace cagra_repro::engineered

