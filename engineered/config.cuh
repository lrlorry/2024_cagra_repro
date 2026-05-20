#pragma once

namespace cagra_repro::engineered {

constexpr int kInitialDegree = 128;  // cuVS: intermediate_graph_degree=128
constexpr int kGraphDegree = 64;     // cuVS: graph_degree=64
constexpr int kNnDescentIters = 8;
constexpr int kTopK = 10;
constexpr int kInternalTopM = 64;
constexpr int kSingleCtaSearchWidth = 2;
constexpr int kMaxSearchIters = 64;
constexpr int kBlockThreads = 128;
// cuVS 动态计算: itopk_size + search_width * graph_degree * max_iterations
// = 64 + 2 * 64 * 64 = 8256 → 取下一个 2 的幂
constexpr int kHashSize = 16384;
constexpr int kMultiCtaPerQuery = 4;

constexpr unsigned kParentMask = 0x80000000U;
constexpr unsigned kIdMask = 0x7fffffffU;

}  // namespace cagra_repro::engineered

