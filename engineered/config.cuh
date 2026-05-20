#pragma once

namespace cagra_repro::engineered {

constexpr int kInitialDegree = 128;  // cuVS: intermediate_graph_degree=128
constexpr int kGraphDegree = 64;     // cuVS: graph_degree=64
constexpr int kNnDescentIters = 8;
constexpr int kTopK = 10;
constexpr int kInternalTopM = 256;
constexpr int kSingleCtaSearchWidth = 2;
constexpr int kMaxSearchIters = 32;
constexpr int kBlockThreads = 128;
constexpr int kHashSize = 1024;
constexpr int kMultiCtaPerQuery = 4;

constexpr unsigned kParentMask = 0x80000000U;
constexpr unsigned kIdMask = 0x7fffffffU;

}  // namespace cagra_repro::engineered

