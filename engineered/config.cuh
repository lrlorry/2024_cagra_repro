#pragma once

namespace cagra_repro::engineered {

constexpr int kInitialDegree = 32;
constexpr int kGraphDegree = 16;
constexpr int kNnDescentIters = 8;
constexpr int kTopK = 10;
constexpr int kInternalTopM = 64;
constexpr int kSingleCtaSearchWidth = 2;
constexpr int kMaxSearchIters = 32;
constexpr int kBlockThreads = 128;
constexpr int kHashSize = 256;
constexpr int kMultiCtaPerQuery = 4;

constexpr unsigned kParentMask = 0x80000000U;
constexpr unsigned kIdMask = 0x7fffffffU;

}  // namespace cagra_repro::engineered

