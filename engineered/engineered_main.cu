// CAGRA benchmark: reads SIFT-format data, sweeps max_iterations to produce
// a recall@k vs QPS Pareto curve saved as CSV.
//
// Usage:
//   ./engineered_core [--base path.fvecs] [--query path.fvecs]
//                     [--gt path.ivecs] [--n N] [--nq NQ] [--k K]
//                     [--repeats R] [--out cagra_sweep.csv]
//
// --base / --query : fvecs files (SIFT1M format).
// --gt             : precomputed ivecs ground truth (optional; falls back to
//                    CPU brute-force when omitted).
// --n / --nq       : cap on base / query vectors loaded.
// --k              : top-k (must be <= kTopK in config.cuh, default kTopK=10).

#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"
#include "engineered/engineered_build.cuh"
#include "engineered/engineered_plan.cuh"
#include "engineered/engineered_search.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

// ── fvecs / ivecs I/O ────────────────────────────────────────────────────────

std::vector<float> read_fvecs(const char* path, int* out_n, int* out_dim, int limit = 0)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

  int dim = 0;
  f.read(reinterpret_cast<char*>(&dim), 4);
  f.seekg(0, std::ios::end);
  long long total = f.tellg();
  int bpv = 4 + dim * 4;
  int n   = static_cast<int>(total / bpv);
  if (limit > 0 && limit < n) n = limit;

  f.seekg(0);
  std::vector<float> data(static_cast<long long>(n) * dim);
  for (int i = 0; i < n; ++i) {
    int d;
    f.read(reinterpret_cast<char*>(&d), 4);
    f.read(reinterpret_cast<char*>(&data[static_cast<long long>(i) * dim]), 4 * dim);
  }
  *out_n = n; *out_dim = dim;
  return data;
}

std::vector<int> read_ivecs(const char* path, int* out_n, int* out_k, int limit = 0)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

  int k = 0;
  f.read(reinterpret_cast<char*>(&k), 4);
  f.seekg(0, std::ios::end);
  long long total = f.tellg();
  int bpv = 4 + k * 4;
  int n   = static_cast<int>(total / bpv);
  if (limit > 0 && limit < n) n = limit;

  f.seekg(0);
  std::vector<int> data(static_cast<long long>(n) * k);
  for (int i = 0; i < n; ++i) {
    int kk;
    f.read(reinterpret_cast<char*>(&kk), 4);
    f.read(reinterpret_cast<char*>(&data[static_cast<long long>(i) * k]), 4 * k);
  }
  *out_n = n; *out_k = k;
  return data;
}

// ── CPU brute-force ground truth ─────────────────────────────────────────────
// Acceptable for nq ≤ 1000; for larger nq supply a precomputed --gt file.

std::vector<int> compute_gt_cpu(const float* base, int n, int dim,
                                  const float* query, int nq, int k)
{
  printf("computing GT (CPU brute-force, n=%d nq=%d dim=%d k=%d) ...\n",
         n, nq, dim, k);
  fflush(stdout);

  std::vector<int>   gt(static_cast<long long>(nq) * k);
  std::vector<float> dists(n);
  std::vector<int>   idx(n);

  for (int q = 0; q < nq; ++q) {
    const float* qv = query + static_cast<long long>(q) * dim;
    for (int i = 0; i < n; ++i) {
      const float* bv = base + static_cast<long long>(i) * dim;
      float d = 0.f;
      for (int j = 0; j < dim; ++j) { float diff = qv[j] - bv[j]; d += diff * diff; }
      dists[i] = d;
      idx[i]   = i;
    }
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b) { return dists[a] < dists[b]; });
    std::sort(idx.begin(), idx.begin() + k,
              [&](int a, int b) { return dists[a] < dists[b]; });
    for (int j = 0; j < k; ++j)
      gt[static_cast<long long>(q) * k + j] = idx[j];

    if ((q + 1) % 100 == 0) { printf("  GT %d/%d\r", q + 1, nq); fflush(stdout); }
  }
  printf("  GT done.           \n");
  return gt;
}

// ── recall@k ─────────────────────────────────────────────────────────────────

float recall_at_k(const int* result, int nq, int k, const int* gt, int gt_k)
{
  double total = 0.0;
  for (int q = 0; q < nq; ++q) {
    const int* r = result + static_cast<long long>(q) * k;
    const int* g = gt     + static_cast<long long>(q) * gt_k;
    int hit = 0;
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < gt_k; ++j)
        if (r[i] == g[j]) { ++hit; break; }
    total += static_cast<double>(hit) / k;
  }
  return static_cast<float>(total / nq);
}

} // namespace

int main(int argc, char** argv)
{
  using namespace cagra_repro::engineered;

  // ── arg parsing ──────────────────────────────────────────────────────────
  const char* base_path  = nullptr;
  const char* query_path = nullptr;
  const char* gt_path    = nullptr;
  const char* out_csv    = "cagra_sweep.csv";
  int n_limit  = 0;
  int nq_limit = 1000;
  int k        = kTopK;
  int repeats  = 5;

  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--base")    && i+1 < argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--query")   && i+1 < argc) query_path = argv[++i];
    else if (!strcmp(argv[i], "--gt")      && i+1 < argc) gt_path    = argv[++i];
    else if (!strcmp(argv[i], "--out")     && i+1 < argc) out_csv    = argv[++i];
    else if (!strcmp(argv[i], "--n")       && i+1 < argc) n_limit    = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nq")      && i+1 < argc) nq_limit   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--k")       && i+1 < argc) k          = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--repeats") && i+1 < argc) repeats    = atoi(argv[++i]);
  }

  if (k > kTopK) {
    fprintf(stderr, "k=%d > kTopK=%d; set kTopK >= k in config.cuh and recompile\n", k, kTopK);
    return 1;
  }

  // ── load data ─────────────────────────────────────────────────────────────
  int n = 0, dim = 0, nq = 0, qdim = 0;
  std::vector<float> h_base, h_query;

  if (base_path && query_path) {
    h_base  = read_fvecs(base_path,  &n,  &dim,  n_limit);
    h_query = read_fvecs(query_path, &nq, &qdim, nq_limit);
    printf("base  : n=%d dim=%d\n", n, dim);
    printf("query : nq=%d dim=%d\n", nq, qdim);
  } else {
    printf("[smoke-test] no --base/--query given; using random n=8192 dim=128 nq=200\n");
    n = 8192; dim = 128; nq = 200;
    h_base .resize(static_cast<long long>(n)  * dim);
    h_query.resize(static_cast<long long>(nq) * dim);
    srand(42);
    for (float& x : h_base)  x = static_cast<float>(rand()) / RAND_MAX;
    for (float& x : h_query) x = static_cast<float>(rand()) / RAND_MAX;
  }

  // ── ground truth ─────────────────────────────────────────────────────────
  int gt_n = 0, gt_k = 0;
  std::vector<int> gt;
  if (gt_path) {
    gt = read_ivecs(gt_path, &gt_n, &gt_k, nq_limit);
    if (gt_k < k) {
      fprintf(stderr, "GT k=%d < required k=%d\n", gt_k, k);
      return 1;
    }
    printf("GT    : nq=%d k=%d (loaded from file)\n", gt_n, gt_k);
  } else {
    gt_k = k;
    gt   = compute_gt_cpu(h_base.data(), n, dim, h_query.data(), nq, k);
  }

  // ── GPU memory ────────────────────────────────────────────────────────────
  float* d_base  = nullptr;
  float* d_query = nullptr;
  int*   d_graph = nullptr;
  int*   d_ids   = nullptr;
  float* d_dists = nullptr;

  CUDA_CHECK(cudaMalloc(&d_base,  sizeof(float) * (long long)n  * dim));
  CUDA_CHECK(cudaMalloc(&d_query, sizeof(float) * (long long)nq * dim));
  CUDA_CHECK(cudaMalloc(&d_graph, sizeof(int)   * (long long)n  * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_ids,   sizeof(int)   * (long long)nq * kTopK));
  CUDA_CHECK(cudaMalloc(&d_dists, sizeof(float) * (long long)nq * kTopK));

  CUDA_CHECK(cudaMemcpy(d_base,  h_base.data(),
                        sizeof(float) * (long long)n  * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, h_query.data(),
                        sizeof(float) * (long long)nq * dim, cudaMemcpyHostToDevice));

  // ── build ─────────────────────────────────────────────────────────────────
  printf("building graph (n=%d degree=%d) ...\n", n, kGraphDegree);
  fflush(stdout);
  auto tb0 = std::chrono::high_resolution_clock::now();
  build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph);
  CUDA_CHECK(cudaDeviceSynchronize());
  double build_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - tb0).count();
  printf("build : %.1f ms\n\n", build_ms);

  // ── sweep max_iterations ─────────────────────────────────────────────────
  // Varying max_iterations traces the recall-QPS Pareto frontier:
  // few iterations → high QPS / low recall; many → lower QPS / high recall.
  const int sweep[] = {4, 8, 12, 16, 20, 24, 32, 48, 64};
  const int n_sweep = static_cast<int>(sizeof(sweep) / sizeof(sweep[0]));

  std::vector<int> h_ids((long long)nq * kTopK);

  FILE* csv = fopen(out_csv, "w");
  if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv); return 1; }
  fprintf(csv, "max_iters,recall,qps,search_ms\n");

  printf("%-12s  %-10s  %-14s  %s\n", "max_iters", "recall@k", "QPS", "ms/batch");
  printf("%s\n", std::string(52, '-').c_str());

  for (int si = 0; si < n_sweep; ++si) {
    SearchPlan plan;
    plan.algo           = SearchAlgo::Auto;
    plan.max_iterations = sweep[si];
    plan.search_width   = kSingleCtaSearchWidth;

    // warm-up (allocates CUDA context, JIT, caches)
    search_engineered(d_base, d_graph, d_query, n, dim, nq, plan, d_ids, d_dists);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed runs — report median to suppress outliers
    std::vector<double> times;
    times.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      search_engineered(d_base, d_graph, d_query, n, dim, nq, plan, d_ids, d_dists);
      CUDA_CHECK(cudaDeviceSynchronize());
      times.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    double med_ms = times[times.size() / 2];
    double qps    = nq / (med_ms / 1000.0);

    CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids,
                          sizeof(int) * (long long)nq * kTopK, cudaMemcpyDeviceToHost));
    float recall = recall_at_k(h_ids.data(), nq, k, gt.data(), gt_k);

    printf("%-12d  %-10.4f  %-14.1f  %.2f\n", sweep[si], recall, qps, med_ms);
    fprintf(csv, "%d,%.6f,%.2f,%.3f\n", sweep[si], recall, qps, med_ms);
  }

  fclose(csv);
  printf("\nbuild_ms=%.1f   results -> %s\n", build_ms, out_csv);

  CUDA_CHECK(cudaFree(d_base));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_graph));
  CUDA_CHECK(cudaFree(d_ids));
  CUDA_CHECK(cudaFree(d_dists));
  return 0;
}
