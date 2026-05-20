// CAGRA benchmark: reads SIFT-format data, sweeps max_iterations to produce
// a recall@k vs QPS Pareto curve saved as CSV.
//
// Usage:
//   ./engineered_core [--base path.fvecs] [--query path.fvecs]
//                     [--gt path.ivecs] [--n N] [--nq NQ] [--k K]
//                     [--repeats R] [--out stem] [--mode MODE]
//
// --base / --query : fvecs files (SIFT1M format).
// --gt             : precomputed ivecs ground truth (optional; falls back to
//                    CPU brute-force when omitted).
// --n / --nq       : cap on base / query vectors loaded.
// --k              : top-k (must be <= kTopK in config.cuh, default kTopK=10).
// --mode           : pareto (default), build_breakdown, nn_descent,
//                    search_width, cta_regime, scalability

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

// ── Timed search helper ───────────────────────────────────────────────────────
// Returns median search time in ms over `repeats` runs.
double timed_search(const float* d_base, const int* d_graph,
                    const float* d_query, int n, int dim, int num_q,
                    cagra_repro::engineered::SearchPlan plan,
                    int* d_ids, float* d_dists, int repeats)
{
  using namespace cagra_repro::engineered;
  // warm-up
  search_engineered(d_base, d_graph, d_query, n, dim, num_q, plan, d_ids, d_dists);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> times;
  times.reserve(repeats);
  for (int r = 0; r < repeats; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    search_engineered(d_base, d_graph, d_query, n, dim, num_q, plan, d_ids, d_dists);
    CUDA_CHECK(cudaDeviceSynchronize());
    times.push_back(std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
  using namespace cagra_repro::engineered;

  // ── arg parsing ──────────────────────────────────────────────────────────
  const char* base_path  = nullptr;
  const char* query_path = nullptr;
  const char* gt_path    = nullptr;
  const char* out_stem   = nullptr;   // mode-specific suffix added below
  const char* mode_str   = "pareto";
  int n_limit  = 0;
  int nq_limit = 1000;
  int k        = kTopK;
  int repeats  = 5;

  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--base")    && i+1 < argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--query")   && i+1 < argc) query_path = argv[++i];
    else if (!strcmp(argv[i], "--gt")      && i+1 < argc) gt_path    = argv[++i];
    else if (!strcmp(argv[i], "--out")     && i+1 < argc) out_stem   = argv[++i];
    else if (!strcmp(argv[i], "--mode")    && i+1 < argc) mode_str   = argv[++i];
    else if (!strcmp(argv[i], "--n")       && i+1 < argc) n_limit    = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nq")      && i+1 < argc) nq_limit   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--k")       && i+1 < argc) k          = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--repeats") && i+1 < argc) repeats    = atoi(argv[++i]);
  }

  // Determine mode
  enum class Mode { Pareto, BuildBreakdown, NnDescent, SearchWidth, CtaRegime, Scalability };
  Mode mode = Mode::Pareto;
  if      (!strcmp(mode_str, "pareto"))           mode = Mode::Pareto;
  else if (!strcmp(mode_str, "build_breakdown"))  mode = Mode::BuildBreakdown;
  else if (!strcmp(mode_str, "nn_descent"))       mode = Mode::NnDescent;
  else if (!strcmp(mode_str, "search_width"))     mode = Mode::SearchWidth;
  else if (!strcmp(mode_str, "cta_regime"))       mode = Mode::CtaRegime;
  else if (!strcmp(mode_str, "scalability"))      mode = Mode::Scalability;
  else { fprintf(stderr, "unknown mode: %s\n", mode_str); return 1; }

  // Determine output CSV path
  // pareto mode: backwards compat — out_stem IS the csv name (default cagra_sweep.csv)
  // other modes: out_stem + "_" + mode + ".csv"
  std::string out_csv;
  if (mode == Mode::Pareto) {
    out_csv = out_stem ? out_stem : "cagra_sweep.csv";
  } else {
    std::string stem = out_stem ? out_stem : "results";
    out_csv = stem + "_" + mode_str + ".csv";
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
  // For scalability mode we defer GT computation per sub-dataset size.
  // For other modes, load or compute GT now.
  if (mode != Mode::Scalability) {
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

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: pareto (default)
  // ═══════════════════════════════════════════════════════════════════════════
  if (mode == Mode::Pareto) {
    printf("building graph (n=%d degree=%d) ...\n", n, kGraphDegree);
    fflush(stdout);
    auto tb0 = std::chrono::high_resolution_clock::now();
    build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph);
    CUDA_CHECK(cudaDeviceSynchronize());
    double build_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tb0).count();
    printf("build : %.1f ms\n\n", build_ms);

    // Varying max_iterations traces the recall-QPS Pareto frontier.
    const int sweep[] = {4, 8, 12, 16, 20, 24, 32, 48, 64};
    const int n_sweep = static_cast<int>(sizeof(sweep) / sizeof(sweep[0]));

    std::vector<int> h_ids((long long)nq * kTopK);

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "max_iters,recall,qps,search_ms\n");

    printf("%-12s  %-10s  %-14s  %s\n", "max_iters", "recall@k", "QPS", "ms/batch");
    printf("%s\n", std::string(52, '-').c_str());

    for (int si = 0; si < n_sweep; ++si) {
      SearchPlan plan;
      plan.algo           = SearchAlgo::Auto;
      plan.max_iterations = sweep[si];
      plan.search_width   = kSingleCtaSearchWidth;

      double med_ms = timed_search(d_base, d_graph, d_query, n, dim, nq, plan,
                                   d_ids, d_dists, repeats);
      double qps = nq / (med_ms / 1000.0);

      CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids,
                            sizeof(int) * (long long)nq * kTopK, cudaMemcpyDeviceToHost));
      float recall = recall_at_k(h_ids.data(), nq, k, gt.data(), gt_k);

      printf("%-12d  %-10.4f  %-14.1f  %.2f\n", sweep[si], recall, qps, med_ms);
      fprintf(csv, "%d,%.6f,%.2f,%.3f\n", sweep[si], recall, qps, med_ms);
    }

    fclose(csv);
    printf("\nbuild_ms=%.1f   results -> %s\n", build_ms, out_csv.c_str());
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: build_breakdown
  // ═══════════════════════════════════════════════════════════════════════════
  else if (mode == Mode::BuildBreakdown) {
    printf("build_breakdown: timing each build stage for engineered impl\n");
    fflush(stdout);

    BuildTiming bt;
    build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph,
                           /*nn_iters=*/-1, &bt);
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = bt.init_ms + bt.nn_descent_ms + bt.prune_ms
                    + bt.reverse_ms + bt.connectivity_ms;
    printf("  init         : %.2f ms\n", bt.init_ms);
    printf("  nn_descent   : %.2f ms\n", bt.nn_descent_ms);
    printf("  prune        : %.2f ms\n", bt.prune_ms);
    printf("  reverse      : %.2f ms\n", bt.reverse_ms);
    printf("  connectivity : %.2f ms\n", bt.connectivity_ms);
    printf("  total        : %.2f ms\n", total_ms);

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "stage,impl,time_ms\n");
    fprintf(csv, "init,engineered,%.3f\n",         bt.init_ms);
    fprintf(csv, "nn_descent,engineered,%.3f\n",   bt.nn_descent_ms);
    fprintf(csv, "prune,engineered,%.3f\n",        bt.prune_ms);
    fprintf(csv, "reverse,engineered,%.3f\n",      bt.reverse_ms);
    fprintf(csv, "connectivity,engineered,%.3f\n", bt.connectivity_ms);
    fclose(csv);
    printf("\nresults -> %s\n", out_csv.c_str());
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: nn_descent
  // ═══════════════════════════════════════════════════════════════════════════
  else if (mode == Mode::NnDescent) {
    const int nn_iters_sweep[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12};
    const int n_sw = static_cast<int>(sizeof(nn_iters_sweep) / sizeof(nn_iters_sweep[0]));

    std::vector<int> h_ids((long long)nq * kTopK);

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "nn_iters,recall,build_ms\n");

    printf("%-10s  %-10s  %-12s\n", "nn_iters", "recall@k", "build_ms");
    printf("%s\n", std::string(36, '-').c_str());

    SearchPlan plan;
    plan.algo           = SearchAlgo::Auto;
    plan.max_iterations = kMaxSearchIters;
    plan.search_width   = kSingleCtaSearchWidth;

    for (int si = 0; si < n_sw; ++si) {
      int ni = nn_iters_sweep[si];

      auto tb0 = std::chrono::high_resolution_clock::now();
      build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph, ni);
      CUDA_CHECK(cudaDeviceSynchronize());
      double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tb0).count();

      // single run for recall (warm-up first)
      search_engineered(d_base, d_graph, d_query, n, dim, nq, plan, d_ids, d_dists);
      CUDA_CHECK(cudaDeviceSynchronize());
      search_engineered(d_base, d_graph, d_query, n, dim, nq, plan, d_ids, d_dists);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids,
                            sizeof(int) * (long long)nq * kTopK, cudaMemcpyDeviceToHost));
      float recall = recall_at_k(h_ids.data(), nq, k, gt.data(), gt_k);

      printf("%-10d  %-10.4f  %.2f\n", ni, recall, build_ms);
      fprintf(csv, "%d,%.6f,%.3f\n", ni, recall, build_ms);
    }

    fclose(csv);
    printf("\nresults -> %s\n", out_csv.c_str());
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: search_width
  // ═══════════════════════════════════════════════════════════════════════════
  else if (mode == Mode::SearchWidth) {
    printf("search_width: building graph once (nn_iters=kNnDescentIters) ...\n");
    fflush(stdout);
    build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("build done.\n\n");

    const int sw_vals[]    = {1, 2, 4, 8};
    const int iters_vals[] = {4, 8, 12, 16, 20, 24, 32, 48, 64};
    const int n_sw  = static_cast<int>(sizeof(sw_vals)    / sizeof(sw_vals[0]));
    const int n_its = static_cast<int>(sizeof(iters_vals) / sizeof(iters_vals[0]));

    std::vector<int> h_ids((long long)nq * kTopK);

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "search_width,max_iters,recall,qps\n");

    printf("%-14s  %-12s  %-10s  %-14s\n", "search_width", "max_iters", "recall@k", "QPS");
    printf("%s\n", std::string(54, '-').c_str());

    for (int wi = 0; wi < n_sw; ++wi) {
      for (int ii = 0; ii < n_its; ++ii) {
        SearchPlan plan;
        plan.algo           = SearchAlgo::Auto;
        plan.max_iterations = iters_vals[ii];
        plan.search_width   = sw_vals[wi];

        double med_ms = timed_search(d_base, d_graph, d_query, n, dim, nq, plan,
                                     d_ids, d_dists, repeats);
        double qps = nq / (med_ms / 1000.0);

        CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids,
                              sizeof(int) * (long long)nq * kTopK, cudaMemcpyDeviceToHost));
        float recall = recall_at_k(h_ids.data(), nq, k, gt.data(), gt_k);

        printf("%-14d  %-12d  %-10.4f  %.1f\n", sw_vals[wi], iters_vals[ii], recall, qps);
        fprintf(csv, "%d,%d,%.6f,%.2f\n", sw_vals[wi], iters_vals[ii], recall, qps);
      }
    }

    fclose(csv);
    printf("\nresults -> %s\n", out_csv.c_str());
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: cta_regime
  // ═══════════════════════════════════════════════════════════════════════════
  else if (mode == Mode::CtaRegime) {
    printf("cta_regime: building graph once ...\n");
    fflush(stdout);
    build_graph_engineered(d_base, n, dim, /*guarantee_connectivity=*/true, d_graph);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("build done.\n\n");

    const int num_q_vals[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000};
    const int n_qv = static_cast<int>(sizeof(num_q_vals) / sizeof(num_q_vals[0]));

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "num_queries,algo,qps\n");

    printf("%-14s  %-12s  %-14s\n", "num_queries", "algo", "QPS");
    printf("%s\n", std::string(44, '-').c_str());

    SearchPlan plan_base;
    plan_base.max_iterations = kMaxSearchIters;
    plan_base.search_width   = kSingleCtaSearchWidth;

    const SearchAlgo algos[] = {SearchAlgo::SingleCta, SearchAlgo::MultiCta};
    const char* algo_names[] = {"SingleCta", "MultiCta"};

    for (int qi = 0; qi < n_qv; ++qi) {
      int num_q = std::min(num_q_vals[qi], nq);

      for (int ai = 0; ai < 2; ++ai) {
        SearchPlan plan = plan_base;
        plan.algo = algos[ai];

        double med_ms = timed_search(d_base, d_graph, d_query, n, dim, num_q, plan,
                                     d_ids, d_dists, repeats);
        double qps = num_q / (med_ms / 1000.0);

        printf("%-14d  %-12s  %.1f\n", num_q, algo_names[ai], qps);
        fprintf(csv, "%d,%s,%.2f\n", num_q, algo_names[ai], qps);
      }
    }

    fclose(csv);
    printf("\nresults -> %s\n", out_csv.c_str());
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MODE: scalability
  // ═══════════════════════════════════════════════════════════════════════════
  else if (mode == Mode::Scalability) {
    const int n_sub_vals[] = {
      1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000
    };
    const int n_sv = static_cast<int>(sizeof(n_sub_vals) / sizeof(n_sub_vals[0]));

    FILE* csv = fopen(out_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv.c_str()); return 1; }
    fprintf(csv, "n,build_ms,qps,recall\n");

    printf("%-12s  %-12s  %-14s  %-10s\n", "n_sub", "build_ms", "QPS", "recall@k");
    printf("%s\n", std::string(52, '-').c_str());

    SearchPlan plan;
    plan.algo           = SearchAlgo::Auto;
    plan.max_iterations = kMaxSearchIters;
    plan.search_width   = kSingleCtaSearchWidth;

    std::vector<int> h_ids((long long)nq * kTopK);

    for (int si = 0; si < n_sv; ++si) {
      int n_sub = n_sub_vals[si];
      if (n_sub > n) break;  // cap at actual dataset size

      // Build graph on first n_sub vectors
      auto tb0 = std::chrono::high_resolution_clock::now();
      build_graph_engineered(d_base, n_sub, dim, /*guarantee_connectivity=*/true, d_graph);
      CUDA_CHECK(cudaDeviceSynchronize());
      double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tb0).count();

      // Search: pass first nq queries, graph is over n_sub base vectors
      double med_ms = timed_search(d_base, d_graph, d_query, n_sub, dim, nq, plan,
                                   d_ids, d_dists, repeats);
      double qps = nq / (med_ms / 1000.0);

      // Recall: only compute for n_sub <= 20000 (brute-force GT is feasible)
      float recall_val = -1.f;
      if (n_sub <= 20000) {
        std::vector<int> sub_gt = compute_gt_cpu(
          h_base.data(), n_sub, dim, h_query.data(), nq, k);
        CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids,
                              sizeof(int) * (long long)nq * kTopK, cudaMemcpyDeviceToHost));
        recall_val = recall_at_k(h_ids.data(), nq, k, sub_gt.data(), k);
      }

      if (recall_val >= 0.f)
        printf("%-12d  %-12.1f  %-14.1f  %.4f\n", n_sub, build_ms, qps, recall_val);
      else
        printf("%-12d  %-12.1f  %-14.1f  (skipped)\n", n_sub, build_ms, qps);

      if (recall_val >= 0.f)
        fprintf(csv, "%d,%.3f,%.2f,%.6f\n", n_sub, build_ms, qps, recall_val);
      else
        fprintf(csv, "%d,%.3f,%.2f,-1\n", n_sub, build_ms, qps);
    }

    fclose(csv);
    printf("\nresults -> %s\n", out_csv.c_str());
  }

  // ── cleanup ───────────────────────────────────────────────────────────────
  CUDA_CHECK(cudaFree(d_base));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_graph));
  CUDA_CHECK(cudaFree(d_ids));
  CUDA_CHECK(cudaFree(d_dists));
  return 0;
}
