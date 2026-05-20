// cuvs_bench.cu — 直接调用 cuVS C++ API 的对比 benchmark
//
// 与 engineered_core 的区别：
//   engineered_core 调用本项目自己的 CUDA kernel；
//   此文件调用官方 libcuvs 的 C++ API，底层 kernel 完全一致，
//   无 Python 层开销，保证对比公平。
//
// 输出格式与 engineered_core 完全相同（max_iters,recall,qps,search_ms），
// 可直接用 bench_plot.py pareto 叠图。
//
// 构建：
//   cd cuvs_bench && mkdir build && cd build
//   cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
//   make -j
//
// 运行：
//   ./build/cuvs_bench \
//       --base  ../sift/sift_base.fvecs  \
//       --query ../sift/sift_query.fvecs \
//       --gt    ../sift/sift_groundtruth.ivecs \
//       --out   ../results/cuvs_bench_sweep.csv

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ── fvecs / ivecs I/O ─────────────────────────────────────────────────────────

std::vector<float> read_fvecs(const char* path, int* out_n, int* out_dim, int limit = 0)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  int dim = 0;
  f.read(reinterpret_cast<char*>(&dim), 4);
  f.seekg(0, std::ios::end);
  long long total = f.tellg();
  int bpv = 4 + dim * 4;
  int n = static_cast<int>(total / bpv);
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
  int n = static_cast<int>(total / bpv);
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

// ── CPU brute-force GT ────────────────────────────────────────────────────────

std::vector<int> compute_gt_cpu(const float* base, int n, int dim,
                                  const float* query, int nq, int k)
{
  printf("computing GT (CPU brute-force, n=%d nq=%d k=%d) ...\n", n, nq, k);
  fflush(stdout);
  std::vector<int>   gt(static_cast<long long>(nq) * k);
  std::vector<float> dists(n);
  std::vector<int>   idx(n);
  for (int q = 0; q < nq; ++q) {
    const float* qv = query + static_cast<long long>(q) * dim;
    for (int i = 0; i < n; ++i) {
      const float* bv = base + static_cast<long long>(i) * dim;
      float d = 0.f;
      for (int j = 0; j < dim; ++j) { float diff = qv[j]-bv[j]; d += diff*diff; }
      dists[i] = d; idx[i] = i;
    }
    std::nth_element(idx.begin(), idx.begin()+k, idx.end(),
                     [&](int a, int b){ return dists[a] < dists[b]; });
    std::sort(idx.begin(), idx.begin()+k,
              [&](int a, int b){ return dists[a] < dists[b]; });
    for (int j = 0; j < k; ++j)
      gt[static_cast<long long>(q)*k + j] = idx[j];
    if ((q+1) % 100 == 0) { printf("  GT %d/%d\r", q+1, nq); fflush(stdout); }
  }
  printf("  GT done.           \n");
  return gt;
}

// ── recall@k（neighbors 为 uint32_t，与 cuVS 输出类型一致）──────────────────

float recall_at_k(const uint32_t* result, int nq, int k,
                   const int* gt, int gt_k)
{
  double total = 0.0;
  for (int q = 0; q < nq; ++q) {
    const uint32_t* r = result + static_cast<long long>(q) * k;
    const int*      g = gt     + static_cast<long long>(q) * gt_k;
    int hit = 0;
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < gt_k; ++j)
        if ((int)r[i] == g[j]) { ++hit; break; }
    total += static_cast<double>(hit) / k;
  }
  return static_cast<float>(total / nq);
}

} // namespace

int main(int argc, char** argv)
{
  // ── 参数解析 ──────────────────────────────────────────────────────────────
  const char* base_path  = nullptr;
  const char* query_path = nullptr;
  const char* gt_path    = nullptr;
  const char* out_csv    = "cuvs_bench_sweep.csv";
  int n_limit   = 0;
  int nq_limit  = 1000;
  int k         = 10;
  int repeats   = 5;
  // 构建参数（与 cuVS 默认值对齐）
  int intermediate_graph_degree = 128;
  int graph_degree              = 64;
  // 搜索参数（固定，只 sweep max_iterations）
  int itopk_size   = 64;
  int search_width = 2;

  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--base")    && i+1<argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--query")   && i+1<argc) query_path = argv[++i];
    else if (!strcmp(argv[i], "--gt")      && i+1<argc) gt_path    = argv[++i];
    else if (!strcmp(argv[i], "--out")     && i+1<argc) out_csv    = argv[++i];
    else if (!strcmp(argv[i], "--n")       && i+1<argc) n_limit    = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nq")      && i+1<argc) nq_limit   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--k")       && i+1<argc) k          = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--repeats") && i+1<argc) repeats    = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--intermediate-graph-degree") && i+1<argc)
      intermediate_graph_degree = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--graph-degree")  && i+1<argc)
      graph_degree  = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--itopk-size")    && i+1<argc)
      itopk_size    = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--search-width")  && i+1<argc)
      search_width  = atoi(argv[++i]);
  }

  // ── 加载数据 ──────────────────────────────────────────────────────────────
  int n = 0, dim = 0, nq = 0, qdim = 0;
  std::vector<float> h_base, h_query;

  if (base_path && query_path) {
    h_base  = read_fvecs(base_path,  &n,  &dim,  n_limit);
    h_query = read_fvecs(query_path, &nq, &qdim, nq_limit);
    printf("base  : n=%d dim=%d\n", n, dim);
    printf("query : nq=%d dim=%d\n", nq, qdim);
  } else {
    printf("[smoke-test] no --base/--query; using random n=8192 dim=128 nq=200\n");
    n = 8192; dim = 128; nq = 200;
    h_base .resize(static_cast<long long>(n)  * dim);
    h_query.resize(static_cast<long long>(nq) * dim);
    srand(42);
    for (float& x : h_base)  x = static_cast<float>(rand()) / RAND_MAX;
    for (float& x : h_query) x = static_cast<float>(rand()) / RAND_MAX;
  }

  // ── Ground truth ──────────────────────────────────────────────────────────
  int gt_n = 0, gt_k = 0;
  std::vector<int> gt;
  if (gt_path) {
    gt = read_ivecs(gt_path, &gt_n, &gt_k, nq_limit);
    printf("GT    : nq=%d k=%d (from file)\n", gt_n, gt_k);
  } else {
    gt_k = k;
    gt   = compute_gt_cpu(h_base.data(), n, dim, h_query.data(), nq, k);
  }

  // ── cuVS 资源句柄（管理 CUDA stream、cublas handle 等）────────────────────
  raft::device_resources res;

  // ── 上传数据到 GPU（使用 raft managed device matrix）─────────────────────
  auto d_base  = raft::make_device_matrix<float, int64_t>(res, (int64_t)n,  (int64_t)dim);
  auto d_query = raft::make_device_matrix<float, int64_t>(res, (int64_t)nq, (int64_t)dim);

  raft::copy(d_base.data_handle(),  h_base.data(),
             static_cast<long long>(n)  * dim, raft::resource::get_cuda_stream(res));
  raft::copy(d_query.data_handle(), h_query.data(),
             static_cast<long long>(nq) * dim, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // ── 构建 cuVS CAGRA 索引 ──────────────────────────────────────────────────
  //
  // 参数选择说明：
  //   intermediate_graph_degree=128, graph_degree=64 是 cuVS 默认值，
  //   对应 SIFT1M 标准 benchmark 配置。
  //   build_algo 选 nn_descent 与本项目 engineered_build 一致（均为 NN-Descent）。
  //
  cuvs::neighbors::cagra::index_params build_params;
  build_params.intermediate_graph_degree = static_cast<size_t>(intermediate_graph_degree);
  build_params.graph_degree              = static_cast<size_t>(graph_degree);
  build_params.build_algo =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      static_cast<size_t>(intermediate_graph_degree));

  printf("building index (n=%d graph_degree=%d intermediate=%d algo=nn_descent) ...\n",
         n, graph_degree, intermediate_graph_degree);
  fflush(stdout);

  auto t0_build = std::chrono::high_resolution_clock::now();
  auto index = cuvs::neighbors::cagra::build(
    res, build_params,
    raft::make_const_mdspan(d_base.view()));
  raft::resource::sync_stream(res);
  double build_ms = std::chrono::duration<double, std::milli>(
    std::chrono::high_resolution_clock::now() - t0_build).count();
  printf("build : %.1f ms\n\n", build_ms);

  // ── 搜索缓冲区 ────────────────────────────────────────────────────────────
  // cuVS 的邻居输出为 uint32_t（节点 id），距离为 float。
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, (int64_t)nq, (int64_t)k);
  auto d_distances = raft::make_device_matrix<float,    int64_t>(res, (int64_t)nq, (int64_t)k);
  std::vector<uint32_t> h_neighbors(static_cast<long long>(nq) * k);

  // ── Sweep max_iterations → Pareto 曲线 ───────────────────────────────────
  const int sweep[]  = {4, 8, 12, 16, 20, 24, 32, 48, 64};
  const int n_sweep  = static_cast<int>(sizeof(sweep) / sizeof(sweep[0]));

  FILE* csv = fopen(out_csv, "w");
  if (!csv) { fprintf(stderr, "cannot write %s\n", out_csv); return 1; }
  // 首行写 build 元数据，便于与 engineered_core 对比构建时间
  fprintf(csv, "# build_ms=%.1f n=%d graph_degree=%d intermediate=%d\n",
          build_ms, n, graph_degree, intermediate_graph_degree);
  fprintf(csv, "max_iters,recall,qps,search_ms\n");

  printf("%-12s  %-10s  %-14s  %s\n", "max_iters", "recall@k", "QPS", "ms/batch");
  printf("%s\n", std::string(52, '-').c_str());

  for (int si = 0; si < n_sweep; ++si) {
    cuvs::neighbors::cagra::search_params sp;
    sp.max_iterations = static_cast<size_t>(sweep[si]);
    sp.itopk_size     = static_cast<size_t>(itopk_size);
    sp.search_width   = static_cast<size_t>(search_width);

    // warm-up：初始化 cuVS 内部 JIT / workspace
    cuvs::neighbors::cagra::search(
      res, sp, index,
      raft::make_const_mdspan(d_query.view()),
      d_neighbors.view(), d_distances.view());
    raft::resource::sync_stream(res);

    // 计时：取 repeats 次中位数，抑制 outlier
    std::vector<double> times;
    times.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      cuvs::neighbors::cagra::search(
        res, sp, index,
        raft::make_const_mdspan(d_query.view()),
        d_neighbors.view(), d_distances.view());
      raft::resource::sync_stream(res);  // 与 engineered_core 的 cudaDeviceSynchronize 等价
      times.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    double med_ms = times[times.size() / 2];
    double qps    = nq / (med_ms / 1000.0);

    // 拷回 host，计算 recall
    raft::copy(h_neighbors.data(), d_neighbors.data_handle(),
               static_cast<long long>(nq) * k,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    float recall = recall_at_k(h_neighbors.data(), nq, k, gt.data(), gt_k);

    printf("%-12d  %-10.4f  %-14.1f  %.2f\n", sweep[si], recall, qps, med_ms);
    fprintf(csv, "%d,%.6f,%.2f,%.3f\n", sweep[si], recall, qps, med_ms);
  }

  fclose(csv);
  printf("\nbuild_ms=%.1f   results -> %s\n", build_ms, out_csv);
  return 0;
}
