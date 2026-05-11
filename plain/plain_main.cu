#include "common/cuda_utils.cuh"
#include "plain/config.cuh"
#include "plain/plain_build.cuh"
#include "plain/plain_search.cuh"

#include <cstdio>
#include <random>
#include <vector>

int main()
{
  using namespace cagra_repro::plain;

  const int n = 512;
  const int dim = 64;
  const int num_queries = 4;

  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  std::vector<float> h_dataset(n * dim);
  std::vector<float> h_queries(num_queries * dim);
  for (float& x : h_dataset) x = normal(rng);
  for (float& x : h_queries) x = normal(rng);

  float* d_dataset = nullptr;
  float* d_queries = nullptr;
  int* d_graph = nullptr;
  int* d_out_ids = nullptr;
  float* d_out_dists = nullptr;

  CUDA_CHECK(cudaMalloc(&d_dataset, sizeof(float) * h_dataset.size()));
  CUDA_CHECK(cudaMalloc(&d_queries, sizeof(float) * h_queries.size()));
  CUDA_CHECK(cudaMalloc(&d_graph, sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_out_ids, sizeof(int) * num_queries * kTopK));
  CUDA_CHECK(cudaMalloc(&d_out_dists, sizeof(float) * num_queries * kTopK));

  CUDA_CHECK(cudaMemcpy(d_dataset, h_dataset.data(), sizeof(float) * h_dataset.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), sizeof(float) * h_queries.size(),
                        cudaMemcpyHostToDevice));

  build_graph_plain(d_dataset, n, dim, d_graph);
  search_plain(d_dataset, d_graph, d_queries, n, dim, num_queries, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h_ids(num_queries * kTopK);
  std::vector<float> h_dists(num_queries * kTopK);
  CUDA_CHECK(cudaMemcpy(h_ids.data(), d_out_ids, sizeof(int) * h_ids.size(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dists.data(), d_out_dists, sizeof(float) * h_dists.size(),
                        cudaMemcpyDeviceToHost));

  std::printf("Plain CAGRA core result:\n");
  for (int q = 0; q < num_queries; ++q) {
    std::printf("query %d:", q);
    for (int k = 0; k < kTopK; ++k) {
      std::printf(" (%d, %.3f)", h_ids[q * kTopK + k], h_dists[q * kTopK + k]);
    }
    std::printf("\n");
  }

  CUDA_CHECK(cudaFree(d_dataset));
  CUDA_CHECK(cudaFree(d_queries));
  CUDA_CHECK(cudaFree(d_graph));
  CUDA_CHECK(cudaFree(d_out_ids));
  CUDA_CHECK(cudaFree(d_out_dists));
  return 0;
}

