#include "plain/plain_build.cuh"

#include "common/cagra_common.cuh"
#include "common/cuda_utils.cuh"
#include "plain/config.cuh"

#include <climits>

namespace cagra_repro::plain {

__global__ void init_random_graph_kernel(const float* dataset, int n, int dim, int* graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  int local_ids[kInitialDegree];
  float local_dists[kInitialDegree];
  for (int i = 0; i < kInitialDegree; ++i) {
    local_ids[i] = -1;
    local_dists[i] = CUDART_INF_F;
  }

  for (int t = 0; t < kInitialDegree * 8; ++t) {
    int cand = static_cast<int>(hash_u32(src * 4099U + t * 131U)) % n;
    if (cand == src) continue;
    float dist = l2_point_point_scalar(dataset, src, cand, dim);
    insert_sorted_unique<kInitialDegree>(local_ids, local_dists, cand, dist);
  }

  for (int t = 0; t < n && local_ids[kInitialDegree - 1] < 0; ++t) {
    int cand = (src + t + 1) % n;
    if (cand == src) continue;
    float dist = l2_point_point_scalar(dataset, src, cand, dim);
    insert_sorted_unique<kInitialDegree>(local_ids, local_dists, cand, dist);
  }

  for (int k = 0; k < kInitialDegree; ++k) {
    graph[src * kInitialDegree + k] = local_ids[k];
  }
}

__global__ void nn_descent_relax_kernel(const float* dataset, int n, int dim, int* graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  int local_ids[kInitialDegree];
  float local_dists[kInitialDegree];

  for (int k = 0; k < kInitialDegree; ++k) {
    int cand = graph[src * kInitialDegree + k];
    local_ids[k] = cand;
    local_dists[k] = cand >= 0 ? l2_point_point_scalar(dataset, src, cand, dim) : CUDART_INF_F;
  }

  // NN-Descent idea: if B is close to A, then B's neighbors are candidate neighbors for A.
  for (int nb_rank = 0; nb_rank < kInitialDegree; ++nb_rank) {
    int nb = graph[src * kInitialDegree + nb_rank];
    if (nb < 0 || nb >= n) continue;

    for (int second_rank = 0; second_rank < kInitialDegree; ++second_rank) {
      int cand = graph[nb * kInitialDegree + second_rank];
      if (cand < 0 || cand == src) continue;
      float dist = l2_point_point_scalar(dataset, src, cand, dim);
      insert_sorted_unique<kInitialDegree>(local_ids, local_dists, cand, dist);
    }
  }

  for (int k = 0; k < kInitialDegree; ++k) {
    graph[src * kInitialDegree + k] = local_ids[k];
  }
}

__device__ int detour_count_for_candidate(const int* initial_graph,
                                          int src,
                                          int cand_rank,
                                          int cand)
{
  int detours = 0;
  for (int prev_rank = 0; prev_rank < cand_rank; ++prev_rank) {
    int detour_node = initial_graph[src * kInitialDegree + prev_rank];
    if (detour_node < 0) continue;
    for (int k = 0; k < kInitialDegree; ++k) {
      if (initial_graph[detour_node * kInitialDegree + k] == cand) {
        ++detours;
        break;
      }
    }
  }
  return detours;
}

__global__ void rank_prune_kernel(const int* initial_graph, int n, int* pruned_graph)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src >= n) return;

  int out_ids[kGraphDegree];
  int out_detours[kGraphDegree];
  int out_ranks[kGraphDegree];
  for (int i = 0; i < kGraphDegree; ++i) {
    out_ids[i] = -1;
    out_detours[i] = INT_MAX;
    out_ranks[i] = INT_MAX;
  }

  for (int rank = 0; rank < kInitialDegree; ++rank) {
    int cand = initial_graph[src * kInitialDegree + rank];
    if (cand < 0 || cand == src) continue;
    int detours = detour_count_for_candidate(initial_graph, src, rank, cand);

    int pos = -1;
    for (int i = 0; i < kGraphDegree; ++i) {
      bool better = detours < out_detours[i] ||
                    (detours == out_detours[i] && rank < out_ranks[i]);
      if (better) {
        pos = i;
        break;
      }
    }
    if (pos < 0) continue;

    for (int j = kGraphDegree - 1; j > pos; --j) {
      out_ids[j] = out_ids[j - 1];
      out_detours[j] = out_detours[j - 1];
      out_ranks[j] = out_ranks[j - 1];
    }
    out_ids[pos] = cand;
    out_detours[pos] = detours;
    out_ranks[pos] = rank;
  }

  for (int k = 0; k < kGraphDegree; ++k) {
    if (out_ids[k] < 0) out_ids[k] = (src + k + 1) % n;
    pruned_graph[src * kGraphDegree + k] = out_ids[k];
  }
}

__global__ void slow_reverse_merge_kernel(const int* pruned_graph, int n, int* output_graph)
{
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= n) return;

  int out_ids[kGraphDegree];
  for (int k = 0; k < kGraphDegree; ++k) {
    out_ids[k] = pruned_graph[dst * kGraphDegree + k];
  }

  const int protected_edges = kGraphDegree / 2;

  // Plain version: scan every edge to find incoming reverse edges.
  for (int src = 0; src < n; ++src) {
    for (int k = 0; k < kGraphDegree; ++k) {
      if (pruned_graph[src * kGraphDegree + k] != dst) continue;
      if (contains_id<kGraphDegree>(out_ids, src)) continue;

      for (int j = kGraphDegree - 1; j > protected_edges; --j) {
        out_ids[j] = out_ids[j - 1];
      }
      out_ids[protected_edges] = src;
    }
  }

  for (int k = 0; k < kGraphDegree; ++k) {
    output_graph[dst * kGraphDegree + k] = out_ids[k];
  }
}

void build_graph_plain(const float* d_dataset, int n, int dim, int* d_graph)
{
  int* d_initial = nullptr;
  int* d_pruned = nullptr;
  CUDA_CHECK(cudaMalloc(&d_initial, sizeof(int) * n * kInitialDegree));
  CUDA_CHECK(cudaMalloc(&d_pruned, sizeof(int) * n * kGraphDegree));

  dim3 block(kThreads);
  dim3 grid((n + block.x - 1) / block.x);

  init_random_graph_kernel<<<grid, block>>>(d_dataset, n, dim, d_initial);
  CUDA_CHECK(cudaGetLastError());

  for (int iter = 0; iter < kNnDescentIters; ++iter) {
    nn_descent_relax_kernel<<<grid, block>>>(d_dataset, n, dim, d_initial);
    CUDA_CHECK(cudaGetLastError());
  }

  rank_prune_kernel<<<grid, block>>>(d_initial, n, d_pruned);
  CUDA_CHECK(cudaGetLastError());

  slow_reverse_merge_kernel<<<grid, block>>>(d_pruned, n, d_graph);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_initial));
  CUDA_CHECK(cudaFree(d_pruned));
}

}  // namespace cagra_repro::plain

