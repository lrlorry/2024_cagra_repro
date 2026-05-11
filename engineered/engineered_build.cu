#include "engineered/engineered_build.cuh"

#include "common/cagra_common.cuh"
#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"

#include <climits>

namespace cagra_repro::engineered {

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

__global__ void fill_int_kernel(int* ptr, int count, int value)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) ptr[tid] = value;
}

__global__ void make_reverse_graph_atomic_kernel(const int* pruned_graph,
                                                 int n,
                                                 int* reverse_graph,
                                                 int* reverse_count)
{
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  int total_edges = n * kGraphDegree;
  if (edge >= total_edges) return;

  int src = edge / kGraphDegree;
  int dst = pruned_graph[edge];
  if (dst < 0 || dst >= n) return;

  int pos = atomicAdd(reverse_count + dst, 1);
  if (pos < kGraphDegree) {
    reverse_graph[dst * kGraphDegree + pos] = src;
  }
}

__device__ bool local_contains(const int* values, int count, int cand)
{
  for (int i = 0; i < count; ++i) {
    if (values[i] == cand) return true;
  }
  return false;
}

__global__ void combine_graph_kernel(const int* pruned_graph,
                                     const int* reverse_graph,
                                     int n,
                                     int* output_graph)
{
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n) return;

  int out[kGraphDegree];
  for (int k = 0; k < kGraphDegree; ++k) {
    out[k] = pruned_graph[node * kGraphDegree + k];
  }

  // Protected slots [0, protected_edges) keep the best pruned forward edges.
  // Reverse edges fill [protected_edges, kGraphDegree).
  // Mirrors cuVS graph_core.cuh: merge_pruned_and_reverse_edges.
  const int protected_edges = kGraphDegree / 2;
  for (int r = kGraphDegree - 1; r >= 0; --r) {
    int cand = reverse_graph[node * kGraphDegree + r];
    if (cand < 0 || cand >= n) continue;
    if (local_contains(out, kGraphDegree, cand)) continue;

    for (int j = kGraphDegree - 1; j > protected_edges; --j) {
      out[j] = out[j - 1];
    }
    out[protected_edges] = cand;
  }

  for (int k = 0; k < kGraphDegree; ++k) {
    output_graph[node * kGraphDegree + k] = out[k];
  }
}

// Propagate reachability from node 0 one hop at a time.
// Call repeatedly until *changed == false.
// Mirrors cuVS graph_core.cuh: BFS inside guarantee_connectivity.
__global__ void bfs_mark_kernel(const int* graph, bool* reachable, bool* changed, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !reachable[i]) return;
  for (int k = 0; k < kGraphDegree; ++k) {
    int nb = graph[i * kGraphDegree + k];
    if (nb >= 0 && nb < n && !reachable[nb]) {
      reachable[nb] = true;
      *changed      = true;
    }
  }
}

// For each node unreachable from node 0, scan the dataset to find the nearest
// reachable node and force-add it as the first outgoing edge.
// Mirrors cuVS graph_core.cuh: guarantee_connectivity bridge insertion.
__global__ void repair_unreachable_kernel(const float* dataset,
                                          int*         graph,
                                          const bool*  reachable,
                                          int          n,
                                          int          dim)
{
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n || reachable[node]) return;

  float best_dist = CUDART_INF_F;
  int   best      = -1;
  for (int c = 0; c < n; ++c) {
    if (!reachable[c]) continue;
    float d = l2_point_point_scalar(dataset, node, c, dim);
    if (d < best_dist) { best_dist = d; best = c; }
  }
  if (best >= 0) graph[node * kGraphDegree + 0] = best;
}

void build_graph_engineered(const float* d_dataset,
                            int n,
                            int dim,
                            bool guarantee_connectivity,
                            int* d_graph)
{
  int* d_initial = nullptr;
  int* d_pruned = nullptr;
  int* d_reverse = nullptr;
  int* d_reverse_count = nullptr;

  CUDA_CHECK(cudaMalloc(&d_initial, sizeof(int) * n * kInitialDegree));
  CUDA_CHECK(cudaMalloc(&d_pruned, sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_reverse, sizeof(int) * n * kGraphDegree));
  CUDA_CHECK(cudaMalloc(&d_reverse_count, sizeof(int) * n));

  dim3 block(kBlockThreads);
  dim3 grid_nodes((n + block.x - 1) / block.x);
  dim3 grid_edges((n * kGraphDegree + block.x - 1) / block.x);

  init_random_graph_kernel<<<grid_nodes, block>>>(d_dataset, n, dim, d_initial);
  CUDA_CHECK(cudaGetLastError());

  for (int iter = 0; iter < kNnDescentIters; ++iter) {
    nn_descent_relax_kernel<<<grid_nodes, block>>>(d_dataset, n, dim, d_initial);
    CUDA_CHECK(cudaGetLastError());
  }

  rank_prune_kernel<<<grid_nodes, block>>>(d_initial, n, d_pruned);
  CUDA_CHECK(cudaGetLastError());

  fill_int_kernel<<<grid_edges, block>>>(d_reverse, n * kGraphDegree, -1);
  fill_int_kernel<<<grid_nodes, block>>>(d_reverse_count, n, 0);
  CUDA_CHECK(cudaGetLastError());

  make_reverse_graph_atomic_kernel<<<grid_edges, block>>>(
    d_pruned, n, d_reverse, d_reverse_count);
  CUDA_CHECK(cudaGetLastError());

  combine_graph_kernel<<<grid_nodes, block>>>(d_pruned, d_reverse, n, d_graph);
  CUDA_CHECK(cudaGetLastError());

  if (guarantee_connectivity) {
    bool* d_reachable = nullptr;
    bool* d_changed   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reachable, sizeof(bool) * n));
    CUDA_CHECK(cudaMalloc(&d_changed,   sizeof(bool)));

    CUDA_CHECK(cudaMemset(d_reachable, 0, sizeof(bool) * n));
    bool init_true = true;
    CUDA_CHECK(cudaMemcpy(d_reachable, &init_true, sizeof(bool), cudaMemcpyHostToDevice));

    // BFS until no new nodes are discovered.
    bool h_changed;
    do {
      h_changed = false;
      CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
      bfs_mark_kernel<<<grid_nodes, block>>>(d_graph, d_reachable, d_changed, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (h_changed);

    repair_unreachable_kernel<<<grid_nodes, block>>>(d_dataset, d_graph, d_reachable, n, dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_reachable));
    CUDA_CHECK(cudaFree(d_changed));
  }

  CUDA_CHECK(cudaFree(d_initial));
  CUDA_CHECK(cudaFree(d_pruned));
  CUDA_CHECK(cudaFree(d_reverse));
  CUDA_CHECK(cudaFree(d_reverse_count));
}

}  // namespace cagra_repro::engineered

