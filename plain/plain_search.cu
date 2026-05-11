#include "plain/plain_search.cuh"

#include "common/cagra_common.cuh"
#include "common/cuda_utils.cuh"
#include "plain/config.cuh"

namespace cagra_repro::plain {

__device__ void insert_topm(int* ids,
                            float* dists,
                            bool* expanded,
                            int cand,
                            float dist)
{
  if (cand < 0) return;
  for (int i = 0; i < kInternalTopM; ++i) {
    if (ids[i] == cand) return;
  }
  if (dist >= dists[kInternalTopM - 1]) return;

  int pos = kInternalTopM - 1;
  while (pos > 0 && dist < dists[pos - 1]) {
    ids[pos] = ids[pos - 1];
    dists[pos] = dists[pos - 1];
    expanded[pos] = expanded[pos - 1];
    --pos;
  }
  ids[pos] = cand;
  dists[pos] = dist;
  expanded[pos] = false;
}

__global__ void search_plain_kernel(const float* dataset,
                                    const int* graph,
                                    const float* queries,
                                    int n,
                                    int dim,
                                    int num_queries,
                                    int* out_ids,
                                    float* out_dists)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;

  int top_ids[kInternalTopM];
  float top_dists[kInternalTopM];
  bool expanded[kInternalTopM];
  for (int i = 0; i < kInternalTopM; ++i) {
    top_ids[i] = -1;
    top_dists[i] = CUDART_INF_F;
    expanded[i] = false;
  }

  for (int s = 0; s < kInternalTopM; ++s) {
    int cand = static_cast<int>(hash_u32(q * 1009U + s * 9176U)) % n;
    float dist = l2_query_point_scalar(dataset, queries, cand, q, dim);
    insert_topm(top_ids, top_dists, expanded, cand, dist);
  }

  for (int iter = 0; iter < kMaxSearchIters; ++iter) {
    int expanded_this_iter = 0;

    for (int m = 0; m < kInternalTopM && expanded_this_iter < kSearchWidth; ++m) {
      int parent = top_ids[m];
      if (parent < 0 || expanded[m]) continue;
      expanded[m] = true;
      ++expanded_this_iter;

      for (int e = 0; e < kGraphDegree; ++e) {
        int cand = graph[parent * kGraphDegree + e];
        if (cand < 0 || cand >= n) continue;
        float dist = l2_query_point_scalar(dataset, queries, cand, q, dim);
        insert_topm(top_ids, top_dists, expanded, cand, dist);
      }
    }

    if (expanded_this_iter == 0) break;
  }

  for (int k = 0; k < kTopK; ++k) {
    out_ids[q * kTopK + k] = top_ids[k];
    out_dists[q * kTopK + k] = top_dists[k];
  }
}

void search_plain(const float* d_dataset,
                  const int* d_graph,
                  const float* d_queries,
                  int n,
                  int dim,
                  int num_queries,
                  int* d_out_ids,
                  float* d_out_dists)
{
  dim3 block(kThreads);
  dim3 grid((num_queries + block.x - 1) / block.x);
  search_plain_kernel<<<grid, block>>>(
    d_dataset, d_graph, d_queries, n, dim, num_queries, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cagra_repro::plain

