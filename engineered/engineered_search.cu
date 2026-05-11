#include "engineered/engineered_search.cuh"

#include "common/cagra_common.cuh"
#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"

namespace cagra_repro::engineered {

__device__ __forceinline__ unsigned strip_parent(unsigned x) { return x & kIdMask; }
__device__ __forceinline__ unsigned mark_parent(unsigned x) { return x | kParentMask; }
__device__ __forceinline__ bool is_parented(unsigned x) { return (x & kParentMask) != 0; }

__device__ float block_l2_query_to_point_vec4(const float* dataset,
                                              const float* queries,
                                              int point,
                                              int query,
                                              int dim)
{
  __shared__ float partial[kBlockThreads];
  float sum = 0.0f;

  const float* q = queries + static_cast<long long>(query) * dim;
  const float* x = dataset + static_cast<long long>(point) * dim;

  if ((dim & 3) == 0) {
    int dim4 = dim / 4;
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int i = threadIdx.x; i < dim4; i += blockDim.x) {
      float4 a = q4[i];
      float4 b = x4[i];
      float dx = a.x - b.x;
      float dy = a.y - b.y;
      float dz = a.z - b.z;
      float dw = a.w - b.w;
      sum += dx * dx + dy * dy + dz * dz + dw * dw;
    }
  } else {
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float diff = q[d] - x[d];
      sum += diff * diff;
    }
  }

  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  return partial[0];
}

__device__ void shared_hash_clear(int* table)
{
  for (int i = threadIdx.x; i < kHashSize; i += blockDim.x) {
    table[i] = -1;
  }
  __syncthreads();
}

__device__ bool shared_hash_contains_or_insert_thread0(int* table, int value)
{
  int h = static_cast<int>(hash_u32(static_cast<unsigned>(value))) & (kHashSize - 1);
  for (int probe = 0; probe < kHashSize; ++probe) {
    int slot = (h + probe) & (kHashSize - 1);
    int old = table[slot];
    if (old == value) return true;
    if (old == -1) {
      table[slot] = value;
      return false;
    }
  }
  return true;
}

__device__ void insert_topm_flagged(unsigned* ids, float* dists, int cand, float dist)
{
  if (cand < 0) return;

  for (int i = 0; i < kInternalTopM; ++i) {
    if (static_cast<int>(strip_parent(ids[i])) == cand) return;
  }
  if (dist >= dists[kInternalTopM - 1]) return;

  int pos = kInternalTopM - 1;
  while (pos > 0 && dist < dists[pos - 1]) {
    ids[pos] = ids[pos - 1];
    dists[pos] = dists[pos - 1];
    --pos;
  }
  ids[pos] = static_cast<unsigned>(cand);
  dists[pos] = dist;
}

__device__ int select_parent_thread0(unsigned* top_ids)
{
  for (int i = 0; i < kInternalTopM; ++i) {
    if (!is_parented(top_ids[i])) {
      int parent = static_cast<int>(strip_parent(top_ids[i]));
      top_ids[i] = mark_parent(top_ids[i]);
      return parent;
    }
  }
  return -1;
}

__global__ void search_single_cta_kernel(const float* dataset,
                                         const int* graph,
                                         const float* queries,
                                         int n,
                                         int dim,
                                         int num_queries,
                                         int max_iters,
                                         int sw,
                                         int* out_ids,
                                         float* out_dists)
{
  int q = blockIdx.x;
  if (q >= num_queries) return;

  __shared__ unsigned top_ids[kInternalTopM];
  __shared__ float top_dists[kInternalTopM];
  __shared__ int hash_table[kHashSize];
  __shared__ int candidate;
  __shared__ int parent;

  shared_hash_clear(hash_table);

  if (threadIdx.x == 0) {
    for (int i = 0; i < kInternalTopM; ++i) {
      top_ids[i] = static_cast<unsigned>(-1);
      top_dists[i] = CUDART_INF_F;
    }
  }
  __syncthreads();

  for (int s = 0; s < kInternalTopM; ++s) {
    if (threadIdx.x == 0) {
      int cand = static_cast<int>(hash_u32(q * 1009U + s * 9176U)) % n;
      bool seen = shared_hash_contains_or_insert_thread0(hash_table, cand);
      candidate = seen ? -1 : cand;
    }
    __syncthreads();

    float dist = candidate >= 0
                   ? block_l2_query_to_point_vec4(dataset, queries, candidate, q, dim)
                   : CUDART_INF_F;

    if (threadIdx.x == 0 && candidate >= 0) {
      insert_topm_flagged(top_ids, top_dists, candidate, dist);
    }
    __syncthreads();
  }

  // Expand up to sw parents per iteration; max_iters bounds total iterations.
  // Mirrors cuVS search_single_cta.cuh: search_width + max_iterations params.
  for (int iter = 0; iter < max_iters; ++iter) {
    int expanded_this_iter = 0;
    for (int w = 0; w < sw; ++w) {
      if (threadIdx.x == 0) {
        parent = select_parent_thread0(top_ids);
      }
      __syncthreads();
      if (parent < 0) break;
      ++expanded_this_iter;

      for (int e = 0; e < kGraphDegree; ++e) {
        if (threadIdx.x == 0) {
          int cand = graph[parent * kGraphDegree + e];
          bool seen = cand < 0 || cand >= n ||
                      shared_hash_contains_or_insert_thread0(hash_table, cand);
          candidate = seen ? -1 : cand;
        }
        __syncthreads();

        float dist = candidate >= 0
                       ? block_l2_query_to_point_vec4(dataset, queries, candidate, q, dim)
                       : CUDART_INF_F;

        if (threadIdx.x == 0 && candidate >= 0) {
          insert_topm_flagged(top_ids, top_dists, candidate, dist);
        }
        __syncthreads();
      }
    }
    if (expanded_this_iter == 0) break;
  }

  if (threadIdx.x == 0) {
    for (int k = 0; k < kTopK; ++k) {
      out_ids[q * kTopK + k] = static_cast<int>(strip_parent(top_ids[k]));
      out_dists[q * kTopK + k] = top_dists[k];
    }
  }
}

__global__ void search_multi_cta_kernel(const float* dataset,
                                        const int* graph,
                                        const float* queries,
                                        int n,
                                        int dim,
                                        int num_queries,
                                        int max_iters,
                                        int sw,
                                        int* partial_ids,
                                        float* partial_dists)
{
  int global_block = blockIdx.x;
  int q = global_block / kMultiCtaPerQuery;
  int cta_lane = global_block % kMultiCtaPerQuery;
  if (q >= num_queries) return;

  __shared__ unsigned top_ids[kInternalTopM];
  __shared__ float top_dists[kInternalTopM];
  __shared__ int hash_table[kHashSize];
  __shared__ int candidate;
  __shared__ int parent;

  shared_hash_clear(hash_table);

  if (threadIdx.x == 0) {
    for (int i = 0; i < kInternalTopM; ++i) {
      top_ids[i] = static_cast<unsigned>(-1);
      top_dists[i] = CUDART_INF_F;
    }
  }
  __syncthreads();

  for (int s = 0; s < kInternalTopM; ++s) {
    if (threadIdx.x == 0) {
      int cand =
        static_cast<int>(hash_u32(q * 1009U + cta_lane * 65537U + s * 9176U)) % n;
      bool seen = shared_hash_contains_or_insert_thread0(hash_table, cand);
      candidate = seen ? -1 : cand;
    }
    __syncthreads();

    float dist = candidate >= 0
                   ? block_l2_query_to_point_vec4(dataset, queries, candidate, q, dim)
                   : CUDART_INF_F;

    if (threadIdx.x == 0 && candidate >= 0) {
      insert_topm_flagged(top_ids, top_dists, candidate, dist);
    }
    __syncthreads();
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    int expanded_this_iter = 0;
    for (int w = 0; w < sw; ++w) {
      if (threadIdx.x == 0) {
        parent = select_parent_thread0(top_ids);
      }
      __syncthreads();
      if (parent < 0) break;
      ++expanded_this_iter;

      for (int e = 0; e < kGraphDegree; ++e) {
        if (threadIdx.x == 0) {
          int cand = graph[parent * kGraphDegree + e];
          bool seen = cand < 0 || cand >= n ||
                      shared_hash_contains_or_insert_thread0(hash_table, cand);
          candidate = seen ? -1 : cand;
        }
        __syncthreads();

        float dist = candidate >= 0
                       ? block_l2_query_to_point_vec4(dataset, queries, candidate, q, dim)
                       : CUDART_INF_F;

        if (threadIdx.x == 0 && candidate >= 0) {
          insert_topm_flagged(top_ids, top_dists, candidate, dist);
        }
        __syncthreads();
      }
    }
    if (expanded_this_iter == 0) break;
  }

  if (threadIdx.x == 0) {
    int base = (q * kMultiCtaPerQuery + cta_lane) * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      partial_ids[base + k] = static_cast<int>(strip_parent(top_ids[k]));
      partial_dists[base + k] = top_dists[k];
    }
  }
}

__global__ void merge_partial_results_kernel(const int* partial_ids,
                                             const float* partial_dists,
                                             int num_queries,
                                             int* out_ids,
                                             float* out_dists)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;

  int ids[kTopK];
  float dists[kTopK];
  for (int k = 0; k < kTopK; ++k) {
    ids[k] = -1;
    dists[k] = CUDART_INF_F;
  }

  for (int lane = 0; lane < kMultiCtaPerQuery; ++lane) {
    int base = (q * kMultiCtaPerQuery + lane) * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      int cand = partial_ids[base + k];
      float dist = partial_dists[base + k];

      bool dup = false;
      for (int j = 0; j < kTopK; ++j) {
        if (ids[j] == cand) dup = true;
      }
      if (dup || dist >= dists[kTopK - 1]) continue;

      int pos = kTopK - 1;
      while (pos > 0 && dist < dists[pos - 1]) {
        ids[pos] = ids[pos - 1];
        dists[pos] = dists[pos - 1];
        --pos;
      }
      ids[pos] = cand;
      dists[pos] = dist;
    }
  }

  for (int k = 0; k < kTopK; ++k) {
    out_ids[q * kTopK + k] = ids[k];
    out_dists[q * kTopK + k] = dists[k];
  }
}

void search_engineered(const float* d_dataset,
                       const int* d_graph,
                       const float* d_queries,
                       int n,
                       int dim,
                       int num_queries,
                       SearchPlan plan,
                       int* d_out_ids,
                       float* d_out_dists)
{
  SearchAlgo algo = choose_algo(plan.algo, num_queries);
  int max_iters = plan.max_iterations > 0 ? plan.max_iterations : kMaxSearchIters;
  int sw        = plan.search_width   > 0 ? plan.search_width   : kSingleCtaSearchWidth;

  if (algo == SearchAlgo::SingleCta) {
    search_single_cta_kernel<<<num_queries, kBlockThreads>>>(
      d_dataset, d_graph, d_queries, n, dim, num_queries,
      max_iters, sw, d_out_ids, d_out_dists);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  int* d_partial_ids = nullptr;
  float* d_partial_dists = nullptr;
  int partial_count = num_queries * kMultiCtaPerQuery * kTopK;
  CUDA_CHECK(cudaMalloc(&d_partial_ids, sizeof(int) * partial_count));
  CUDA_CHECK(cudaMalloc(&d_partial_dists, sizeof(float) * partial_count));

  search_multi_cta_kernel<<<num_queries * kMultiCtaPerQuery, kBlockThreads>>>(
    d_dataset, d_graph, d_queries, n, dim, num_queries,
    max_iters, sw, d_partial_ids, d_partial_dists);
  CUDA_CHECK(cudaGetLastError());

  dim3 block(kBlockThreads);
  dim3 grid((num_queries + block.x - 1) / block.x);
  merge_partial_results_kernel<<<grid, block>>>(
    d_partial_ids, d_partial_dists, num_queries, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_partial_ids));
  CUDA_CHECK(cudaFree(d_partial_dists));
}

}  // namespace cagra_repro::engineered

