#pragma once

#include <cuda_runtime.h>

namespace cagra_repro {

__device__ __forceinline__ unsigned hash_u32(unsigned x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ inline float l2_point_point_scalar(const float* dataset, int a, int b, int dim)
{
  float acc = 0.0f;
  const float* x = dataset + static_cast<long long>(a) * dim;
  const float* y = dataset + static_cast<long long>(b) * dim;
  for (int d = 0; d < dim; ++d) {
    float diff = x[d] - y[d];
    acc += diff * diff;
  }
  return acc;
}

__device__ inline float l2_query_point_scalar(const float* dataset,
                                              const float* queries,
                                              int point,
                                              int query,
                                              int dim)
{
  float acc = 0.0f;
  const float* q = queries + static_cast<long long>(query) * dim;
  const float* x = dataset + static_cast<long long>(point) * dim;
  for (int d = 0; d < dim; ++d) {
    float diff = q[d] - x[d];
    acc += diff * diff;
  }
  return acc;
}

template <int K>
__device__ void insert_sorted_unique(int (&ids)[K], float (&dists)[K], int cand, float dist)
{
  if (cand < 0) return;
  for (int i = 0; i < K; ++i) {
    if (ids[i] == cand) return;
  }
  if (dist >= dists[K - 1]) return;

  int pos = K - 1;
  while (pos > 0 && dist < dists[pos - 1]) {
    ids[pos] = ids[pos - 1];
    dists[pos] = dists[pos - 1];
    --pos;
  }
  ids[pos] = cand;
  dists[pos] = dist;
}

template <int K>
__device__ bool contains_id(const int (&ids)[K], int cand)
{
  for (int i = 0; i < K; ++i) {
    if (ids[i] == cand) return true;
  }
  return false;
}

}  // namespace cagra_repro

