#include "engineered/engineered_plan.cuh"

#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"

namespace cagra_repro::engineered {

SearchAlgo choose_algo(SearchAlgo requested, int num_queries)
{
  if (requested != SearchAlgo::Auto) return requested;

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

  // Same idea as cuVS search_plan.cuh:
  // enough query-level parallelism -> single CTA;
  // otherwise use several CTAs per query to spend more GPU work on one query.
  if (kInternalTopM <= 512 && num_queries >= sm_count * 2) {
    return SearchAlgo::SingleCta;
  }
  return SearchAlgo::MultiCta;
}

}  // namespace cagra_repro::engineered

