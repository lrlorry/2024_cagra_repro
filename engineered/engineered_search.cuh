#pragma once

#include "engineered/engineered_plan.cuh"

namespace cagra_repro::engineered {

void search_engineered(const float* d_dataset,
                       const int* d_graph,
                       const float* d_queries,
                       int n,
                       int dim,
                       int num_queries,
                       SearchPlan plan,
                       int* d_out_ids,
                       float* d_out_dists);

}  // namespace cagra_repro::engineered

