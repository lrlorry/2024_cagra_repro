#pragma once

namespace cagra_repro::plain {

void search_plain(const float* d_dataset,
                  const int* d_graph,
                  const float* d_queries,
                  int n,
                  int dim,
                  int num_queries,
                  int* d_out_ids,
                  float* d_out_dists);

}  // namespace cagra_repro::plain

