#pragma once

namespace cagra_repro::engineered {

struct BuildTiming {
  double init_ms         = 0;
  double nn_descent_ms   = 0;  // total across all iters
  double prune_ms        = 0;
  double reverse_ms      = 0;
  double connectivity_ms = 0;
};

void build_graph_engineered(const float* d_dataset,
                            int n, int dim,
                            bool guarantee_connectivity,
                            int* d_graph,
                            int nn_iters = -1,
                            BuildTiming* timing = nullptr);

}  // namespace cagra_repro::engineered
