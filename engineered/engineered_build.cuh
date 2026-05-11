#pragma once

namespace cagra_repro::engineered {

void build_graph_engineered(const float* d_dataset,
                            int n,
                            int dim,
                            bool guarantee_connectivity,
                            int* d_graph);

}  // namespace cagra_repro::engineered

