#pragma once

namespace cagra_repro::engineered {

enum class SearchAlgo {
  Auto,
  SingleCta,
  MultiCta,
};

struct SearchPlan {
  SearchAlgo algo = SearchAlgo::Auto;
  int internal_top_m = 64;
  int search_width = 2;
  int max_iterations = 32;
  int small_hash_reset_interval = 4;
  int multi_cta_per_query = 4;
};

SearchAlgo choose_algo(SearchAlgo requested, int num_queries);

}  // namespace cagra_repro::engineered

