# CAGRA Source Map and Call Chains

> Source baseline: `rapidsai/cuvs` `main`, commit `1fb6b18981ff`. This file contains only source map and call-chain information; behavioral findings are in `cagra_audit_report.md`.

---

## 1. Critical files / functions / kernels

| File | Lines | Main symbols | Called by | Calls | Path |
|---|---:|---|---|---|---|
| `cpp/include/cuvs/neighbors/cagra.hpp` | 39-224 | `graph_build_params::ace_params`, `index_params` | C++ public API / wrappers | parameter structs | API/build |
| `cpp/include/cuvs/neighbors/cagra.hpp` | 268-351 | `search_algo`, `hash_mode`, `search_params` | search API / plan | parameter structs | API/search |
| `cpp/include/cuvs/neighbors/cagra.hpp` | 396+ | `index<T, IdxT>` | build/search/update/serialize | dataset/graph/norm accessors | API/index |
| `cpp/src/neighbors/cagra.cuh` | 102-236 | `build_knn_graph`, `sort_knn_graph` | public C++ API instantiations | `detail::build_knn_graph`, `graph::sort_knn_graph` | API/build |
| `cpp/src/neighbors/cagra.cuh` | 244-284 | `optimize`, `build` | public API / C API | `detail::optimize`, `detail::build`, `detail::build_ace` | API/build |
| `cpp/src/neighbors/cagra.cuh` | 325-385 | `search_with_filtering`, `search` | public API / wrappers | `detail::search_main`, filter dispatch | API/search |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 54-71 | `check_graph_degree` | build/optimize | validation helpers | build |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 75-520 | ACE helpers | `build_ace` | partition/gather/reorder/disk helpers | build/ACE |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1102+ | `build_ace` | `cagra::build` ACE branch | sub-index build, `optimize` | build/ACE |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1537-1605 | `write_to_graph`, `refine_host_and_write_graph` | IVF-PQ build path | refine/write helpers | build |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1607-1835 | IVF-PQ `build_knn_graph` | `detail::build` | `ivf_pq::build`, `ivf_pq::search`, refine/write | build |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1874-1898 | NN-Descent `build_knn_graph` | `detail::build` | `nn_descent::build`, `graph::sort_knn_graph` | build |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1903-1931 | `detail::optimize` | `cagra::optimize`, `detail::build` | `graph::optimize` | graph optimize |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1983-2164 | `iterative_build_graph` | `detail::build` | `search`, `optimize` | build/search bridge |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 2166+ | `detail::build` | `cagra::build` | IVF-PQ / NN-Descent / iterative / ACE / optimize | build dispatcher |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 174-195 | `kern_make_rev_graph` | reverse graph helper | CUDA kernel body | graph/GPU |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 203-326 | `kern_fused_prune` | `prune_graph_gpu` | detour count/select helpers | graph/GPU prune |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 604+ | MST helpers | `mst_optimization` | MST update kernels/helpers | graph connectivity |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 1200-1580 approx | `mst_optimization` | `graph::optimize` | CPU/GPU MST helpers | graph connectivity |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 1595-1698 | `prune_graph_gpu` | `graph::optimize` | `kern_fused_prune` | graph/GPU prune |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 1708-1820 | `graph::optimize` | `detail::optimize` | prune/reverse/merge/optional MST | graph optimize |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 98-133 | `search_plan_impl_base` | factory/search classes | algo selection / params copy | search plan |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 135-182 | `search_plan_impl<...>` | single/multi/multikernel plans | descriptor/cache/filter setup | search plan |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 199-245 | `adjust_search_params` | plan constructor | max_iterations/filtering/M | search plan |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 247-372 | `calc_hashmap_params` | plan constructor | hash sizing | search/hash |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 381-425 | `check_params` | plan constructor/check | validation | search plan |
| `cpp/src/neighbors/detail/cagra/cagra_search.cuh` | 44-112 | `search_main_core` | `search_main` | factory plan, batched plan execution | search |
| `cpp/src/neighbors/detail/cagra/cagra_search.cuh` | 137-204 | `search_main` | `cagra::search` | dataset type dispatch, norms handling | search/metric |
| `cpp/src/neighbors/detail/cagra/factory.cuh` | whole file | search plan factory | `search_main_core` | single/multi/multikernel plan creation | search |
| `cpp/src/neighbors/detail/cagra/search_single_cta.cuh` | 106-206 | `single_cta_search::search::set_params` | factory plan | buffer/hash/workspace setup | SINGLE_CTA |
| `cpp/src/neighbors/detail/cagra/search_single_cta.cuh` | 208+ | `single_cta_search::search::operator()` | `search_main_core` | `select_and_run` | SINGLE_CTA kernel path |
| `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh` | 117-206 | `multi_cta_search::search::set_params/check` | factory plan | intermediate buffer/hash setup | MULTI_CTA |
| `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh` | 208+ | `multi_cta_search::search::operator()` | `search_main_core` | `select_and_run`, `_cuann_find_topk` | MULTI_CTA kernel path |
| `cpp/src/neighbors/detail/cagra/search_multi_kernel.cuh` | whole file | multi-kernel search plan | factory plan | setup/random/distance/filter/topk kernels | MULTI_KERNEL |
| `cpp/src/neighbors/detail/cagra/search_single_cta_kernel*.cuh` | whole files | single-CTA kernel launch/select templates | `search_single_cta.cuh` | CUDA/JIT kernels | search/GPU |
| `cpp/src/neighbors/detail/cagra/search_multi_cta_kernel.cuh` | whole file | multi-CTA kernel launch/select templates | `search_multi_cta.cuh` | CUDA/JIT kernels | search/GPU |
| `cpp/src/neighbors/detail/cagra/jit_lto_kernels/*.cuh`, `*.cu.in` | whole files | JIT kernel implementations/templates | planner/launcher | search/filter/distance/normalization | search/metric/GPU |
| `cpp/src/neighbors/detail/cagra/compute_distance*.{hpp,cuh}` | whole files | dataset descriptors, standard/VPQ distance dispatch | search/build kernels | distance op / normalization helpers | metric |
| `cpp/src/neighbors/detail/cagra/hashmap.hpp` | whole file | device hash helpers | search kernels / plan hash config | open-addressing helpers | hash/visited |
| `cpp/src/neighbors/detail/cagra/sample_filter*.{cuh,hpp}` | whole files | filter/device filter wrappers | search plan/kernels | bitset/filter helpers | filtering |
| `cpp/src/neighbors/cagra*.cu`, C API files | whole files | explicit instantiations / C wrappers | Python/Go/Rust/C callers | C++ `cagra::*` APIs | API |

---

## 2. Public API -> Build path

```text
cuvs::neighbors::cagra::build
  -> cpp/src/neighbors/cagra.cuh:265
  -> detail::build
  -> cpp/src/neighbors/detail/cagra/cagra_build.cuh:2166
  -> backend branch:
       IVF-PQ build_knn_graph    (cagra_build.cuh:1607-1835)
       NN-Descent build_knn_graph(cagra_build.cuh:1874-1898)
       iterative_build_graph     (cagra_build.cuh:1983-2164)
       build_ace                 (cagra_build.cuh:1102+)
  -> detail::optimize            (cagra_build.cuh:1903-1931)
  -> graph::optimize             (graph_core.cuh:1708-1820)
  -> optional mst_optimization
  -> prune_graph_gpu
  -> make_reverse_graph_gpu
  -> merge_graph_gpu
```

---

## 3. Public API -> Search path

```text
cuvs::neighbors::cagra::search / search_with_filtering
  -> cpp/src/neighbors/cagra.cuh:325-385
  -> detail::search_main
  -> cpp/src/neighbors/detail/cagra/cagra_search.cuh:137-204
  -> search_main_core
  -> cpp/src/neighbors/detail/cagra/cagra_search.cuh:44-112
  -> factory creates one of:
       single_cta_search::search
       multi_cta_search::search
       multi_kernel_search
  -> set_params
  -> operator()
  -> select_and_run / JIT kernels
  -> write neighbors/distances
```

---

## 4. Metric dispatch path

```text
index.metric() + dataset descriptor
  -> cagra_search.cuh::search_main
  -> if CosineExpanded: require/compute dataset norms
  -> search_plan_impl descriptor/cache setup
  -> compute_distance_standard or compute_distance_vpq
  -> jit_lto_kernels/dist_op_impl.cuh
  -> optional apply_normalization_standard_impl.cuh
```

Metric behavior summary:

| Metric | Internal key | Ranking direction | Output transform |
|---|---|---|---|
| L2Expanded | squared L2 | smaller is better | scaled distance |
| InnerProduct | negative dot | smaller is better | sign-flipped score |
| CosineExpanded | `-dot(q,x)/||x||` | smaller is better | `1 + key/||q||` |

---

## 5. CPU/GPU path branch points

| Branch point | Host path | Device path | Note |
|---|---|---|---|
| API entry | C++/C/Python wrappers create params/index/search calls | CUDA kernels execute the actual build/search | Language defaults may differ |
| Build backend | host selects backend | IVF-PQ / NN-Descent / iterative GPU work | C++ auto vs C/Python IVF-PQ default |
| ACE | host/disk partition/reorder/metadata helpers | per-partition GPU build/optimize/search | not fully audited |
| Graph optimize | host validates extents and dispatches | GPU prune/reverse/merge kernels | current source differs from paper CPU optimization wording |
| MST helper | CPU helper exists | default GPU helper | exact edge set equivalence not established |
| Search plan | host computes plan, buffer sizes, algo | SINGLE_CTA / MULTI_CTA / MULTI_KERNEL kernels | params rewritten per algo |
| Metric | host stores metric/norm metadata | distance op and post-normalization kernels | cosine ranking correct for nonzero vectors |
| Filtering/hash | host selects filter overload and hash mode | device hash/filter/sample-filter code | filtering_rate auto behavior limited |
