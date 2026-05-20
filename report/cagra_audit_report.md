# CAGRA 源码审计最终报告

> **审计基准**：`rapidsai/cuvs` `main`，commit `1fb6b18981ff`，检查日期 2026-05-19。  
> **重要限制**：前几轮报告明确说明，本轮审计没有在本地跑 GPU recall/QPS，因为环境缺少可用的 `cuvs.neighbors` / `cupy`。因此本报告的性能与召回影响均为源码静态审计推断，凡需要实验确认的地方都标为 `needs_test` 或 `TODO`。

---

## 1. Executive Summary

本报告整理了 CAGRA 前 7 轮源码审计结果，并把 Round 5 verifier 的反方意见合并进最终结论。总体判断是：**CAGRA paper 的核心设计思想与 current cuVS source 大体一致，但 current source 已经比 ICDE paper 描述复杂得多，并存在若干 paper/source、docs/source、API/source drift。**

最值得关注的高价值结论有四类。第一，**paper implementation vs current source drift**：论文实验口径中 graph optimization 在 CPU，而 current cuVS `graph::optimize` 的 prune / reverse / merge 主要走 GPU kernel；论文中的 bitonic/radix threshold 与 current source 审计到的 threshold 不一致；paper 描述的 convergence 是 top-M index stable，而 current source 停止条件更接近“没有可扩展 parent”。第二，**API/docs 与 source 默认值不一致**：C++ 默认 build backend 是 heuristic auto-select，C/Python 默认 IVF-PQ；docs 中 degree 默认值与源码默认值不一致。第三，**search 参数语义复杂**：`search_width`、`itopk_size` 在 SINGLE_CTA 和 MULTI_CTA 下不是同一语义，`filtering_rate` 只在 MULTI_CTA 下自动放大 `itopk_size`。第四，**若干 finding 被 verifier 降级**：rank-based pruning 不使用几何 RNG 距离、top-M 不是 heap priority queue、forgettable hash reset 后只恢复 internal top-M，这些主要是 CAGRA 的 GPU-friendly 设计取舍，不应当作 bug。

CAGRA 与 SONG 的核心区别是：SONG 更像“把传统 graph search 的 `q/topk/visited` 搬上 GPU，并通过三阶段解耦并行化距离计算”；CAGRA 则是“重新设计 graph 和 search control”，用 fixed out-degree graph、internal top-M buffer、candidate buffer、warp/team distance、SINGLE_CTA/MULTI_CTA 策略和工程化 hash 管理来提高 GPU 利用率。CAGRA 更强也更复杂，最终报告不应把 paper-level 描述、current cuVS source、docs/API 默认行为混为一谈。

---

## 2. Top 20 Findings

| ID | Severity | Confidence | Type | Summary | Status |
|---|---:|---|---|---|---|
| BUILD-01 | ★★★★☆ | High | paper-current source drift / CPU vs GPU | Paper evaluation describes graph optimization on CPU; current source dispatches prune/reverse/merge through GPU kernels. | Keep |
| BUILD-02 | ★★★★☆ | High | default mismatch / paper vs source / API mismatch | C++ default build is heuristic auto-select, C/Python default IVF-PQ, docs list stale degree defaults. | Keep |
| BUILD-03 | ★★★☆☆ | High | implementation detail / paper nuance | Prune uses rank-based 2-hop detour count, not geometric RNG; verifier downgraded because paper explicitly adopts rank-based approximation. | Design clarification |
| BUILD-04 | ★★★☆☆ | High | semantic ambiguity / edge case | Second-hop D→B membership appears unrestricted by rank; `kDB < kAB` condition is commented out. | Needs small-graph test |
| BUILD-05 | ★★★☆☆ | Medium-High | paper-current source drift | Reverse merge is protected-prefix + reverse insertion, not a literal d/2 interleave. | Keep, downgraded |
| BUILD-06 | ★★★★☆ | High | docs/source mismatch / name vs behavior | `guarantee_connectivity` uses approximate MST-like augmentation, not strict MST. | Keep |
| BUILD-07 | ★★☆☆☆ | High | internal API footgun | Public `guarantee_connectivity` default false; internal `graph::optimize` default true. | Downgraded |
| BUILD-08 | ★★★☆☆ | High | docs clarity / name vs behavior | `ef_construction` maps to build-backend heuristics, not CAGRA prune/search loop width. | Keep |
| SEARCH-01 | ★★★★☆ | High | parameter semantic split | `search_width` means parent count in SINGLE_CTA but CTA fanout lower bound in MULTI_CTA. | Keep |
| SEARCH-02 | ★★★☆☆ | High | performance heuristic | MULTI_CTA automatic `max_iterations` ignores public `search_width`. | Keep |
| SEARCH-03 | ★★★★☆ | High | filtering semantics | `filtering_rate` only auto-enlarges `itopk_size` in MULTI_CTA when known. | Keep |
| SEARCH-04 | ★★★★☆ | High | paper-source drift | Paper says bitonic threshold ≤512; source audit found ≤256 before radix. | Needs all-path verification |
| SEARCH-05 | ★★★★☆ | High | convergence semantics | Source stop condition is no expandable parent / per-CTA stop, not explicit top-M equality. | Keep |
| SEARCH-06 | ★★★☆☆ | Medium | capacity heuristic | MULTI_CTA traversed hash capacity/tombstone behavior may be under-modeled; not confirmed bug. | Medium-confidence TODO |
| SEARCH-07 | ★★☆☆☆ | High | intentional trade-off | Forgettable hash reset restores top-M only; verifier notes this matches paper. | Downgraded |
| SEARCH-08 | ★★★☆☆ | High | AUTO policy drift | AUTO threshold uses `2*SM` rather than paper’s `SM` recommendation. | Keep |
| SEARCH-09 | ★★★☆☆ | High | edge case / docs clarity | MULTI_CTA per-CTA buffer `32+degree` must round to ≤256. | Keep |
| API-01 | ★★★☆☆ | Medium | potential allocation bug | Potential byte-count-as-element-count resize in multi-CTA hash; verify container type first. | High-priority TODO |
| METRIC-01 | ★★★☆☆ | High | metric semantics / edge case | Cosine ranking is semantically correct for nonzero vectors; zero-norm behavior needs test. | Corrected by verifier |
| SONG-01 | N/A | Medium | comparison scope limitation | CAGRA side is source-level; SONG side is paper-level because no SONG official source audit was done. | Scope note |

---

## 3. Source Map

### 3.1 关键 API / Build / Search 文件

| File | Lines | Symbols | Path |
|---|---:|---|---|
| `cpp/include/cuvs/neighbors/cagra.hpp` | 39-224 | `graph_build_params::ace_params`, `index_params` | API / build params |
| `cpp/include/cuvs/neighbors/cagra.hpp` | 268-351 | `search_algo`, `hash_mode`, `search_params` | API / search params |
| `cpp/include/cuvs/neighbors/cagra.hpp` | 396+ | `index<T, IdxT>` | API / index object |
| `cpp/src/neighbors/cagra.cuh` | 244-284 | `optimize`, `build` | C++ API -> detail build |
| `cpp/src/neighbors/cagra.cuh` | 325-385 | `search_with_filtering`, `search` | C++ API -> search |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1607-1835 | IVF-PQ `build_knn_graph` | build backend |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1874-1898 | NN-Descent `build_knn_graph` | build backend |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1903-1931 | `detail::optimize` | build -> graph optimize |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 1983-2164 | `iterative_build_graph` | build/search bridge |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | 2166+ | `detail::build` | build dispatcher |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 174-195 | `kern_make_rev_graph` | graph GPU path |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 203-326 | `kern_fused_prune` | graph GPU prune |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 604+ | MST helpers | connectivity augmentation |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 1595-1698 | `prune_graph_gpu` | graph optimize |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | 1708-1820 | `graph::optimize` | prune/reverse/merge/MST |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 98-133 | `search_plan_impl_base` | algo selection |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 199-245 | `adjust_search_params` | max_iterations/filtering/M |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | 247-372 | `calc_hashmap_params` | hash sizing |
| `cpp/src/neighbors/detail/cagra/cagra_search.cuh` | 44-112 | `search_main_core` | factory + execution |
| `cpp/src/neighbors/detail/cagra/cagra_search.cuh` | 137-204 | `search_main` | dataset/metric dispatch |
| `cpp/src/neighbors/detail/cagra/search_single_cta.cuh` | 106-206 | `single_cta_search::set_params` | SINGLE_CTA buffers/hash |
| `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh` | 117-206 | `multi_cta_search::set_params` | MULTI_CTA buffers/hash |
| `cpp/src/neighbors/detail/cagra/hashmap.hpp` | whole file | `hashmap::insert/remove/search` | visited/traversed hash |
| `cpp/src/neighbors/detail/cagra/compute_distance*.{hpp,cuh}` | whole files | distance descriptors/dispatch | metric path |
| `cpp/src/neighbors/detail/cagra/jit_lto_kernels/*.cuh` | whole files | JIT distance/search kernels | GPU kernel path |

### 3.2 Build 调用链

```text
cuvs::neighbors::cagra::build
  -> cpp/src/neighbors/cagra.cuh:265
  -> detail::build
  -> cpp/src/neighbors/detail/cagra/cagra_build.cuh:2166
  -> build_knn_graph IVF-PQ or NN-Descent or iterative_build_graph
  -> detail::optimize
  -> graph::optimize
  -> prune_graph_gpu / make_reverse_graph_gpu / merge_graph_gpu / optional mst_optimization
  -> kern_fused_prune / kern_make_rev_graph / merge + MST kernels
```

ACE build branch:

```text
cuvs::neighbors::cagra::build
  -> holds_alternative<graph_build_params::ace_params>
  -> detail::build_ace
  -> partition/gather/reorder/write helpers
  -> per-partition CAGRA index build/optimize
  -> merged output index
```

### 3.3 Search 调用链

```text
cuvs::neighbors::cagra::search / search_with_filtering
  -> cpp/src/neighbors/cagra.cuh:325
  -> detail::search_main
  -> cagra_search.cuh:137
  -> dataset type dispatch: standard / VPQ / CosineExpanded norms path
  -> search_main_core
  -> factory creates one of:
       single_cta_search::search
       multi_cta_search::search
       multi_kernel_search
  -> set_params
  -> operator()
  -> select_and_run / JIT or non-JIT kernels
```

### 3.4 Metric dispatch path

```text
index.metric() + dataset descriptor
  -> cagra_search.cuh::search_main
  -> search_plan_impl descriptor/cache setup
  -> compute_distance_standard / compute_distance_vpq
  -> jit_lto_kernels/dist_op_impl.cuh
  -> optional apply_normalization_standard_impl.cuh for CosineExpanded output transform
```

### 3.5 CPU/GPU path 分叉点

| Stage | CPU/Host path | GPU/Device path | Notes |
|---|---|---|---|
| API entry | C++/C/Python wrappers create params/index/search calls | actual kernels execute on CUDA | public params can differ by API |
| Build backend | host dispatches backend | IVF-PQ / NN-Descent / iterative GPU work | C++ default auto-select differs from C/Python |
| Graph optimize | host validates/allocates/dispatches | `prune_graph_gpu`, reverse/merge kernels, optional MST kernels | current source differs from paper CPU optimize wording |
| Search plan | host chooses algo/hash params | SINGLE_CTA / MULTI_CTA / MULTI_KERNEL kernels | parameters rewritten per algo |
| Metric | host stores metric and prepares norms/descriptors | distance op and normalization kernels | cosine ranks by base-norm-normalized dot |
| Filtering/hash | host dispatches filter type | device hashmap/filter logic | `filtering_rate` auto-estimation limited |

---

## 4. Graph Construction Audit

### 4.1 Initial graph and defaults

The paper’s main construction story is: build an initial kNN graph, usually by NN-Descent, sort neighbor rows by distance, then optimize the graph by rank-based reordering and reverse-edge addition. Current cuVS source supports more paths: IVF-PQ, NN-Descent, iterative build, and ACE. This creates an important default mismatch:

- C++ `index_params{}`: `graph_degree=64`, `intermediate_graph_degree=128`, `graph_build_params=monostate`, with backend selected heuristically.
- C API / Python: default build algorithm appears to be IVF-PQ, with degree defaults 64/128.
- Docs table was reported to list `graph_degree=32`, `intermediate_graph_degree=64`.

**Finding:** BUILD-02.

### 4.2 Rank-based pruning

Current `kern_fused_prune` does not evaluate geometric RNG inequalities. It uses rank-based 2-hop membership counts. A verifier correction is crucial here: this should not be labeled as a bug, because CAGRA paper explicitly adopts rank-based reordering to avoid the distance computation and memory overhead of distance-based reordering.

Pseudocode reconstructed from the audit:

```text
for each node A:
  load A's neighbor row N[0..K-1]
  detour_count[k] = 0

  for kAD in 0..K-2:
    D = N[kAD]
    for B2 in row(D):
      for kAB in kAD+1..K-1:
        B = N[kAB]
        if B == B2:
          detour_count[kAB] += 1
          break

  output_degree times:
    select smallest (detour_count, original_rank)
    mark duplicate selected nodes invalid
```

The remaining uncertainty is whether the second-hop membership should be rank-restricted. The audit found `kDB < kAB` commented out, so current source appears to allow any-rank second-hop membership.

**Findings:** BUILD-03, BUILD-04.

### 4.3 Reverse graph and merge

The paper describes reverse edge addition as improving reachability and reducing strongly connected components. Current source has a reverse graph kernel and a merge kernel. However, the merge is not a literal paper-style `d/2` interleave. The audit describes it as protecting a prefix and inserting reverse edges around that boundary.

**Finding:** BUILD-05.

### 4.4 Connectivity / MST

`guarantee_connectivity` should be documented carefully. The audited source is better described as approximate connectivity augmentation or approximate degree-constrained MST-like augmentation, not a strict minimum spanning tree. It does not appear to optimize edge weights globally. Under disconnected input, fallback can connect to scanned main-cluster nodes.

**Finding:** BUILD-06.

Verifier note: public default appears to be false; the internal default true is a footgun, not a major public behavior mismatch.

**Finding:** BUILD-07.

---

## 5. Search Plan Audit

### 5.1 Parameter flow

```text
public search_params
  -> cagra.cuh::search / search_with_filtering
  -> cagra_search.cuh::search_main_core
  -> factory.cuh::create
  -> search_plan_impl_base selects algo
  -> adjust_search_params
  -> calc_hashmap_params
  -> SINGLE_CTA / MULTI_CTA / MULTI_KERNEL set_params
  -> kernels
```

Key defaults:

```text
itopk_size = 64
search_width = 1
max_iterations = 0
algo = AUTO
hashmap_mode = AUTO
filtering_rate = -1
```

### 5.2 SINGLE_CTA vs MULTI_CTA semantics

| Parameter | SINGLE_CTA | MULTI_CTA |
|---|---|---|
| `search_width` | number of parents expanded per iteration | lower bound on CTAs per query; per CTA `p=1` |
| `itopk_size` | internal top-M size | global M converted to CTA fanout; per CTA M=32 |
| candidate list | `search_width * graph_degree` | per CTA `graph_degree`; global approximately `cta_count * degree` |
| auto `max_iterations` | `M / p + reachability_extra` | `32 / 1 + reachability_extra` |
| filtering adjustment | not automatic | adjusts global M, increasing CTA count |

**Findings:** SEARCH-01, SEARCH-02, SEARCH-03.

### 5.3 Top-M and convergence

CAGRA’s search state is not a traditional heap priority queue. It is a fixed internal top-M buffer and candidate buffer, updated by sort/merge. This is an expected design, not a bug. However, paper/source drift appears in the stopping condition: the paper describes top-M indices converging, while current source appears to stop when no expandable parent exists or max iterations are reached.

**Finding:** SEARCH-05.

### 5.4 Sorting threshold

The paper states that small candidate buffers up to 512 use a single-warp bitonic sort; larger lists use radix. The audited current source threshold is 256. This is a high-value source drift and needs all-path confirmation.

**Finding:** SEARCH-04.

### 5.5 Hash / visited

CAGRA uses open-addressing hash structures. The important verifier correction is that small/forgettable hash reset restoring only internal top-M is intentional and described in the paper. It should be treated as a memory/computation trade-off, not a bug.

The medium-confidence concern is MULTI_CTA traversed hash sizing and tombstone behavior. It should not be called a confirmed bug until the exact insert/remove semantics and container storage are verified.

**Findings:** SEARCH-06, SEARCH-07, API-01.

---

## 6. Metric / Distance Audit

### 6.1 Internal key and output semantics

| Metric | Internal key | Internal top-M direction | Output |
|---|---|---|---|
| L2Expanded | `sum((q-x)^2)` | smaller is better | scaled L2 distance |
| InnerProduct | `-dot(q,x)` | smaller is better | positive IP score after sign flip |
| CosineExpanded | `-dot(q,x)/||x||` | smaller is better | `1 + key/||q|| = 1 - cos(q,x)` |

The earlier concern that “cosine only normalizes base vectors” should be removed as a bug. Sorting without query norm is valid for a fixed query because `||q||` is a query-constant. The real TODO is zero-norm behavior and VPQ/non-standard path consistency.

**Finding:** METRIC-01.

### 6.2 Dtype notes

The audit found standard descriptor support for `float / half / int8_t / uint8_t`, with distance accumulation in `float`. Integer types are mapped/scaled before distance. Team size support needs additional confirmation, especially whether `team_size=4` is actually instantiated in all paths.

---

## 7. Paper Claim vs Source

| Paper claim | Source behavior | Judgment |
|---|---|---|
| Fixed out-degree directed graph | Source graph/search revolves around `graph_degree` fixed rows | Consistent |
| No hierarchy, random sampling | Search init uses random sampling; no HNSW hierarchy | Consistent |
| Initial graph uses NN-Descent | Source also supports IVF-PQ/iterative/ACE and different defaults | Paper/source/API drift |
| Rank-based reordering | Source uses rank-based 2-hop counts | Mostly consistent |
| Geometric RNG not used | Source does not compute geometry in prune | Expected under rank-based design |
| Reverse edge addition | Source has reverse graph helper | Consistent, with implementation details |
| Merge pruned + reversed graph | Source protected-prefix insertion differs from literal interleave | Partial drift |
| Top-M + candidate buffer | Source fixed buffers and sort/merge | Consistent |
| Candidate list length `p*d` | SINGLE_CTA consistent; MULTI_CTA specializes per CTA | Specialization/docs nuance |
| Warp splitting/team distance | Source has team/block dispatch; team_size=4 needs confirmation | Mostly consistent |
| Small buffer bitonic vs radix | Source threshold appears 256, paper says 512 | Drift |
| Forgettable hash reset restores top-M | Source matches paper | Consistent trade-off |
| MSB parent flag | Source uses MSB; also hash tombstone uses MSB | Consistent with extra caveat |
| Convergence by top-M stable | Source stops by parent availability / max iteration | Drift |

---

## 8. CPU vs GPU Path

| Stage | CPU/Host | GPU/Device | Equivalence / Risk |
|---|---|---|---|
| Regular build dispatch | Host selects backend | IVF-PQ / NN-Descent / iterative GPU work | API defaults can select different backends |
| Graph optimize | Host validates/allocates | prune/reverse/merge GPU kernels | Current source differs from paper CPU optimize wording |
| Build prune | No CPU build prune path in audited current source | `kern_fused_prune` | No CPU prune equivalence claim |
| Extend reorder | CPU `add_nodes` reorder path | build prune GPU path | Not guaranteed equivalent |
| Reverse edge | host output storage | GPU reverse kernel | fixed-degree truncation/tie-order need test |
| MST helper | optional CPU helper exists | default GPU helper | exact edge set not guaranteed equivalent |
| Search plan | host computes plan and buffers | kernels execute | algo changes parameter meaning |
| Distance metric | host stores metric/norm metadata | GPU distance kernels/postprocess | brute-force reference must mimic source semantics |

---

## 9. CAGRA vs SONG

### 9.1 CAGRA query flow

```text
Input query q + fixed-out-degree CAGRA graph
  -> random sampling initialization
  -> internal top-M + candidate buffer
  -> loop:
       sort/merge top-M
       pick unexpanded parents
       mark parent with MSB
       load neighbors
       hash/filter visited
       compute distances with team/warp kernels
       stop if no expandable parent or max_iterations
  -> output top-k
```

### 9.2 SONG query flow

```text
Input query p + graph G
  -> q/topk/visited initialization
  -> loop:
       candidate locating: pop q, fetch neighbors
       bulk distance computation: GPU warp reduction
       data-structure maintenance: update q/topk/visited
       stop if popped candidate worse than top-k boundary
  -> output topk
```

### 9.3 Comparison table

| Dimension | SONG | CAGRA | Judgment |
|---|---|---|---|
| Core strategy | Accelerate traditional graph search | Graph/search co-design | CAGRA more GPU-native |
| Graph construction | Not core contribution; loads prebuilt graph | Builds/optimizes CAGRA graph | CAGRA advantage |
| Search state | `q/topk/visited` | internal top-M + candidate buffer | CAGRA more batch-friendly |
| PQ/top-M | bounded priority queue | fixed buffer sort/merge | CAGRA better for GPU |
| Distance computation | bulk distance + warp reduction | team/warp split kernels | CAGRA more specialized |
| Visited | hash/Bloom/Cuckoo + selected insertion/deletion | open hash + forgettable reset + MSB + traversed hash | CAGRA more engineered, harder to verify |
| Single query | limited; multi-query in warp helps batches | MULTI_CTA can fan out one query | CAGRA advantage |
| Large batch | multi-query processing | SINGLE_CTA large-batch policy | CAGRA systematic |
| Out-of-GPU-memory | random projection/hashing in paper | current source has more backends; paper core less focused | no clear winner from current audit |
| Complexity | lower | much higher | SONG simpler |

### 9.4 Scope limitation

SONG comparison is paper-level because no official SONG source audit was performed. CAGRA conclusions are source-level for `cuvs@1fb6b18981ff`.

---

## 10. Finding Classification

### Confirmed paper/source or docs/source drift

- BUILD-01: paper CPU optimize vs current GPU optimize.
- BUILD-02: C++/C/Python/docs defaults mismatch.
- BUILD-05: reverse merge implementation differs from literal paper interleave.
- SEARCH-03: `filtering_rate` docs too broad.
- SEARCH-04: bitonic/radix threshold paper/source drift.
- SEARCH-05: convergence semantics drift.
- SEARCH-08: AUTO threshold heuristic drift.

### Design trade-offs, not bugs

- BUILD-03: rank-based pruning not geometric RNG.
- SEARCH-07: forgettable hash restores top-M only.
- Top-M buffer is not traditional PQ.
- Cosine omits query norm during ranking because it is query-constant.

### Potential issues requiring verification

- BUILD-04: second-hop rank condition commented out.
- SEARCH-06: traversed hash capacity/tombstone behavior.
- API-01: possible byte-count-as-element-count allocation.
- Reverse graph atomic truncation and nondeterministic tie retention.
- Team size 4 support.
- VPQ cosine norm handling.

### Low-priority / footguns

- BUILD-07: internal `graph::optimize` default true vs public false.
- G15-style iterative build only final connectivity guarantee; likely expected.

---

## 11. Reproduction Tests

See `cagra_repro_tests.md` for detailed test designs. The highest-value tests are:

1. API defaults: compare C++, C, Python default params.
2. Current graph optimize execution: NVTX/Nsight verify prune/reverse/merge GPU path.
3. Prune second-hop rank: hand-written 8-node graph.
4. Reverse merge layout: small ring graph.
5. MST fallback: disconnected components.
6. MULTI_CTA search_width/M rewrite: inspect plan and kernel config.
7. filtering_rate behavior: SINGLE_CTA vs MULTI_CTA.
8. bitonic/radix threshold: candidate sizes 256/257/512/513.
9. convergence semantics: top-M changes but no expandable parent.
10. cosine zero-norm behavior.

---

## 12. TODO

1. Re-clone current `rapidsai/cuvs` and verify all file paths/line numbers against `1fb6b18981ff` or update commit hash.
2. Run GPU build/search tests once `cuvs.neighbors` and `cupy` are available.
3. Confirm whether all launcher paths use bitonic threshold 256.
4. Confirm `multi_cta` hashmap storage type and `resize()` units before labeling API-01 as bug.
5. Verify `traversed_hash` insert/remove semantics and tombstone behavior with counters.
6. Test second-hop rank condition with a small graph.
7. Test reverse-edge atomic truncation stability across repeated runs.
8. Test zero-norm cosine and VPQ cosine path.
9. Locate and audit an official or high-quality SONG implementation if source-level CAGRA-vs-SONG claims are needed.
10. Split final maintainer-facing issue list into: confirmed bug, docs issue, design note, benchmark TODO.
