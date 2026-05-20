# CAGRA Reproduction Tests and Suggested Unit Tests

> Source baseline: `rapidsai/cuvs` `main`, commit `1fb6b18981ff`. These are test designs, not executed results. Each test maps to one or more findings in `cagra_findings_table.csv`.

---

## 1. API defaults across C++, C, Python

**Test target:** verify language/API/docs default mismatches.  
**Corresponding findings:** BUILD-02.

**Setup:**
- C++: construct `cuvs::neighbors::cagra::index_params{}`.
- C API: call `cuvsCagraIndexParamsCreate`.
- Python: construct `cagra.IndexParams()`.

**Parameters:** default only.

**Expected behavior:**
- C++ default: `graph_degree=64`, `intermediate_graph_degree=128`, `graph_build_params=monostate` / heuristic backend.
- C/Python default: build algo IVF-PQ, degree 64/128.
- Docs may still list stale degree values; verify current docs.

**Pass/Fail:** test passes if printed defaults match source audit and docs mismatch is documented.

---

## 2. Graph optimize CPU vs GPU execution

**Test target:** verify current prune/reverse/merge run on GPU.  
**Corresponding findings:** BUILD-01.

**Setup:** build or optimize a small CAGRA graph with NVTX/Nsight Systems enabled.

**Small graph:** 1K random vectors, degree 32/64.

**Parameters:** default build; also call public `cagra::optimize` on a host graph.

**Expected behavior:** timeline shows calls/kernels corresponding to `prune_graph_gpu`, reverse graph, merge graph; no CPU prune fallback.

**Pass/Fail:** source and profiler agree current source uses GPU for graph optimization.

---

## 3. Rank-based prune vs geometric RNG

**Test target:** show prune uses rank-based 2-hop membership rather than geometric distances.  
**Corresponding findings:** BUILD-03.

**Small graph construction:**
- Hand-write a graph with node `A=0` and neighbors `[1,2,3,4]`.
- Provide vector distances that disagree with row rank order.
- Keep neighbor rows sorted in one run, deliberately permuted in another.

**Parameters:** `input_degree=4`, `output_degree=2 or 3`.

**Expected behavior:** output changes with neighbor-row order/rank, not with geometric distance comparisons in prune.

**Pass/Fail:** if two graphs with same neighbor set but different row rank order produce different pruned edges, rank dependence is confirmed.

---

## 4. Second-hop rank condition

**Test target:** verify whether second-hop `D->B` membership is any-rank or rank-limited.  
**Corresponding findings:** BUILD-04.

**Small graph:**

```text
0: [1,2,3,4]
1: [6,7,4,2]
2: [...]
3: [...]
4: [...]
```

**Parameters:** `output_degree=3`.

**Expected behavior:**
- If source uses any-rank second-hop membership, node `2` or `4` can receive detour count from row 1 even when its second-hop rank is late.
- If strict `kDB < kAB` were enforced, selected ordering may differ.

**Pass/Fail:** compare source output with a reference implementation of both semantics.

---

## 5. Reverse graph truncation and determinism

**Test target:** verify reverse graph degree cap and atomic tie stability.  
**Corresponding findings:** BUILD-05 and reverse-truncation TODO.

**Small graph:** 8 nodes where all nodes point to 0 as first neighbor:

```text
1 -> 0
2 -> 0
3 -> 0
...
7 -> 0
```

**Parameters:** `rev_graph_degree=2`.

**Expected behavior:** `rev_graph[0]` contains only two incoming sources. Which two may depend on atomic ordering unless tie-breaking is stabilized.

**Pass/Fail:** run multiple times; compare `rev_graph[0]` for determinism.

---

## 6. Reverse merge protected-prefix behavior

**Test target:** verify source merge is protected-prefix + reverse insertion, not literal d/2 interleave.  
**Corresponding findings:** BUILD-05.

**Small graph:** 6-node ring after prune:

```text
0 -> [1,2]
1 -> [2,3]
2 -> [3,4]
3 -> [4,5]
4 -> [5,0]
5 -> [0,1]
```

**Parameters:** degree 2 or 4; run reverse+merge.

**Expected behavior:** output retains protected prefix and inserts reverse edge around protected boundary. Compare to a reference literal interleave.

**Pass/Fail:** source output matches protected-prefix reference, not interleave reference.

---

## 7. `guarantee_connectivity` / MST fallback

**Test target:** verify approximate connectivity augmentation and fallback behavior.  
**Corresponding findings:** BUILD-06.

**Small graph:** two disconnected components:

```text
Component A: 0,1,2,3 connected internally
Component B: 4,5,6,7 connected internally
```

**Parameters:** `guarantee_connectivity=true`, small degree cap.

**Expected behavior:** fallback creates cross-component edge(s), possibly to a scanned main-cluster node rather than nearest cross-component node.

**Pass/Fail:** resulting graph becomes connected or more connected; selected cross-edge does not necessarily minimize distance.

---

## 8. CPU vs GPU MST helper edge set

**Test target:** compare optional CPU and GPU MST helper branches.  
**Corresponding findings:** BUILD-06, CPU/GPU path TODO.

**Small graph:** 8 nodes with multiple equally valid cross-cluster candidate edges.

**Parameters:** call internal helper with `use_gpu_for_mst_optimization=true` and `false` if accessible.

**Expected behavior:** both aim to reduce components, but edge sets may differ.

**Pass/Fail:** if edge sets differ but connectivity target holds, report as implementation nondeterminism/branch difference, not bug.

---

## 9. `ef_construction` mapping

**Test target:** show `ef_construction` maps into backend params rather than prune/search width.  
**Corresponding findings:** BUILD-08.

**Setup:** call `from_hnsw_params` or construct HNSW-compatible params for `ef_construction = 64,128,256`.

**Expected behavior:** intermediate degree, NN-Descent iterations, or IVF-PQ `n_probes` changes according to source formulas.

**Pass/Fail:** emitted params match source formulas.

---

## 10. `search_width` SINGLE_CTA vs MULTI_CTA

**Test target:** verify parameter semantic split.  
**Corresponding findings:** SEARCH-01.

**Small graph:** 16 nodes, degree 4, deterministic distances.

**Parameters:** force `algo=SINGLE_CTA` and `algo=MULTI_CTA`; scan `search_width=1,2,4,8`, `itopk_size=64`.

**Expected behavior:**
- SINGLE_CTA candidate slots = `search_width * degree`.
- MULTI_CTA per-CTA candidate slots = `degree`; `search_width` changes CTA count lower bound.

**Pass/Fail:** inspect plan or instrument kernel launch config.

---

## 11. MULTI_CTA auto `max_iterations`

**Test target:** verify auto iterations ignore public `search_width`.  
**Corresponding findings:** SEARCH-02.

**Parameters:** `algo=MULTI_CTA`, `max_iterations=0`, `search_width=1,8,32`, fixed graph degree and dataset size.

**Expected behavior:** planned `max_iterations` remains the same across public search_width values, except for unrelated reachability extras.

**Pass/Fail:** print/inspect plan values.

---

## 12. Filtering-rate adjustment

**Test target:** verify `filtering_rate` only auto-enlarges `itopk_size` in MULTI_CTA.  
**Corresponding findings:** SEARCH-03.

**Small data:** simple vectors with a bitset filter dropping 50% of candidates.

**Parameters:** `filtering_rate=-1` for bitset auto-estimation, and explicit `0.5`; force SINGLE_CTA and MULTI_CTA.

**Expected behavior:** MULTI_CTA adjusted M increases; SINGLE_CTA M stays unchanged.

**Pass/Fail:** inspect plan M and recall@k under both modes.

---

## 13. Bitonic/radix threshold

**Test target:** verify threshold 256 vs 512 across launcher paths.  
**Corresponding findings:** SEARCH-04.

**Parameters:** force SINGLE_CTA with candidate counts:

```text
search_width * graph_degree = 256, 257, 512, 513
```

**Expected behavior:** source audit predicts bitonic for <=256 and radix for >256.

**Pass/Fail:** inspect launch config / compiled template path / profiler symbols.

---

## 14. Convergence semantics

**Test target:** compare paper-style top-M equality with source-style parent availability.  
**Corresponding findings:** SEARCH-05.

**Small graph:** construct a graph where:
- Case A: top-M content still changes but all valid entries are already parent-marked.
- Case B: top-M content is stable but contains unexpanded nodes.

**Parameters:** small `M`, controlled distances, fixed graph.

**Expected behavior:** source stops based on no expandable parent or max_iterations, not explicit equality with previous top-M.

**Pass/Fail:** source behavior differs from a reference top-M equality loop.

---

## 15. MULTI_CTA traversed hash capacity / tombstone behavior

**Test target:** verify whether traversed hash capacity can be exhausted or degraded by tombstones.  
**Corresponding findings:** SEARCH-06.

**Setup:** force small hash bitlen if possible; instrument `insert`, `remove`, and probe counts.

**Small graph:** high overlap graph where many CTAs attempt repeated parents and removals.

**Expected behavior:** if tombstones are not generally reusable, probe length grows; insert failures may occur.

**Pass/Fail:** collect counters; do not call it a recall bug unless insert failures skip valid candidates.

---

## 16. Forgettable hash repeated distance computations

**Test target:** quantify repeated distance computations caused by reset restoring only top-M.  
**Corresponding findings:** SEARCH-07.

**Small graph:** 10 nodes where node X appears in candidate buffer but not internal top-M, then appears again from another parent after reset.

**Parameters:** force small hash and `reset_interval=1` or very small.

**Expected behavior:** X is recomputed after reset.

**Pass/Fail:** distance-computation counter for X > 1.

---

## 17. AUTO algo threshold around SM and 2*SM

**Test target:** verify current AUTO threshold.  
**Corresponding findings:** SEARCH-08.

**Parameters:** choose `itopk_size=512`, vary `max_queries` around `SM` and `2*SM`.

**Expected behavior:** current source selects SINGLE_CTA only at `max_queries >= 2*SM`, not at `SM`.

**Pass/Fail:** inspect plan selected algo.

---

## 18. MULTI_CTA degree/buffer limit

**Test target:** verify `32 + graph_degree` rounded to 32 must be <=256.  
**Corresponding findings:** SEARCH-09.

**Parameters:** force `algo=MULTI_CTA`; use graph degree 224 and 225.

**Expected behavior:** degree 224 is near threshold; degree 225 should fail or select unsupported buffer size depending on rounding.

**Pass/Fail:** error path reports buffer size/degree clearly.

---

## 19. Potential multi-CTA hashmap over-allocation

**Test target:** verify whether `resize()` receives bytes or element count.  
**Corresponding findings:** API-01.

**Setup:** inspect type of `hashmap` storage in `multi_cta_search::set_params`, then record RMM allocation sizes.

**Parameters:** `INDEX_T=uint32`, `max_queries=10`, `entries=2048`.

**Expected behavior if bug exists:** allocated element count is 4x expected for uint32.

**Pass/Fail:** only label confirmed after verifying storage type and RMM allocation.

---

## 20. Cosine metric semantics and zero-norm edge cases

**Test target:** verify cosine output and zero-norm behavior.  
**Corresponding findings:** METRIC-01.

**Data:**

```text
q = [3, 0]
X = [[2,0], [1,0], [0,3], [-1,0], [0,0]]
```

**Expected nonzero behavior:** cosine distance for nonzero vectors should be `[0, 0, 1, 2]` for `[2,0], [1,0], [0,3], [-1,0]`.

**Zero vector:** behavior for `[0,0]` must be measured and documented.

**Pass/Fail:** output matches source formula `1 - cos(q,x)` for nonzero vectors; zero behavior documented.

---

## 21. VPQ cosine path consistency

**Test target:** verify VPQ path handles norms/output consistently with standard path.  
**Corresponding findings:** METRIC-01 TODO.

**Setup:** build standard and VPQ indexes on a tiny dataset where exact cosine ranking is obvious.

**Expected behavior:** both paths rank nonzero vectors consistently with cosine distance semantics, within approximation limits.

**Pass/Fail:** source-level and runtime-level consistency documented.

---

## 22. SONG source availability

**Test target:** determine whether CAGRA-vs-SONG comparison can be source-level.  
**Corresponding findings:** SONG-01.

**Steps:**
1. Search for official SONG repository or author release.
2. If found, fix commit hash and audit analogs: graph storage, priority queue, hash/Bloom/Cuckoo, distance kernels.
3. If not found, label SONG comparison as paper-level only.

**Expected behavior:** final report does not invent SONG source paths.

**Pass/Fail:** source availability status is explicit.
