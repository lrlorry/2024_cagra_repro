# CAGRA Core CUDA Reproduction

这个目录把 CAGRA 的核心过程拆成小文件，方便按源码结构讲解。

它不是 cuVS 的完整替代实现，而是一个 CUDA/C++ 版的“核心流程复现”：只保留建图和搜索，并分别给出无工程优化版和带工程优化版。

## 目录结构

```text
.
├── Makefile
├── common/
│   ├── cagra_common.cuh       # hash、L2 距离、top-M 插入等共用 device 工具
│   └── cuda_utils.cuh         # CUDA_CHECK
├── plain/
│   ├── config.cuh             # 无优化版参数
│   ├── plain_build.cuh        # build_graph_plain 声明
│   ├── plain_build.cu         # 初始化图、NN-Descent relax、rank prune、慢速 reverse merge
│   ├── plain_search.cuh       # search_plain 声明
│   ├── plain_search.cu        # 每 query 一个 thread 的朴素搜索
│   └── plain_main.cu          # demo main
└── engineered/
    ├── config.cuh             # 工程版参数
    ├── engineered_plan.cuh    # SearchAlgo / SearchPlan / auto plan
    ├── engineered_plan.cu
    ├── engineered_build.cuh   # build_graph_engineered 声明
    ├── engineered_build.cu    # atomic reverse graph、protected merge、connectivity repair
    ├── engineered_search.cuh  # search_engineered 声明
    ├── engineered_search.cu   # single-CTA / multi-CTA / shared hash / MSB flag
    └── engineered_main.cu     # demo main
```

## 两个版本的区别

### `plain/`：无工程优化版

建图：

1. `init_random_graph_kernel`
2. `nn_descent_relax_kernel`
3. `rank_prune_kernel`
4. `slow_reverse_merge_kernel`

搜索：

1. 一个 CUDA thread 处理一个 query。
2. 维护 top-M list。
3. 每轮扩展最好的未扩展 parent。
4. 返回 top-k。

这个版本的目的：看懂算法骨架。

### `engineered/`：带工程优化版

建图：

1. `init_random_graph_kernel`
2. `nn_descent_relax_kernel`
3. `rank_prune_kernel`
4. `make_reverse_graph_atomic_kernel`
5. `combine_graph_kernel`

搜索：

1. `choose_algo` 自动选 single-CTA / multi-CTA。
2. CTA 内线程协作计算 L2。
3. `float4` 做 128-bit vectorized load。
4. shared-memory hash 避免重复访问。
5. MSB parent flag 标记候选点是否已经展开。
6. multi-CTA 搜索后再 merge partial top-k。

这个版本的目的：看懂 cuVS 源码为什么会拆 search plan、single/multi CTA、hashmap、distance helper。

## 编译

```bash
make
```

如果机器没有 NVIDIA GPU 或没有 `nvcc`，就只能读代码，不能本地运行。

## 运行

```bash
./plain_core
./engineered_core
```

## 对应 cuVS 源码

- `cpp/src/neighbors/detail/cagra/cagra_build.cuh`
  - 选择初始图构建方法。
  - 本复现只写简化 NN-Descent-like 初始图。

- `cpp/src/neighbors/detail/cagra/graph_core.cuh`
  - rank-based prune。
  - reverse graph。
  - merge pruned and reverse edges。
  - optional `guarantee_connectivity`。

- `cpp/src/neighbors/detail/cagra/search_plan.cuh`
  - 自动选择 search 策略。
  - 调整 internal top-k、iterations、hash 参数。

- `cpp/src/neighbors/detail/cagra/search_single_cta.cuh`
  - batch 足够大时高吞吐 single-CTA。

- `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh`
  - 小 batch 或更高 recall 时多个 CTA 处理一个 query。

- `cpp/src/neighbors/detail/cagra/compute_distance.hpp`
  - team / warp splitting 计算距离。

- `cpp/src/neighbors/detail/cagra/hashmap.hpp`
  - visited / traversed hash table。

