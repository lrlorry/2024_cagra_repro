#!/usr/bin/env python3
"""Run cuVS CAGRA on SIFT-format data, sweep max_iterations, output CSV.

Output format is identical to engineered_core so bench_plot.py can overlay both curves.

Usage:
  python3 cuvs_sweep.py \
      --base  sift/sift_base.fvecs \
      --query sift/sift_query.fvecs \
      --gt    sift/sift_groundtruth.ivecs \
      --out   cuvs_sweep.csv
"""

import argparse
import csv
import time

import numpy as np

try:
    import cupy as cp
    from cuvs.neighbors import cagra
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with:  conda install -c rapidsai -c conda-forge -c nvidia cuvs\n"
        "           or: pip install cuvs-cu12   (CUDA 12)"
    )


# ── sweep points — must match engineered_main.cu ─────────────────────────────
SWEEP = [4, 8, 12, 16, 20, 24, 32, 48, 64]


# ── fvecs / ivecs readers ─────────────────────────────────────────────────────

def read_fvecs(path, limit=0):
    with open(path, "rb") as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0, 2)
        total = f.tell()
        bpv = 4 + dim * 4
        n = total // bpv
        if limit > 0:
            n = min(n, limit)
        f.seek(0)
        buf = np.empty((n, dim), dtype=np.float32)
        for i in range(n):
            f.read(4)  # skip dim prefix
            buf[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return buf


def read_ivecs(path, limit=0):
    with open(path, "rb") as f:
        k = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0, 2)
        total = f.tell()
        bpv = 4 + k * 4
        n = total // bpv
        if limit > 0:
            n = min(n, limit)
        f.seek(0)
        buf = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            f.read(4)
            buf[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)
    return buf


# ── recall@k ──────────────────────────────────────────────────────────────────

def recall_at_k(result, gt, k):
    nq = result.shape[0]
    gt_k = gt.shape[1]
    hits = 0
    for q in range(nq):
        r = set(result[q, :k].tolist())
        g = set(gt[q, :k].tolist())
        hits += len(r & g)
    return hits / (nq * k)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",    help="fvecs base file")
    parser.add_argument("--query",   help="fvecs query file")
    parser.add_argument("--gt",      help="ivecs ground-truth file")
    parser.add_argument("--nq",      type=int, default=1000)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out",     default="cuvs_sweep.csv")
    # Graph params — cuVS defaults
    parser.add_argument("--intermediate-graph-degree", type=int, default=128,
                        dest="intermediate_graph_degree")
    parser.add_argument("--graph-degree",              type=int, default=64,
                        dest="graph_degree")
    parser.add_argument("--itopk-size",                type=int, default=64,
                        dest="itopk_size")
    parser.add_argument("--search-width",              type=int, default=2,
                        dest="search_width")
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    if args.base and args.query:
        print(f"loading base  : {args.base}")
        base  = read_fvecs(args.base)
        print(f"loading query : {args.query}")
        query = read_fvecs(args.query, limit=args.nq)
        print(f"base  : n={base.shape[0]} dim={base.shape[1]}")
        print(f"query : nq={query.shape[0]} dim={query.shape[1]}")
    else:
        print("[smoke-test] no --base/--query; using random n=8192 dim=128 nq=200")
        rng   = np.random.default_rng(42)
        base  = rng.random((8192, 128), dtype=np.float32)
        query = rng.random((200,  128), dtype=np.float32)

    nq  = query.shape[0]
    k   = args.k

    # ── ground truth ──────────────────────────────────────────────────────────
    if args.gt:
        gt = read_ivecs(args.gt, limit=nq)
        print(f"GT    : nq={gt.shape[0]} k={gt.shape[1]} (loaded from file)")
    else:
        print("computing GT (CPU brute-force) ...")
        gt = np.empty((nq, k), dtype=np.int32)
        for q in range(nq):
            dists = np.sum((base - query[q]) ** 2, axis=1)
            gt[q] = np.argsort(dists)[:k]
        print("GT done.")

    # ── build CAGRA index ─────────────────────────────────────────────────────
    base_gpu  = cp.asarray(base,  dtype=cp.float32)
    query_gpu = cp.asarray(query, dtype=cp.float32)

    build_params = cagra.IndexParams(
        intermediate_graph_degree=args.intermediate_graph_degree,
        graph_degree=args.graph_degree,
        build_algo="nn_descent",
    )
    print(f"building index (n={base.shape[0]} degree={args.graph_degree}) ...")
    t0 = time.perf_counter()
    index = cagra.build(build_params, base_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"build : {build_ms:.1f} ms\n")

    # ── sweep max_iterations ──────────────────────────────────────────────────
    print(f"{'max_iters':<12}  {'recall@k':<10}  {'QPS':<14}  ms/batch")
    print("-" * 52)

    with open(args.out, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["max_iters", "recall", "qps", "search_ms"])

        for max_iters in SWEEP:
            sp = cagra.SearchParams(
                max_iterations=max_iters,
                itopk_size=args.itopk_size,
                search_width=args.search_width,
            )

            # warm-up
            _, _ = cagra.search(sp, index, query_gpu, k)
            cp.cuda.stream.get_current_stream().synchronize()

            # timed runs
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                _, neighbors_gpu = cagra.search(sp, index, query_gpu, k)
                cp.cuda.stream.get_current_stream().synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            times.sort()
            med_ms = times[len(times) // 2]
            qps    = nq / (med_ms / 1000.0)

            neighbors = cp.asarray(neighbors_gpu).get().astype(np.int32)
            recall = recall_at_k(neighbors, gt, k)

            print(f"{max_iters:<12}  {recall:<10.4f}  {qps:<14.1f}  {med_ms:.2f}")
            writer.writerow([max_iters, f"{recall:.6f}", f"{qps:.2f}", f"{med_ms:.3f}"])

    print(f"\nbuild_ms={build_ms:.1f}   results -> {args.out}")


if __name__ == "__main__":
    main()
