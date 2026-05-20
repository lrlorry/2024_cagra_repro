#!/usr/bin/env python3
"""
bench_plot.py — 科研级 CAGRA 实验图生成脚本

用法:
  python bench_plot.py pareto        cagra_sweep.csv [cuvs_sweep.csv ...] [--labels L1 L2] [--out fig1]
  python bench_plot.py build         build_breakdown.csv                                    [--out fig2]
  python bench_plot.py nn_descent    nn_descent.csv                                         [--out fig3]
  python bench_plot.py search_width  search_width.csv                                       [--out fig4]
  python bench_plot.py cta_regime    cta_regime.csv                                         [--out fig5]
  python bench_plot.py scalability   scalability.csv                                        [--out fig6]
  python bench_plot.py all           results/          (读取目录下所有 CSV，生成全部图)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 论文级全局样式 ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         150,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

# colorblind-friendly palette (Wong 2011)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS  = ["o", "s", "^", "D", "v", "P", "*", "X"]


def save(fig, stem):
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path)
        print(f"  saved {path}")
    plt.close(fig)


# ── CSV loaders ────────────────────────────────────────────────────────────────

def load_csv(path):
    """Load CSV, skipping # comment lines. Returns (rows, meta) where
    meta is a dict parsed from '# key=val ...' lines."""
    meta = {}
    lines = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                for token in line[1:].split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try: meta[k] = float(v)
                        except: meta[k] = v
            else:
                lines.append(line)
    rows = list(csv.DictReader(lines))
    return rows, meta


def col_f(rows, key):
    return [float(r[key]) for r in rows]


def col_i(rows, key):
    return [int(r[key]) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Recall@k vs QPS Pareto 曲线
# ══════════════════════════════════════════════════════════════════════════════
def plot_pareto(csvfiles, labels, out_stem):
    """
    每个 CSV 一条 Pareto 线（recall vs QPS），Y 轴对数刻度。
    CSV 格式: max_iters, recall, qps, search_ms
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for idx, (path, label) in enumerate(zip(csvfiles, labels)):
        rows, meta = load_csv(path)
        recall = col_f(rows, "recall")
        qps    = col_f(rows, "qps")
        iters  = col_i(rows, "max_iters")
        c, m   = PALETTE[idx % len(PALETTE)], MARKERS[idx % len(MARKERS)]

        ax.plot(recall, qps, marker=m, color=c, label=label, zorder=3)
        for x, y, it in zip(recall, qps, iters):
            ax.annotate(str(it), (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=c)

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / second)")
    ax.set_title("CAGRA — Recall@10 vs Throughput")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="lower right")
    ax.set_xlim(left=max(0, min(col_f(load_csv(csvfiles[0])[0], "recall")) - 0.05))

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Build 阶段耗时分解（水平堆叠柱状图）
# ══════════════════════════════════════════════════════════════════════════════
def plot_build_breakdown(csvfile, out_stem):
    """
    CSV 格式: stage, impl, time_ms
    stages: init, nn_descent, prune, reverse, connectivity
    """
    rows, _ = load_csv(csvfile)
    impls = list(dict.fromkeys(r["impl"] for r in rows))  # 保序去重

    stage_order  = ["init", "nn_descent", "prune", "reverse", "connectivity"]
    stage_labels = {
        "init":           "Init graph",
        "nn_descent":     "NN-Descent",
        "prune":          "Rank prune",
        "reverse":        "Reverse merge",
        "connectivity":   "Connectivity",
    }
    stage_colors = {
        "init":           "#56B4E9",
        "nn_descent":     "#0072B2",
        "prune":          "#009E73",
        "reverse":        "#E69F00",
        "connectivity":   "#CC79A7",
    }

    # 按 impl × stage 建 dict
    data = {impl: {s: 0.0 for s in stage_order} for impl in impls}
    for r in rows:
        s = r["stage"]
        if s in data[r["impl"]]:
            data[r["impl"]][s] = float(r["time_ms"])

    fig, ax = plt.subplots(figsize=(7, 2.8 + 0.5 * len(impls)))
    y_pos = np.arange(len(impls))

    lefts = np.zeros(len(impls))
    handles = []
    for stage in stage_order:
        vals = np.array([data[impl][stage] for impl in impls])
        if vals.sum() == 0:
            continue
        bars = ax.barh(y_pos, vals, left=lefts,
                       color=stage_colors[stage],
                       label=stage_labels[stage], height=0.55)
        # 在每段里标注时间（仅当宽度足够时）
        for bar, v in zip(bars, vals):
            if v > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        handles.append(bars[0])
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(impls)
    ax.set_xlabel("Build time (ms)")
    ax.set_title("Graph Build — Stage Timing Breakdown")
    ax.legend(handles=handles, labels=[stage_labels[s] for s in stage_order
                                        if data[impls[0]][s] > 0 or any(data[im][s] > 0 for im in impls)],
              loc="lower right", ncol=2)
    ax.invert_yaxis()

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — NN-Descent 迭代数消融
# ══════════════════════════════════════════════════════════════════════════════
def plot_nn_descent(csvfile, out_stem):
    """
    CSV 格式: nn_iters, recall, build_ms
    """
    rows, _ = load_csv(csvfile)
    iters   = col_i(rows, "nn_iters")
    recall  = col_f(rows, "recall")
    build   = col_f(rows, "build_ms")

    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()

    l1, = ax1.plot(iters, recall, marker="o", color=PALETTE[0], label="Recall@10")
    l2, = ax2.plot(iters, build,  marker="s", color=PALETTE[1],
                   linestyle="--", label="Build time (ms)")

    ax1.set_xlabel("NN-Descent iterations")
    ax1.set_ylabel("Recall@10", color=PALETTE[0])
    ax2.set_ylabel("Build time (ms)", color=PALETTE[1])
    ax1.tick_params(axis="y", colors=PALETTE[0])
    ax2.tick_params(axis="y", colors=PALETTE[1])
    ax1.set_title("Effect of NN-Descent Iterations on Graph Quality")
    ax1.set_xticks(iters)
    ax1.set_ylim(0, 1.05)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Search Width 参数敏感性
# ══════════════════════════════════════════════════════════════════════════════
def plot_search_width(csvfile, out_stem):
    """
    CSV 格式: search_width, max_iters, recall, qps
    每个 search_width 值画一条 Pareto 线。
    """
    rows, _ = load_csv(csvfile)
    sws  = sorted(set(int(r["search_width"]) for r in rows))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for idx, sw in enumerate(sws):
        sub = [r for r in rows if int(r["search_width"]) == sw]
        sub.sort(key=lambda r: float(r["recall"]))
        recall = col_f(sub, "recall")
        qps    = col_f(sub, "qps")
        ax.plot(recall, qps,
                marker=MARKERS[idx % len(MARKERS)],
                color=PALETTE[idx % len(PALETTE)],
                label=f"search_width={sw}")

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / second)")
    ax.set_title("Search Width Sensitivity — Recall@10 vs Throughput")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Search width", loc="lower right")

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — SingleCTA vs MultiCTA 切换点
# ══════════════════════════════════════════════════════════════════════════════
def plot_cta_regime(csvfile, out_stem):
    """
    CSV 格式: num_queries, algo, qps
    """
    rows, _ = load_csv(csvfile)
    algos = list(dict.fromkeys(r["algo"] for r in rows))

    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, algo in enumerate(algos):
        sub = [r for r in rows if r["algo"] == algo]
        sub.sort(key=lambda r: int(r["num_queries"]))
        nq  = col_i(sub, "num_queries")
        qps = col_f(sub, "qps")
        ax.plot(nq, qps, marker=MARKERS[idx], color=PALETTE[idx], label=algo)

    # 标注切换阈值（两线交叉附近）
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries (batch size)")
    ax.set_ylabel("QPS (queries / second)")
    ax.set_title("Single-CTA vs Multi-CTA: Throughput Crossover")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left")

    # 灰色竖线标注近似切换点（两线 QPS 最接近处）
    all_nq = sorted(set(int(r["num_queries"]) for r in rows))
    if len(algos) == 2:
        qps_map = {algo: {int(r["num_queries"]): float(r["qps"])
                           for r in rows if r["algo"] == algo}
                   for algo in algos}
        crossover = None
        for nq in all_nq:
            vals = [qps_map[a].get(nq) for a in algos]
            if all(v is not None for v in vals):
                if abs(vals[0] - vals[1]) / max(vals) < 0.15:
                    crossover = nq
                    break
        if crossover:
            ax.axvline(crossover, color="gray", linewidth=1, linestyle=":",
                       label=f"crossover ≈ {crossover}")
            ax.legend(loc="upper left")

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — 可扩展性（Build time + QPS vs Dataset size）
# ══════════════════════════════════════════════════════════════════════════════
def plot_scalability(csvfile, out_stem):
    """
    CSV 格式: n, build_ms, qps, recall  (recall=-1 表示未计算)
    """
    rows, _ = load_csv(csvfile)
    rows.sort(key=lambda r: int(r["n"]))
    ns       = col_i(rows, "n")
    build    = col_f(rows, "build_ms")
    qps      = col_f(rows, "qps")
    recall   = col_f(rows, "recall")
    has_recall = any(r > 0 for r in recall)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(ns, build, marker="o", color=PALETTE[0], label="Build time")
    l2, = ax2.plot(ns, qps,   marker="s", color=PALETTE[1],
                   linestyle="--", label="QPS")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel("Dataset size n")
    ax1.set_ylabel("Build time (ms)", color=PALETTE[0])
    ax2.set_ylabel("QPS (queries / second)", color=PALETTE[1])
    ax1.tick_params(axis="y", colors=PALETTE[0])
    ax2.tick_params(axis="y", colors=PALETTE[1])
    ax1.set_title("Scalability — Build Time and Throughput vs Dataset Size")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x < 1e6 else f"{x/1e6:.0f}M"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    lines = [l1, l2]
    if has_recall:
        # recall 作为散点颜色编码标注在 ax1
        valid = [(n, r) for n, r in zip(ns, recall) if r >= 0]
        if valid:
            vns, vr = zip(*valid)
            sc = ax1.scatter(vns, [build[ns.index(n)] for n in vns],
                             c=vr, cmap="RdYlGn", vmin=0.8, vmax=1.0,
                             s=60, zorder=5, edgecolors="k", linewidths=0.5)
            plt.colorbar(sc, ax=ax2, label="Recall@10", pad=0.12)

    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")

    save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════
# all 模式 — 自动从目录读取所有 CSV 并生成对应图
# ══════════════════════════════════════════════════════════════════════════════
def run_all(data_dir, out_dir):
    d = Path(data_dir)
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)

    # Pareto 图：收集所有 *sweep*.csv
    pareto_files = sorted(d.glob("*sweep*.csv")) + sorted(d.glob("*pareto*.csv"))
    if pareto_files:
        labels = [f.stem for f in pareto_files]
        print(f"[pareto] {[str(f) for f in pareto_files]}")
        plot_pareto([str(f) for f in pareto_files], labels,
                    str(o / "fig1_pareto"))

    for name, fn in [
        ("build_breakdown", plot_build_breakdown),
        ("nn_descent",      plot_nn_descent),
        ("search_width",    plot_search_width),
        ("cta_regime",      plot_cta_regime),
        ("scalability",     plot_scalability),
    ]:
        candidates = list(d.glob(f"*{name}*.csv"))
        if candidates:
            f = candidates[0]
            fig_map = {
                "build_breakdown": "fig2_build_breakdown",
                "nn_descent":      "fig3_nn_descent",
                "search_width":    "fig4_search_width",
                "cta_regime":      "fig5_cta_regime",
                "scalability":     "fig6_scalability",
            }
            print(f"[{name}] {f}")
            fn(str(f), str(o / fig_map[name]))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode",
                   choices=["pareto","build","nn_descent","search_width",
                             "cta_regime","scalability","all"],
                   help="实验类型")
    p.add_argument("inputs", nargs="*",
                   help="CSV 文件（pareto 支持多个）或数据目录（all 模式）")
    p.add_argument("--labels", nargs="*",
                   help="pareto 模式下各曲线的图例标签")
    p.add_argument("--out", default=None,
                   help="输出文件名 stem（不含扩展名）；all 模式下为输出目录")

    args = p.parse_args()

    default_stems = {
        "pareto":       "fig1_pareto",
        "build":        "fig2_build_breakdown",
        "nn_descent":   "fig3_nn_descent",
        "search_width": "fig4_search_width",
        "cta_regime":   "fig5_cta_regime",
        "scalability":  "fig6_scalability",
    }

    if args.mode == "all":
        data_dir = args.inputs[0] if args.inputs else "."
        out_dir  = args.out or "figures"
        print(f"Generating all figures from '{data_dir}' → '{out_dir}/'")
        run_all(data_dir, out_dir)
        return

    if not args.inputs:
        p.error(f"{args.mode} mode requires at least one CSV file")

    stem = args.out or default_stems[args.mode]

    if args.mode == "pareto":
        labels = args.labels or [Path(f).stem for f in args.inputs]
        if len(labels) != len(args.inputs):
            p.error("--labels count must match number of CSV files")
        plot_pareto(args.inputs, labels, stem)

    elif args.mode == "build":
        plot_build_breakdown(args.inputs[0], stem)

    elif args.mode == "nn_descent":
        plot_nn_descent(args.inputs[0], stem)

    elif args.mode == "search_width":
        plot_search_width(args.inputs[0], stem)

    elif args.mode == "cta_regime":
        plot_cta_regime(args.inputs[0], stem)

    elif args.mode == "scalability":
        plot_scalability(args.inputs[0], stem)


if __name__ == "__main__":
    main()
