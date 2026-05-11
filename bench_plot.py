#!/usr/bin/env python3
"""Plot Recall@k vs QPS Pareto curve from cagra_sweep.csv."""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "max_iters": int(row["max_iters"]),
                "recall":    float(row["recall"]),
                "qps":       float(row["qps"]),
                "search_ms": float(row["search_ms"]),
            })
    return rows


def plot(csvfiles, labels, output_stem):
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (path, label) in enumerate(zip(csvfiles, labels)):
        rows = load_csv(path)
        recalls = [r["recall"] for r in rows]
        qps     = [r["qps"]    for r in rows]
        iters   = [r["max_iters"] for r in rows]

        ax.plot(recalls, qps,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                label=label,
                linewidth=1.5,
                markersize=6)

        for x, y, it in zip(recalls, qps, iters):
            ax.annotate(f"{it}", (x, y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colors[idx % len(colors)])

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("QPS (queries / second)", fontsize=12)
    ax.set_title("CAGRA — Recall@10 vs QPS", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(left=0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = f"{output_stem}.{ext}"
        fig.savefig(out, dpi=150)
        print(f"saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvfiles", nargs="+",
                        help="one or more cagra_sweep.csv files")
    parser.add_argument("--labels", nargs="*",
                        help="legend labels (default: csv filename stems)")
    parser.add_argument("--out", default="cagra_pareto",
                        help="output filename stem (default: cagra_pareto)")
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0]
                              for p in args.csvfiles]
    if len(labels) != len(args.csvfiles):
        print("error: --labels count must match number of csv files", file=sys.stderr)
        sys.exit(1)

    plot(args.csvfiles, labels, args.out)


if __name__ == "__main__":
    main()
