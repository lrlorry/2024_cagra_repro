#!/bin/bash
set -e

BASE=${1:-sift/sift_base.fvecs}
QUERY=${2:-sift/sift_query.fvecs}
GT=${3:-sift/sift_groundtruth.ivecs}

make engineered_core

./engineered_core --base "$BASE" --query "$QUERY" --gt "$GT" --out cagra_sweep.csv

python3 cuvs_sweep.py --base "$BASE" --query "$QUERY" --gt "$GT" --out cuvs_sweep.csv

python3 bench_plot.py cagra_sweep.csv cuvs_sweep.csv \
    --labels "CAGRA (ours)" "cuVS official" \
    --out cagra_pareto
