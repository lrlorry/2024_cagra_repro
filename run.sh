#!/bin/bash
# run.sh — 一键运行所有实验并生成全部科研图
# 用法:
#   ./run.sh [base.fvecs] [query.fvecs] [gt.ivecs]
# 无参数时使用随机 smoke-test 数据（n=8192 dim=128 nq=200）

set -e

BASE=${1:-""}
QUERY=${2:-""}
GT=${3:-""}

OUT_DIR="results"
mkdir -p "$OUT_DIR"

# ── 构建可执行文件 ──────────────────────────────────────────────────────────
make engineered_core

# ── 数据参数拼接 ────────────────────────────────────────────────────────────
DATA_ARGS=""
if [ -n "$BASE" ] && [ -n "$QUERY" ]; then
  DATA_ARGS="--base $BASE --query $QUERY"
  [ -n "$GT" ] && DATA_ARGS="$DATA_ARGS --gt $GT"
fi

# ── 实验 1: Pareto 曲线（sweep max_iters）──────────────────────────────────
echo "=== [1/6] Pareto sweep ==="
./engineered_core $DATA_ARGS --mode pareto --out "$OUT_DIR/cagra_sweep.csv"

# cuVS baseline（若脚本存在）
if [ -f cuvs_sweep.py ]; then
  echo "    running cuVS baseline..."
  python3 cuvs_sweep.py $DATA_ARGS --out "$OUT_DIR/cuvs_sweep.csv" || true
fi

# ── 实验 2: Build 阶段计时分解 ────────────────────────────────────────────
echo "=== [2/6] Build breakdown ==="
./engineered_core $DATA_ARGS --mode build_breakdown --out "$OUT_DIR/build_breakdown.csv"

# ── 实验 3: NN-Descent 迭代数消融 ────────────────────────────────────────
echo "=== [3/6] NN-Descent ablation ==="
./engineered_core $DATA_ARGS --mode nn_descent --out "$OUT_DIR/nn_descent.csv"

# ── 实验 4: Search Width 参数敏感性 ──────────────────────────────────────
echo "=== [4/6] Search width sweep ==="
./engineered_core $DATA_ARGS --mode search_width --out "$OUT_DIR/search_width.csv"

# ── 实验 5: SingleCTA vs MultiCTA 切换点 ─────────────────────────────────
echo "=== [5/6] CTA regime sweep ==="
./engineered_core $DATA_ARGS --mode cta_regime --out "$OUT_DIR/cta_regime.csv"

# ── 实验 6: 可扩展性（dataset size sweep）────────────────────────────────
echo "=== [6/6] Scalability sweep ==="
./engineered_core $DATA_ARGS --mode scalability --out "$OUT_DIR/scalability.csv"

# ── 生成全部图 ──────────────────────────────────────────────────────────────
echo ""
echo "=== Generating figures ==="
PARETO_FILES="$OUT_DIR/cagra_sweep.csv"
PARETO_LABELS="CAGRA-repro"
if [ -f "$OUT_DIR/cuvs_sweep.csv" ]; then
  PARETO_FILES="$PARETO_FILES $OUT_DIR/cuvs_sweep.csv"
  PARETO_LABELS="$PARETO_LABELS cuVS-official"
fi

python3 bench_plot.py pareto $PARETO_FILES \
    --labels $PARETO_LABELS --out figures/fig1_pareto

python3 bench_plot.py build       "$OUT_DIR/build_breakdown.csv" --out figures/fig2_build_breakdown
python3 bench_plot.py nn_descent  "$OUT_DIR/nn_descent.csv"      --out figures/fig3_nn_descent
python3 bench_plot.py search_width "$OUT_DIR/search_width.csv"   --out figures/fig4_search_width
python3 bench_plot.py cta_regime  "$OUT_DIR/cta_regime.csv"      --out figures/fig5_cta_regime
python3 bench_plot.py scalability "$OUT_DIR/scalability.csv"     --out figures/fig6_scalability

echo ""
echo "Done. Figures saved to figures/:"
ls figures/*.pdf 2>/dev/null || true
