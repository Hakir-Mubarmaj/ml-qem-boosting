#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test to verify the minimal pipeline runs end-to-end on toy data.
# It will run a tiny feature sweep with only Random Forest and produce analysis figures.

OUT_DIR=results/ci_smoke
mkdir -p "$OUT_DIR"

echo "Running feature sweep (toy data, RF only, tiny runs)..."
python src/experiments/run_feature_sweep.py --out "$OUT_DIR/feature_sweep" --models rf --pca 0 --trials 1

echo "Running analysis to produce figure/table..."
python src/analysis/run_analysis.py --sweep "$OUT_DIR/feature_sweep/sweep_results_final.csv" --out "$OUT_DIR/figures" --metric test_rmse

echo "Smoke test complete. Output in $OUT_DIR"
