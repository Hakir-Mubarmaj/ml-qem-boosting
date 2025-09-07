# ML-QEM Boosting + Feature Group Ensemble

This repository contains code to reproduce and extend ML-QEM experiments with boosted tree models and a feature-group ensemble strategy.

## Repo layout (important files)

* `src/data/feature_encoders.py` — modular feature encoders; enables sweeping feature families.
* `src/models/` — model wrappers for XGBoost, LightGBM, CatBoost, RandomForest and utilities.
* `src/models/train.py` — training orchestrator with optional Optuna tuning (CLI).
* `src/experiments/run_feature_sweep.py` — sweep feature families (increasing #features) and train models.
* `src/experiments/run_group_ensemble.py` — split features into groups, pick best model per group, aggregate.
* `src/analysis/run_analysis.py` — generate Figure 1 (error vs #features) and tables.
* `paper/draft.tex` — LaTeX skeleton for the conference paper.
* `ci/smoke_test.sh` — a small end-to-end smoke test using toy data (quick sanity check).

## Quickstart (smoke test)

Make sure you have Python 3.8+ and required packages installed. For a quick install of common dependencies (recommended):

```bash
python -m pip install -r requirements.txt
```

If you don't have a `requirements.txt`, the minimal packages used are:

* numpy
* pandas
* scikit-learn
* matplotlib

Optional (recommended for full experiments):

* xgboost
* lightgbm
* catboost
* optuna

Run the CI smoke test (quick):

```bash
bash ci/smoke_test.sh
```

This will run a tiny feature sweep with a toy dataset and Random Forest only, then produce a PNG figure and CSV table in `results/ci_smoke`.

## Running a full feature sweep

Example command to run a proper sweep with multiple models:

```bash
python src/experiments/run_feature_sweep.py --out results/feature_sweep --models rf xgb lgbm catboost --pca 2 --trials 30 --tune
```

Be aware: tuning with Optuna and training boosted models can be resource-heavy.

## Group ensemble experiment

After generating feature snapshots (parquet) with `run_feature_sweep.py`, run the group ensemble script on any `features_k*.parquet`:

```bash
python src/experiments/run_group_ensemble.py --in results/feature_sweep/features_k3.parquet --out results/group_ens_k3 --strategy by_family --aggregation weighted
```

## Analysis

Generate the figure/table from sweep results:

```bash
python src/analysis/run_analysis.py --sweep results/feature_sweep/sweep_results_final.csv --out results/figures
```

## Next suggested steps

1. Create a `requirements.txt` (I can generate one for you).
2. Run the smoke test to verify everything runs locally.
3. Replace toy data with the real datasets (simulation + hardware) following the data schema in `src/data/feature_encoders.py`.
4. Run full experiments with Optuna tuning and longer trials.

If you want, I can now:

* generate a `requirements.txt` tuned for your environment; or
* produce the `src/data/make_dataset.py` to read IBMQ job outputs and convert them to the raw JSONL schema; or
* run through a reproducibility checklist and produce the final LaTeX figures placeholders.

Which one should I do next? (If you prefer, I can just produce the `requirements.txt` now.)
