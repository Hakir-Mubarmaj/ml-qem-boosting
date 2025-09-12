# Placeholder for run_feature_sweep.py

"""
Run feature-sweep experiments: progressively increase feature families, train models,
and save summary results for plotting (error vs #features).

Usage examples:
  python run_feature_sweep.py --out results/sweep_default --models rf xgb lgbm catboost
  python run_feature_sweep.py --raw data/raw/dataset.jsonl --out results/sweep1 --models rf

Notes:
- Input can be a JSONL of raw examples (one dict per line, matching FeatureEncoder input),
  or omitted in which case a tiny toy dataset is used.
- This script will attempt to call src.models.train.train_and_tune for tuning when available.
  If model wrappers are missing, it will skip those models and continue.

"""

import os
import argparse
import json
import time
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# local imports (package-style)
from ..data.feature_encoders import FeatureEncoder
from ..models import train as trainer_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_feature_sweep')


DEFAULT_FAMILIES = [
    'global',
    'per_qubit_agg',
    'local_density',
    'graph_spectral',
    'noise_stats',
    'derived_interactions',
]

def _find_complex_columns(df, n_sample=50):
    complex_cols = []
    for col in df.columns:
        try:
            nonnull = df[col].dropna().head(n_sample).tolist()
        except Exception:
            complex_cols.append(col)
            continue
        if not nonnull:
            continue
        for v in nonnull:
            if isinstance(v, (list, dict, tuple, np.ndarray)):
                complex_cols.append(col)
                break
    return complex_cols

def _serialize_complex_columns(df, cols):
    for col in cols:
        def _convert(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            if isinstance(x, np.ndarray):
                x = x.tolist()
            try:
                return json.dumps(x)
            except TypeError:
                return json.dumps(str(x))
        df[col] = df[col].apply(_convert)
    return df

def dataset_to_parquet(df: pd.DataFrame, out_path: str, **to_parquet_kwargs):
    """
    Safely write DataFrame to parquet: serialize list/dict/ndarray columns to JSON strings first.
    """
    df = df.copy()
    # deterministic order
    try:
        df = df.reindex(sorted(df.columns), axis=1)
    except Exception:
        pass

    complex_cols = _find_complex_columns(df)
    if complex_cols:
        print(f"[run_feature_sweep] serializing complex columns before parquet write: {complex_cols}")
        df = _serialize_complex_columns(df, complex_cols)

    to_parquet_kwargs.setdefault("index", False)
    df.to_parquet(out_path, **to_parquet_kwargs)
    print(f"[run_feature_sweep] Wrote parquet: {out_path} (rows={len(df)})")


def load_raw_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def make_toy_dataset(n_samples: int = 200) -> List[Dict[str, Any]]:
    # Creates a synthetic dataset similar to the demo in feature_encoders
    rng = np.random.RandomState(0)
    data = []
    for i in range(n_samples):
        q = rng.choice([4, 6, 8])
        gates = []
        for _ in range(rng.randint(q * 2, q * 6)):
            op = rng.choice(['H', 'X', 'Y', 'Z', 'CNOT'])
            if op == 'CNOT' and q >= 2:
                a = rng.randint(0, q)
                b = (a + rng.randint(1, q)) % q
                gates.append(('CNOT', [int(a), int(b)]))
            else:
                gates.append((op, [int(rng.randint(0, q))]))
        two = []
        for _ in range(rng.randint(max(1, q - 1), q * 2)):
            a = rng.randint(0, q - 1)
            two.append([int(a), int((a + 1) % q)])
        noisy = rng.randn(20).tolist()
        # toy target: a noisy function of n_qubits and gate counts
        target = float(q * 0.1 + len(gates) * 0.01 + rng.randn() * 0.05)
        data.append({'id': f'toy_{i}', 'n_qubits': q, 'gates': gates, 'two_qubit_edges': two, 'noisy_expectations': noisy, 'target': target})
    return data


def run_sweep(raw_dataset: List[Dict[str, Any]], out_dir: str, models: List[str], pca_components: int = 2, tune: bool = False, trials: int = 30):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # Prepare simple train/val/test splits indices after encoding per-sweep
    for k in range(1, len(DEFAULT_FAMILIES) + 1):
        families = DEFAULT_FAMILIES[:k]
        logger.info('Building features for families: %s', families)
        encoder = FeatureEncoder(families=families, pca_components=pca_components)
        # prepare dataset examples; expect 'target' key on each example
        examples = [ex for ex in raw_dataset if 'target' in ex]
        if len(examples) == 0:
            raise RuntimeError('No examples with `target` found in the raw dataset')

        df = encoder.fit_transform(examples)
        # attach target column from raw_dataset
        targets = [ex['target'] for ex in examples]
        df['target'] = targets

        # save a small parquet snapshot for reproducibility
        snapshot_path = os.path.join(out_dir, f'features_k{k}.parquet')
        dataset_to_parquet(df, snapshot_path)

        X = df.drop(columns=['target'])
        y = df['target']

        # basic split
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

        n_features = X.shape[1]

        for model_key in models:
            try:
                logger.info('Training model %s on %d features (families=%s)', model_key, n_features, families)
                out_base = os.path.join(out_dir, 'models')
                # call the trainer API; it will save runs in experiments/runs under out_base
                meta = trainer_module.train_and_tune(model_key=model_key, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, out_base=out_base, tune=tune, n_trials=trials, param_space=None)
                # --- start patch: compute/verify val_score and test_rmse if missing ---
                run_path = meta.get('run_path')
                test_rmse = None
                val_score = meta.get('val_score')

                if run_path:
                    try:
                        # get wrapper class if exists
                        wrapper_cls = getattr(trainer_module, "MODEL_REGISTRY", {}).get(model_key)
                        wrapper = wrapper_cls() if wrapper_cls is not None else None

                        # find candidate model files
                        import glob, joblib, pickle
                        candidates = glob.glob(os.path.join(run_path, '*.joblib')) \
                                + glob.glob(os.path.join(run_path, '*.pkl')) \
                                + glob.glob(os.path.join(run_path, 'model.*')) \
                                + glob.glob(os.path.join(run_path, '*.sav'))
                        model_obj = None
                        if candidates:
                            p = candidates[0]
                            try:
                                model_obj = joblib.load(p)
                            except Exception:
                                try:
                                    with open(p, 'rb') as f:
                                        model_obj = pickle.load(f)
                                except Exception as e:
                                    logger.debug("failed to load model file %s: %s", p, e)

                        # helper to get predictions robustly
                        def safe_predict(mobj, X):
                            # try wrapper.predict first
                            if wrapper is not None:
                                try:
                                    return wrapper.predict(mobj, X)
                                except Exception:
                                    logger.debug("wrapper.predict failed for %s", model_key)
                            # fallback to model_obj.predict
                            try:
                                return mobj.predict(X)
                            except Exception as e:
                                logger.debug("model_obj.predict failed: %s", e)
                                raise

                        from ..models.utils import regression_metrics

                        if model_obj is not None:
                            # compute val_score if missing
                            if val_score is None:
                                try:
                                    preds_val = safe_predict(model_obj, X_val)
                                    val_metrics = regression_metrics(preds_val, y_val)
                                    val_score = val_metrics.get('rmse')
                                    logger.info("Computed val_score for %s at %s -> %s", model_key, run_path, val_score)
                                except Exception as e:
                                    logger.warning("Could not compute val_score for %s: %s", model_key, e)

                            # compute test_rmse if missing
                            try:
                                preds_test = safe_predict(model_obj, X_test)
                                test_metrics = regression_metrics(preds_test, y_test)
                                test_rmse = test_metrics.get('rmse')
                                logger.info("Computed test_rmse for %s at %s -> %s", model_key, run_path, test_rmse)
                            except Exception as e:
                                logger.warning("Could not compute test_rmse for %s: %s", model_key, e)

                    except Exception as e:
                        logger.warning('Could not evaluate model files at %s for model %s: %s', run_path, model_key, e)

                # update rec with possibly computed val_score/test_rmse
                rec = {
                    'families': '|'.join(families),
                    'n_families': len(families),
                    'n_features': n_features,
                    'model': model_key,
                    'train_run_path': meta.get('run_path'),
                    'val_score': val_score,
                    'test_rmse': test_rmse,
                    'duration_sec': meta.get('duration_sec'),
                    'timestamp': meta.get('timestamp'),
                }
                # --- end patch ---

                results.append(rec)
                # persist intermediate CSV
                pd.DataFrame(results).to_csv(os.path.join(out_dir, 'sweep_results.csv'), index=False)
            except Exception as e:
                logger.exception('Training failed for model %s on feature families %s: %s', model_key, families, str(e))
                # continue to next model
                continue

    # final save
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, 'sweep_results_final.csv'), index=False)
    logger.info('Sweep complete. Results saved to %s', out_dir)
    return res_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, default=None, help='path to raw dataset JSONL (one json per line)')
    parser.add_argument('--out', type=str, default='results/feature_sweep', help='output folder')
    parser.add_argument('--models', nargs='+', default=['rf', 'xgb', 'lgbm', 'catboost'], help='models to run')
    parser.add_argument('--pca', type=int, default=2, help='pca components for derived features')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()

    if args.raw is not None:
        raw = load_raw_jsonl(args.raw)
    else:
        logger.info('No raw dataset provided; generating toy dataset')
        raw = make_toy_dataset(n_samples=250)

    run_sweep(raw, args.out, args.models, pca_components=args.pca, tune=args.tune, trials=args.trials)


if __name__ == '__main__':
    main()
