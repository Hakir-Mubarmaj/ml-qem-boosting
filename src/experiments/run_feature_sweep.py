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


def dataset_to_parquet(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)


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
                # evaluate on test set (attempt to load saved model if possible)
                run_path = meta.get('run_path')
                # try to load model predictions via wrapper predict
                # we'll attempt to load the saved model (model.pkl or joblib) and the wrapper to predict
                test_rmse = None
                try:
                    # wrapper class
                    wrapper_cls = trainer_module.MODEL_REGISTRY.get(model_key)
                    if wrapper_cls is not None:
                        wrapper = wrapper_cls()
                        # load saved model object
                        model_obj = None
                        # look for model file
                        import glob, pickle
                        candidates = glob.glob(os.path.join(run_path, '*.joblib')) + glob.glob(os.path.join(run_path, '*.pkl'))
                        if candidates:
                            with open(candidates[0], 'rb') as f:
                                try:
                                    model_obj = pickle.load(f)
                                except Exception:
                                    import joblib

                                    model_obj = joblib.load(candidates[0])
                        if model_obj is not None:
                            preds = wrapper.predict(model_obj, X_test)
                            from ..models.utils import regression_metrics

                            metrics = regression_metrics(preds, y_test)
                            test_rmse = metrics.get('rmse')
                except Exception as e:
                    logger.warning('Could not evaluate test RMSE for model %s: %s', model_key, str(e))

                rec = {
                    'families': '|'.join(families),
                    'n_families': len(families),
                    'n_features': n_features,
                    'model': model_key,
                    'train_run_path': meta.get('run_path'),
                    'val_score': meta.get('val_score'),
                    'test_rmse': test_rmse,
                    'duration_sec': meta.get('duration_sec'),
                    'timestamp': meta.get('timestamp'),
                }
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
