# Placeholder for run_group_ensemble.py

"""
Run feature-group ensemble experiments

This script implements the "split features into groups -> pick best model per group -> aggregate"
workflow described in the project plan.

High-level flow:
  1. Load a dataset parquet file (features + 'target').
  2. Produce feature groupings using one of several strategies:
     - by_family: use feature name prefixes (e.g. 'G_', 'PQ_', 'LD_', 'GS_', 'NS_', 'DI_')
     - by_block: split features into evenly-sized contiguous blocks
     - by_kmeans: cluster features by their correlation (requires sklearn)
  3. For each group, train a small set of candidate models (rf, xgb, lgbm, catboost)
     and select the best on a validation fold.
  4. Aggregate group predictions via weighted averaging (weights = 1/val_rmse) or
     via a stacking meta-learner (Ridge trained on out-of-fold group predictions).
  5. Evaluate the aggregated prediction against a held-out test set and save results.

This is an experiment utility — it's intentionally conservative and uses simple defaults
so it can run feasibly for CI. For full conference-scale experiments, increase training
budgets, use Optuna tuning, and repeat with multiple random seeds.

Usage example:
  python run_group_ensemble.py --in features/features_k3.parquet --out results/group_ens_k3 --strategy by_family

"""

from typing import List, Dict, Any, Tuple
import os
import argparse
import json
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# local wrappers
from ..models.xgboost_wrapper import XGBoostWrapper
from ..models.lgbm_wrapper import LGBMWrapper
from ..models.catboost_wrapper import CatBoostWrapper
from ..models.rf_wrapper import RFWrapper
from ..models.utils import regression_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_group_ensemble')

MODEL_CANDIDATES = {
    'rf': RFWrapper,
    'xgb': XGBoostWrapper,
    'lgbm': LGBMWrapper,
    'catboost': CatBoostWrapper,
}


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'target' not in df.columns:
        raise ValueError('parquet must contain a `target` column')
    return df


# ---------------- grouping strategies ----------------------------------
def group_by_family(feature_names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name in feature_names:
        # get prefix up to first underscore
        if '_' in name:
            prefix = name.split('_')[0]
        else:
            prefix = 'misc'
        groups.setdefault(prefix, []).append(name)
    return groups


def group_by_block(feature_names: List[str], n_blocks: int) -> Dict[str, List[str]]:
    groups = {}
    n = len(feature_names)
    block_size = max(1, n // n_blocks)
    for i in range(n_blocks):
        start = i * block_size
        end = None if i == n_blocks - 1 else (i + 1) * block_size
        sel = feature_names[start:end]
        groups[f'block_{i}'] = sel
    return groups


def group_by_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, List[str]]:
    # cluster features by their correlation patterns across examples
    X = df.values  # shape: (n_examples, n_features)
    # transpose to (n_features, n_examples)
    Xf = X.T
    # normalize features
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-12)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Xf)
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(f'k_{lab}', []).append(df.columns[i])
    return groups


# ---------------- training helpers ------------------------------------

def train_best_model_for_group(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, candidate_keys: List[str]):
    """Train candidate models on the provided data and return best model object and metrics."""
    best_score = float('inf')
    best_key = None
    best_model = None
    best_metrics = None

    for k in candidate_keys:
        wrapper_cls = MODEL_CANDIDATES.get(k)
        if wrapper_cls is None:
            continue
        try:
            wrapper = wrapper_cls()
            res = wrapper.train(X_train, y_train, X_val, y_val, params={})
            model_obj = res.get('model') if isinstance(res, dict) else res
            metrics = res.get('metrics', {}) if isinstance(res, dict) else regression_metrics(wrapper.predict(model_obj, X_val), y_val)
            rmse = metrics.get('rmse', float('inf'))
            logger.info('Group model %s val_rmse=%.5f', k, rmse)
            if rmse < best_score:
                best_score = rmse
                best_key = k
                best_model = model_obj
                best_metrics = metrics
        except Exception as e:
            logger.warning('Training candidate %s failed: %s', k, str(e))
            continue

    return best_key, best_model, best_metrics


def weighted_aggregate(preds_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    stacked = np.vstack(preds_list)  # shape: (n_models, n_examples)
    agg = (weights[:, None] * stacked).sum(axis=0)
    return agg


def stacking_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray) -> Ridge:
    # oof_preds shape: (n_groups, n_examples)
    clf = Ridge(alpha=1.0)
    # transpose to (n_examples, n_groups)
    X = oof_preds.T
    clf.fit(X, y_oof)
    return clf


# ---------------- main experiment -------------------------------------

def run_group_ensemble(feature_parquet: str, out_dir: str, strategy: str = 'by_family', n_blocks: int = 4, n_clusters: int = 4, candidates: List[str] = None, aggregation: str = 'weighted') -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = load_parquet(feature_parquet)
    feature_names = [c for c in df.columns if c != 'target']

    # choose grouping
    if strategy == 'by_family':
        groups = group_by_family(feature_names)
    elif strategy == 'by_block':
        groups = group_by_block(feature_names, n_blocks=n_blocks)
    elif strategy == 'by_kmeans':
        groups = group_by_kmeans(df[feature_names], n_clusters=n_clusters)
    else:
        raise ValueError('Unknown strategy')

    # split data (train/val/test)
    X = df[feature_names]
    y = df['target']
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    # default candidates
    if candidates is None:
        candidates = list(MODEL_CANDIDATES.keys())

    group_models = {}
    group_val_rmses = {}
    group_preds_test = {}
    group_oof_preds = []

    # For stacking, generate simple out-of-fold predictions for meta-learner (here we use val as OOF for simplicity)
    for gname, cols in groups.items():
        logger.info('Processing group %s with %d features', gname, len(cols))
        Xg_train = X_train[cols]
        Xg_val = X_val[cols]
        Xg_test = X_test[cols]
        best_key, best_model, best_metrics = train_best_model_for_group(Xg_train, y_train, Xg_val, y_val, candidates)
        if best_key is None:
            logger.warning('No successful model for group %s; skipping', gname)
            continue
        group_models[gname] = {'model_key': best_key, 'metrics': best_metrics}
        group_val_rmses[gname] = float(best_metrics.get('rmse', float('inf')))
        # predict on test set
        wrapper_cls = MODEL_CANDIDATES[best_key]
        wrapper = wrapper_cls()
        preds_test = wrapper.predict(best_model, Xg_test)
        preds_val = wrapper.predict(best_model, Xg_val)
        group_preds_test[gname] = preds_test
        group_oof_preds.append(preds_val)

    # aggregate
    agg_preds = None
    if aggregation == 'weighted':
        # weights inversely proportional to val_rmse
        weights = []
        preds = []
        for gname, p in group_preds_test.items():
            preds.append(p)
            rmse = group_val_rmses.get(gname, 1.0)
            weights.append(1.0 / (rmse + 1e-12))
        agg_preds = weighted_aggregate(preds, weights)
    elif aggregation == 'stacking':
        # train stacking meta-learner on val predictions
        if len(group_oof_preds) == 0:
            raise RuntimeError('No group predictions available for stacking')
        oof = np.vstack(group_oof_preds)
        meta = stacking_meta_learner(oof, y_val.values)
        # build test matrix
        test_mat = np.vstack([group_preds_test[g] for g in group_preds_test.keys()]).T
        agg_preds = meta.predict(test_mat)
    else:
        raise ValueError('Unknown aggregation')

    # evaluate aggregated prediction
    rmse = float(mean_squared_error(y_test.values, agg_preds, squared=False))
    mae = float(np.mean(np.abs(y_test.values - agg_preds)))

    result = {
        'strategy': strategy,
        'n_groups': len(groups),
        'groups': {k: len(v) for k, v in groups.items()},
        'aggregation': aggregation,
        'rmse_test': rmse,
        'mae_test': mae,
        'group_val_rmses': group_val_rmses,
    }

    # save details
    with open(os.path.join(out_dir, 'group_ensemble_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True, help='input parquet file (features + target)')
    parser.add_argument('--out', dest='out', default='results/group_ensemble', help='output folder')
    parser.add_argument('--strategy', choices=['by_family', 'by_block', 'by_kmeans'], default='by_family')
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--aggregation', choices=['weighted', 'stacking'], default='weighted')
    args = parser.parse_args()

    res = run_group_ensemble(args.infile, args.out, strategy=args.strategy, n_blocks=args.n_blocks, n_clusters=args.n_clusters, aggregation=args.aggregation)
    print('Result:', res)


if __name__ == '__main__':
    main()
