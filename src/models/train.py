# Placeholder for train.py

"""
Training orchestrator for ML-QEM boosting experiments

Features:
- Supports training XGBoost, LightGBM, CatBoost, RandomForest wrappers
- Optional Optuna hyperparameter tuning (falls back to simple random search if Optuna
  is unavailable)
- Saves model artifacts and run metadata to experiments/runs/
- CLI entrypoint for single-run or tuning-run

Expectations: wrapper modules exist in src/models/ (xgboost_wrapper.py, lgbm_wrapper.py,
catboost_wrapper.py, rf_wrapper.py) and expose a `train(X_train, y_train, X_val, y_val, params)`
function that returns a fitted model and a dict of metrics.

"""

import os
import json
import uuid
import argparse
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

# Optuna optional
try:
    import optuna
except Exception:
    optuna = None

# model wrappers (assumed to be present in repo)
from ..models.xgboost_wrapper import XGBoostWrapper
from ..models.lgbm_wrapper import LGBMWrapper
from ..models.catboost_wrapper import CatBoostWrapper
from ..models.rf_wrapper import RFWrapper
from ..models.utils import regression_metrics


MODEL_REGISTRY = {
    'xgb': XGBoostWrapper,
    'lgbm': LGBMWrapper,
    'catboost': CatBoostWrapper,
    'rf': RFWrapper,
}


def _make_outdirs(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    runs_dir = os.path.join(base_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    return runs_dir


def save_run(run_metadata: Dict[str, Any], model_obj: Any, runs_dir: str) -> str:
    run_id = run_metadata.get('run_id', str(uuid.uuid4()))
    run_path = os.path.join(runs_dir, run_id)
    os.makedirs(run_path, exist_ok=True)
    # save metadata
    with open(os.path.join(run_path, 'metadata.json'), 'w') as f:
        json.dump(run_metadata, f, indent=2)
    # save model if joblib available
    if joblib is not None:
        try:
            joblib.dump(model_obj, os.path.join(run_path, 'model.joblib'))
        except Exception:
            # best-effort save
            with open(os.path.join(run_path, 'model.pkl'), 'wb') as f:
                import pickle

                pickle.dump(model_obj, f)
    else:
        # fallback pickle
        with open(os.path.join(run_path, 'model.pkl'), 'wb') as f:
            import pickle

            pickle.dump(model_obj, f)

    return run_path


def simple_random_search(model_cls, X_train, y_train, X_val, y_val, param_space: Dict[str, list], n_iters: int = 20):
    """Fallback tuner: samples randomly from discrete param_space and returns best model."""
    best = None
    best_score = float('inf')
    best_params = None
    for i in range(n_iters):
        params = {k: np.random.choice(v) for k, v in param_space.items()}
        model = model_cls()
        m_obj = model.train(X_train, y_train, X_val, y_val, params)
        metrics = m_obj.get('metrics') if isinstance(m_obj, dict) else {}
        val_rmse = metrics.get('rmse', float('inf'))
        if val_rmse < best_score:
            best_score = val_rmse
            best = m_obj if not isinstance(m_obj, dict) else m_obj.get('model', None)
            best_params = params
    return best, best_params, best_score


def optuna_tune(model_key: str, model_cls, X_train, y_train, X_val, y_val, n_trials: int = 50):
    """Run Optuna tuning for a given wrapper class. The wrapper class is expected to
    implement a `suggest_params(trial)` staticmethod that returns a dict of params
    for the trial."""
    if optuna is None:
        raise RuntimeError('Optuna not available')

    def objective(trial):
        params = model_cls.suggest_params(trial)
        wrapper = model_cls()
        res = wrapper.train(X_train, y_train, X_val, y_val, params)
        metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
        return float(metrics.get('rmse', float('inf')))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    best_params = best_trial.params

    # train final model with best params
    wrapper = model_cls()
    res = wrapper.train(X_train, y_train, X_val, y_val, best_params)
    model_obj = res.get('model') if isinstance(res, dict) else res
    return model_obj, best_params, float(best_trial.value)


def train_single(model_key: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any]):
    model_cls = MODEL_REGISTRY.get(model_key)
    if model_cls is None:
        raise ValueError(f'Unknown model {model_key}. Known: {list(MODEL_REGISTRY.keys())}')
    wrapper = model_cls()
    res = wrapper.train(X_train, y_train, X_val, y_val, params)
    # wrapper.train is expected to return dict: {'model': model_obj, 'metrics': {...}}
    if isinstance(res, dict):
        return res.get('model'), res.get('metrics', {})
    else:
        # legacy: return just the fitted model
        metrics = regression_metrics(wrapper.predict(X_val), y_val)
        return res, metrics


def train_and_tune(model_key: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, out_base: str, tune: bool = True, n_trials: int = 50, param_space: Optional[Dict[str, list]] = None):
    runs_dir = _make_outdirs(out_base)
    start = time.time()
    metadata = {
        'run_id': str(uuid.uuid4()),
        'model_key': model_key,
        'tune': tune,
        'n_trials': n_trials,
        'timestamp': time.time(),
    }

    model_cls = MODEL_REGISTRY.get(model_key)
    if model_cls is None:
        raise ValueError('Unknown model')

    if tune:
        # prefer Optuna if available and wrapper has suggest_params
        if optuna is not None and hasattr(model_cls, 'suggest_params'):
            model_obj, best_params, best_score = optuna_tune(model_key, model_cls, X_train, y_train, X_val, y_val, n_trials=n_trials)
        else:
            if param_space is None:
                raise ValueError('param_space required for non-Optuna tuning')
            model_obj, best_params, best_score = simple_random_search(model_cls, X_train, y_train, X_val, y_val, param_space, n_iters=n_trials)
        # evaluate on validation to collect metrics
        wrapper = model_cls()
        try:
            # if wrapper exposes evaluate
            metrics = wrapper.evaluate(model_obj, X_val, y_val)
        except Exception:
            preds = wrapper.predict(model_obj, X_val) if hasattr(wrapper, 'predict') else None
            metrics = regression_metrics(preds, y_val) if preds is not None else {}
        metadata.update({'best_params': best_params, 'val_score': best_score, 'metrics': metrics})
    else:
        # train with default params (wrapper-defined)
        model_obj, metrics = train_single(model_key, X_train, y_train, X_val, y_val, params={})
        best_params = {}
        metadata.update({'metrics': metrics})

    metadata['duration_sec'] = time.time() - start
    run_path = save_run(metadata, model_obj, runs_dir)
    metadata['run_path'] = run_path
    return metadata


def load_xy(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    # expects parquet with features and a column 'target'
    df = pd.read_parquet(path)
    if 'target' not in df.columns:
        raise ValueError('Expected `target` column in parquet file')
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(MODEL_REGISTRY.keys()), default='xgb')
    parser.add_argument('--train', type=str, required=True, help='parquet file path for train set')
    parser.add_argument('--val', type=str, required=True, help='parquet file path for val set')
    parser.add_argument('--out', type=str, default='experiments')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--param-space', type=str, default=None, help='JSON file describing param grid for fallback tuner')
    args = parser.parse_args()

    X_train, y_train = load_xy(args.train)
    X_val, y_val = load_xy(args.val)

    param_space = None
    if args.param_space is not None:
        with open(args.param_space, 'r') as f:
            param_space = json.load(f)

    metadata = train_and_tune(args.model, X_train, y_train, X_val, y_val, out_base=args.out, tune=args.tune, n_trials=args.trials, param_space=param_space)
    print('Training finished. Metadata saved to:', metadata.get('run_path'))


if __name__ == '__main__':
    main_cli()
