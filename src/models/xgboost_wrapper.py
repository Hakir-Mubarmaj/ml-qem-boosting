# Placeholder for xgboost_wrapper.py

"""
XGBoost wrapper for ML-QEM experiments

This wrapper exposes a small, consistent API used by the training orchestrator:
- train(X_train, y_train, X_val, y_val, params) -> {'model': model_obj, 'metrics': {...}}
- predict(model_obj, X) -> np.ndarray
- evaluate(model_obj, X_val, y_val) -> dict
- suggest_params(trial) -> dict  (used by optuna tuning if present)

The implementation prefers the `xgboost` package if available, otherwise falls back to
`sklearn.ensemble.GradientBoostingRegressor` as a compatible alternative.

"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class XGBoostWrapper:
    def __init__(self):
        # nothing to store for now
        pass

    @staticmethod
    def _make_model(params: Dict[str, Any]):
        # return an instantiated model object (not yet fitted)
        if xgb is not None:
            # use XGBRegressor
            kwargs = {}
            # map commonly used params if present
            for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                if k in params:
                    kwargs[k] = params[k]
            # default objective/reg:squarederror
            kwargs.setdefault('objective', 'reg:squarederror')
            return xgb.XGBRegressor(**kwargs)
        else:
            # fallback to sklearn's GradientBoostingRegressor
            kwargs = {}
            for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample']:
                if k in params:
                    kwargs[k] = params[k]
            return GradientBoostingRegressor(**kwargs)

    def train(self, X_train, y_train, X_val, y_val, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = self._make_model(params)
        # Fit using numpy arrays (xgboost/sklearn accept pandas as well)
        model.fit(X_train, y_train)
        # evaluate on val
        preds = model.predict(X_val)
        rmse = float(mean_squared_error(y_val, preds, squared=False))
        mae = float(mean_absolute_error(y_val, preds))
        metrics = {'rmse': rmse, 'mae': mae}
        return {'model': model, 'metrics': metrics}

    @staticmethod
    def predict(model_obj, X):
        return model_obj.predict(X)

    @staticmethod
    def evaluate(model_obj, X, y):
        preds = model_obj.predict(X)
        rmse = float(mean_squared_error(y, preds, squared=False))
        mae = float(mean_absolute_error(y, preds))
        return {'rmse': rmse, 'mae': mae}

    @staticmethod
    def suggest_params(trial):
        # Optuna trial -> parameter dict
        # This function is intentionally simple and conservative.
        try:
            import optuna
        except Exception:
            optuna = None
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.5),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        }
        # optional regularization
        params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-6, 10.0)
        params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-6, 10.0)
        return params


# quick smoke test when run directly
if __name__ == '__main__':
    import numpy as _np
    X = _np.random.randn(100, 5)
    y = (X[:, 0] * 0.5 + X[:, 1] * -0.2 + _np.random.randn(100) * 0.1)
    wrapper = XGBoostWrapper()
    res = wrapper.train(X[:80], y[:80], X[80:], y[80:], params={'n_estimators': 50})
    print('val metrics:', res['metrics'])
