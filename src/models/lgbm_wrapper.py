# Placeholder for lgbm_wrapper.py

"""
LightGBM wrapper for ML-QEM experiments

API: same conventions as xgboost_wrapper
"""
from typing import Any, Dict, Optional
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error as _mse


class LGBMWrapper:
    def __init__(self):
        pass

    @staticmethod
    def _make_model(params: Dict[str, Any]):
        if lgb is not None:
            kwargs = {}
            for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                if k in params:
                    kwargs[k] = params[k]
            return lgb.LGBMRegressor(**kwargs)
        else:
            kwargs = {}
            for k in ['n_estimators', 'max_depth', 'learning_rate']:
                if k in params:
                    kwargs[k] = params[k]
            return GradientBoostingRegressor(**kwargs)

    def train(self, X_train, y_train, X_val, y_val, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        model = self._make_model(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = _mse(y_val, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_val, preds))
        return {'model': model, 'metrics': {'rmse': rmse, 'mae': mae}}

    @staticmethod
    def predict(model_obj, X):
        return model_obj.predict(X)

    @staticmethod
    def evaluate(model_obj, X, y):
        preds = model_obj.predict(X)
        mse = _mse(y, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y, preds))
        return {'rmse': rmse, 'mae': mae}

    @staticmethod
    def suggest_params(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.5),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        }
        params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-6, 10.0)
        params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-6, 10.0)
        return params


if __name__ == '__main__':
    import numpy as _np
    X = _np.random.randn(200, 6)
    y = X[:, 0] * 0.3 + X[:, 2] * -0.1 + _np.random.randn(200) * 0.2
    w = LGBMWrapper()
    print(w.train(X[:150], y[:150], X[150:], y[150:], params={'n_estimators': 100}))
