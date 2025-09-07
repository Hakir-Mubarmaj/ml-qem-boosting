# Placeholder for catboost_wrapper.py

"""
CatBoost wrapper for ML-QEM experiments

API: same conventions as other wrappers
"""
from typing import Any, Dict, Optional
import numpy as np

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class CatBoostWrapper:
    def __init__(self):
        pass

    @staticmethod
    def _make_model(params: Dict[str, Any]):
        if CatBoostRegressor is not None:
            kwargs = {'verbose': 0}
            for k in ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'random_strength']:
                if k in params:
                    kwargs[k] = params[k]
            return CatBoostRegressor(**kwargs)
        else:
            # fall back
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
        rmse = float(mean_squared_error(y_val, preds, squared=False))
        mae = float(mean_absolute_error(y_val, preds))
        return {'model': model, 'metrics': {'rmse': rmse, 'mae': mae}}

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
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 2, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.5),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-6, 10.0),
        }
        return params


if __name__ == '__main__':
    import numpy as _np
    X = _np.random.randn(200, 6)
    y = X[:, 0] * 0.3 + X[:, 2] * -0.1 + _np.random.randn(200) * 0.2
    w = CatBoostWrapper()
    print(w.train(X[:150], y[:150], X[150:], y[150:], params={'iterations': 100}))
