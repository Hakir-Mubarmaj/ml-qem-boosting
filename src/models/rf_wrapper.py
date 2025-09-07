# Placeholder for rf_wrapper.py

"""
RandomForest wrapper for ML-QEM experiments

Simple wrapper around sklearn's RandomForestRegressor
"""
from typing import Any, Dict, Optional
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RFWrapper:
    def __init__(self):
        pass

    def _make_model(self, params: Dict[str, Any]):
        kwargs = {}
        if params is None:
            params = {}
        for k in ['n_estimators', 'max_depth', 'max_features']:
            if k in params:
                kwargs[k] = params[k]
        kwargs.setdefault('n_estimators', 100)
        return RandomForestRegressor(**kwargs)

    def train(self, X_train, y_train, X_val, y_val, params: Optional[Dict[str, Any]] = None):
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        }
        return params


if __name__ == '__main__':
    import numpy as _np
    X = _np.random.randn(200, 10)
    y = X[:, 0] * 0.4 + _np.random.randn(200) * 0.2
    w = RFWrapper()
    print(w.train(X[:160], y[:160], X[160:], y[160:], params={'n_estimators': 100}))
