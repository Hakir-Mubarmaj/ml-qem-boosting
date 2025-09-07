# Placeholder for mlp_wrapper.py

"""MLP wrapper using scikit-learn's MLPRegressor (simple baseline).

Provides the same wrapper API as other model wrappers in the repo.
"""
from typing import Any, Dict, Optional
import numpy as np
try:
    from sklearn.neural_network import MLPRegressor
except Exception:
    MLPRegressor = None
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MLPWrapper:
    def __init__(self):
        pass

    @staticmethod
    def _make_model(params: Dict[str, Any]):
        kwargs = {}
        if MLPRegressor is None:
            raise RuntimeError('scikit-learn MLPRegressor not available')
        for k in ['hidden_layer_sizes', 'activation', 'alpha', 'learning_rate_init', 'max_iter']:
            if k in params:
                kwargs[k] = params[k]
        kwargs.setdefault('random_state', 0)
        return MLPRegressor(**kwargs)

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
            'hidden_layer_sizes': tuple([trial.suggest_categorical('hl1', [32, 64, 128])]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e-1),
            'learning_rate_init': trial.suggest_loguniform('lr', 1e-4, 1e-1),
            'max_iter': 200,
        }
        return params
