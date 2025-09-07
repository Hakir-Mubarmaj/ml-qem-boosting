# Placeholder for metrics.py

"""Evaluation metrics and bootstrap helpers.
Provides regression metrics and a bootstrap CI function.
"""
from typing import Sequence, Callable, Tuple
import numpy as np

def regression_metrics(preds: Sequence[float], targets: Sequence[float]):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError('Shape mismatch')
    mse = float(((preds - targets) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))
    return {'mse': mse, 'rmse': rmse, 'mae': mae}

def bootstrap_ci(preds, targets, metric_fn: Callable[[np.ndarray, np.ndarray], float], n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    import numpy as _np
    vals = []
    n = len(preds)
    idx = _np.arange(n)
    for _ in range(n_bootstrap):
        sample = _np.random.choice(idx, size=n, replace=True)
        vals.append(metric_fn(preds[sample], targets[sample]))
    vals = _np.array(vals)
    lo = _np.quantile(vals, alpha / 2)
    hi = _np.quantile(vals, 1 - alpha / 2)
    return lo, hi
