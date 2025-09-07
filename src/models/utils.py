# Placeholder for utils.py

"""
Model utilities: metrics, CI helpers, small wrappers
"""
from typing import Sequence, Dict, Any
import numpy as np


def regression_metrics(preds: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError('Shape mismatch')
    n = preds.shape[0]
    mse = float(((preds - targets) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


def bootstrap_ci(preds, targets, metric_fn, n_bootstrap: int = 1000, alpha: float = 0.05):
    # metric_fn(preds, targets) -> float
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
