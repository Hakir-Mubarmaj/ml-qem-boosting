# Placeholder for aggregator.py

"""Aggregation helpers for group predictions (weighted average, stacking meta-learner)."""
import numpy as np
from sklearn.linear_model import Ridge

def weighted_aggregate(preds_list, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    stacked = np.vstack(preds_list)
    agg = (weights[:, None] * stacked).sum(axis=0)
    return agg

def stacking_meta_learner(oof_preds, y_oof):
    clf = Ridge(alpha=1.0)
    X = oof_preds.T
    clf.fit(X, y_oof)
    return clf
