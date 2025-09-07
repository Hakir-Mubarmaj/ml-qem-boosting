# Placeholder for plots.py

"""Plotting utilities for the ML-QEM experiments.

These are lightweight wrappers around matplotlib to produce consistent figures.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_error_vs_features(df, out_path: str, metric_col: str = 'test_rmse'):
    plt.figure(figsize=(6, 4))
    models = sorted(df['model'].unique())
    for m in models:
        sub = df[df['model'] == m]
        agg = sub.groupby('n_features')[metric_col].agg(['mean', 'std']).reset_index()
        if agg.empty:
            continue
        x = agg['n_features'].values
        y = agg['mean'].values
        y_err = agg['std'].values
        plt.plot(x, y, label=m)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    plt.xlabel('Number of features')
    plt.ylabel(metric_col)
    plt.title('Model error vs number of features')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def bar_best_model_per_bucket(df, out_path: str, metric_col: str = 'test_rmse'):
    buckets = [0, 10, 50, 200, 1000000]
    recs = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        sel = df[(df['n_features'] > lo) & (df['n_features'] <= hi)]
        if sel.empty:
            recs.append({'bucket': f'{lo}-{hi}', 'best': None, 'metric': None})
            continue
        agg = sel.groupby('model')[metric_col].mean().reset_index().sort_values(metric_col)
        best = agg.iloc[0]
        recs.append({'bucket': f'{lo}-{hi}', 'best': best['model'], 'metric': float(best[metric_col])})
    plot_df = pd.DataFrame(recs)
    plt.figure(figsize=(8,4))
    plt.bar(plot_df['bucket'].astype(str), [x if x is not None else 0 for x in plot_df['metric']])
    plt.xlabel('Feature-count bucket')
    plt.ylabel(metric_col)
    plt.title('Best model per feature bucket (metric)')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
