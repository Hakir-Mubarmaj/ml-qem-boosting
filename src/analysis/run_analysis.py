"""
Analysis utilities to generate publication-ready figures and tables from experiment outputs.

Generates:
  - Figure 1: Error vs #features (line plot with CI bands if available)
  - Table: Best model per feature-count bucket

Usage:
  python run_analysis.py --sweep results/feature_sweep/sweep_results_final.csv --out results/figures

Notes:
  - This script is intentionally lightweight and depends only on matplotlib/pandas/numpy.
  - It expects the sweep CSV to contain columns: 'n_features', 'model', 'test_rmse' (or 'val_score')
"""

import os
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_analysis')


def plot_error_vs_features(df: pd.DataFrame, out_path: str, metric_col: str = 'test_rmse'):
    """Plot metric vs number of features for each model.

    df: must contain columns ['n_features', 'model', metric_col]
    """
    plt.figure(figsize=(6, 4))
    models = sorted(df['model'].unique())
    for m in models:
        sub = df[df['model'] == m]
        # aggregate by n_features (mean + std)
        agg = sub.groupby('n_features')[metric_col].agg(['mean', 'std', 'count']).reset_index()
        if agg.empty:
            continue
        x = agg['n_features'].values
        y = agg['mean'].values
        y_err = agg['std'].values
        plt.plot(x, y, label=m)
        # error band using std (if count>1)
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
    logger.info('Saved figure to %s', out_path)


def best_model_per_bucket(df: pd.DataFrame, buckets: Optional[list] = None, metric_col: str = 'test_rmse') -> pd.DataFrame:
    """Compute best model per feature-count bucket.

    buckets: list of integers defining bucket edges, e.g. [0, 10, 50, 200, 10000]
    Returns DataFrame with columns: bucket_low, bucket_high, best_model, best_metric, n_features_median
    """
    if buckets is None:
        buckets = [0, 10, 50, 200, 1000000]
    labels = []
    recs = []
    for i in range(len(buckets) - 1):
        lo = buckets[i]
        hi = buckets[i + 1]
        sel = df[(df['n_features'] > lo) & (df['n_features'] <= hi)]
        if sel.empty:
            recs.append({'bucket_low': lo, 'bucket_high': hi, 'best_model': None, 'best_metric': None, 'n_features_median': None})
            continue
        # choose the model with smallest metric mean across rows in this bucket
        agg = sel.groupby('model')[metric_col].mean().reset_index()
        agg = agg.sort_values(metric_col)
        best = agg.iloc[0]
        recs.append({'bucket_low': lo, 'bucket_high': hi, 'best_model': best['model'], 'best_metric': float(best[metric_col]), 'n_features_median': int(sel['n_features'].median())})
    return pd.DataFrame(recs)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', required=True, help='path to sweep_results_final.csv')
    parser.add_argument('--out', default='results/figures', help='output folder for figures and tables')
    parser.add_argument('--metric', default='test_rmse', help='metric column to plot (test_rmse or val_score)')
    parser.add_argument('--buckets', type=str, default=None, help='comma-separated bucket edges, e.g. 0,10,50,200,1000000')
    args = parser.parse_args()

    df = pd.read_csv(args.sweep)
    os.makedirs(args.out, exist_ok=True)

    fig_path = os.path.join(args.out, 'figure_error_vs_features.png')
    plot_error_vs_features(df, fig_path, metric_col=args.metric)

    if args.buckets is not None:
        buckets = [int(x) for x in args.buckets.split(',')]
    else:
        buckets = None
    table = best_model_per_bucket(df, buckets=buckets, metric_col=args.metric)
    table_path = os.path.join(args.out, 'table_best_model_per_bucket.csv')
    table.to_csv(table_path, index=False)
    logger.info('Saved table to %s', table_path)

    print('Analysis complete. Figures and tables saved to', args.out)


if __name__ == '__main__':
    main_cli()
