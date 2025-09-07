# Placeholder for run_baseline.py

"""Run simple baselines (RF, MLP, OLS) on a features parquet file.

Produces a small JSON report with metrics for each baseline.
"""
import argparse
import json
import os
import pandas as pd

from src.models.rf_wrapper import RFWrapper
from src.models.mlp_wrapper import MLPWrapper
from src.models.utils import regression_metrics

def load_parquet(path: str):
    df = pd.read_parquet(path)
    if 'target' not in df.columns:
        raise RuntimeError('Parquet must include a `target` column')
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def run_baselines(feat_parquet: str, out_json: str):
    X, y = load_parquet(feat_parquet)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    results = {}
    rf = RFWrapper()
    rf_res = rf.train(X_train, y_train, X_test, y_test, params={'n_estimators': 100})
    results['rf'] = rf_res.get('metrics')

    try:
        mlp = MLPWrapper()
        mlp_res = mlp.train(X_train, y_train, X_test, y_test, params={})
        results['mlp'] = mlp_res.get('metrics')
    except Exception as e:
        results['mlp'] = {'error': str(e)}

    try:
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        preds = ols.predict(X_test)
        results['ols'] = regression_metrics(preds, y_test)
    except Exception as e:
        results['ols'] = {'error': str(e)}

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote baseline report to', out_json)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True, help='features parquet with target')
    parser.add_argument('--out', dest='out', default='baseline_report.json')
    args = parser.parse_args()
    run_baselines(args.infile, args.out)

if __name__ == '__main__':
    main()
