# compute_missing_rmse_v2.py
import os, glob, json
import joblib, pickle
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

def safe_load(path):
    """Try joblib then pickle load, return (obj, loader_name) or (None, None)."""
    try:
        obj = joblib.load(path)
        return obj, 'joblib'
    except Exception:
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return obj, 'pickle'
        except Exception:
            return None, None

def summarize_obj(o, depth=0, maxlen=200):
    t = type(o)
    if isinstance(o, dict):
        return f"dict(keys={list(o.keys())})"
    if isinstance(o, list):
        return f"list(len={len(o)})"
    import numpy as _np
    if isinstance(o, _np.ndarray):
        return f"ndarray(shape={o.shape}, dtype={o.dtype})"
    # default: small repr
    r = repr(o)
    if len(r) > maxlen:
        r = r[:maxlen] + '...'
    return f"{t.__name__}: {r}"

def find_predictor(obj, seen=None, depth=0):
    """Recursively search object for something with .predict attribute."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    # direct predictor
    try:
        if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
            return obj
    except Exception:
        pass
    # common wrapper attributes
    for attr in ('model','estimator','best_estimator_','clf','regressor'):
        try:
            if hasattr(obj, attr):
                cand = getattr(obj, attr)
                p = find_predictor(cand, seen, depth+1)
                if p is not None:
                    return p
        except Exception:
            pass
    # if dict, iterate values
    if isinstance(obj, dict):
        for v in obj.values():
            p = find_predictor(v, seen, depth+1)
            if p is not None:
                return p
    # if list/tuple/ndarray, iterate
    try:
        import numpy as _np
        if isinstance(obj, (list, tuple)) or isinstance(obj, _np.ndarray):
            for v in obj:
                p = find_predictor(v, seen, depth+1)
                if p is not None:
                    return p
    except Exception:
        pass
    # check attributes (objects that hold others)
    if hasattr(obj, '__dict__'):
        for k,v in vars(obj).items():
            p = find_predictor(v, seen, depth+1)
            if p is not None:
                return p
    return None

def load_and_inspect_model(run_path):
    # look for model files (prioritized)
    patterns = ['model.joblib','model.pkl','model.*','*.joblib','*.pkl','*.model','*.sav']
    for pat in patterns:
        candidates = glob.glob(os.path.join(run_path, pat))
        for c in sorted(candidates):
            obj, loader = safe_load(c)
            if obj is None:
                continue
            summary = summarize_obj(obj)
            print(f"   Loaded {c} via {loader} -> {summary}")
            pred = find_predictor(obj)
            if pred is not None:
                print("     -> Found predictor object:", type(pred).__name__)
                return pred
            else:
                print("     -> No predictor found inside this object.")
    # fallback: try metadata.json to inspect what was saved
    meta_path = os.path.join(run_path, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path,'r') as f:
                meta = json.load(f)
            print("   metadata.json content:", meta)
        except Exception as e:
            print("   Could not read metadata.json:", e)
    return None

def evaluate_and_update(csv_path='results/smoke_sweep/sweep_results.csv', out_csv=None):
    df = pd.read_csv(csv_path)
    updated = df.copy()
    for idx, row in df.iterrows():
        if pd.notna(row.get('test_rmse')) and str(row.get('test_rmse')).strip() != '':
            print(f"Row {idx} already has test_rmse -> {row.get('test_rmse')}")
            continue
        rp = row.get('train_run_path')
        print(f"Row {idx} model {row['model']} rp= {rp}")
        if not rp or not os.path.exists(rp):
            print("  run_path missing")
            continue
        pred_obj = load_and_inspect_model(rp)
        if pred_obj is None:
            print("  No usable predictor found in", rp)
            continue
        # load snapshot
        k = int(row['n_families'])
        snapshot = os.path.join('results/smoke_sweep', f'features_k{k}.parquet')
        if not os.path.exists(snapshot):
            print("  snapshot not found", snapshot)
            continue
        df_snap = pd.read_parquet(snapshot)
        if 'target' not in df_snap.columns:
            print("  snapshot has no target column")
            continue
        X = df_snap.drop(columns=['target'])
        y = df_snap['target']
        # create test split exactly like run_feature_sweep
        from sklearn.model_selection import train_test_split
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
        try:
            preds = pred_obj.predict(X_test)
            rmse = float(sqrt(mean_squared_error(y_test, preds)))
            print(f"   computed rmse = {rmse}")
            updated.at[idx, 'test_rmse'] = rmse
        except Exception as e:
            print("   predict failed:", e)
    if out_csv is None:
        out_csv = csv_path.replace('.csv','._updated.csv')
    updated.to_csv(out_csv, index=False)
    print('Wrote updated csv ->', out_csv)
    return out_csv

if __name__ == '__main__':
    evaluate_and_update()
