# Placeholder for io.py

"""Small IO helpers used across the project."""
import os
import json
import pandas as pd

def load_jsonl(path: str):
    out = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

def save_jsonl(list_of_dicts, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for obj in list_of_dicts:
            f.write(json.dumps(obj) + '\n')

def save_parquet(df, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_parquet(out_path)
    except Exception:
        csv_path = out_path + '.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
    return out_path
