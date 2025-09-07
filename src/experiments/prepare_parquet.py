"""
Prepare parquet feature snapshots from a raw JSONL dataset by encoding with FeatureEncoder
for incremental feature family sets. This lets you precompute feature matrices and reuse them
for different experiments (sweep, group-ensemble).

Usage:
  python prepare_parquet.py --in data/raw/normalized.jsonl --out data/features --pca 2

This script will write files like:
  data/features/features_k1.parquet, features_k2.parquet, ...
and a manifest `data/features/manifest.json` describing which family set corresponds to each file.
"""

import os
import json
import argparse
from typing import List, Dict, Any

import pandas as pd

from ..data.make_dataset import normalize_jsonl, generate_toy_jsonl
from ..data.feature_encoders import FeatureEncoder

# NOTE: we avoid heavy imports at global scope; just use FeatureEncoder

DEFAULT_FAMILIES = [
    'global',
    'per_qubit_agg',
    'local_density',
    'graph_spectral',
    'noise_stats',
    'derived_interactions',
]


def load_jsonl_to_list(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def write_parquet(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)


def prepare_parquets(raw_jsonl: str, out_dir: str, pca_components: int = 2, families_list: List[List[str]] = None):
    if families_list is None:
        families_list = [DEFAULT_FAMILIES[:k] for k in range(1, len(DEFAULT_FAMILIES) + 1)]

    examples = load_jsonl_to_list(raw_jsonl)
    manifest = []
    for i, families in enumerate(families_list, start=1):
        fe = FeatureEncoder(families=families, pca_components=pca_components)
        print(f'Encoding features for families: {families}')
        df = fe.fit_transform(examples)
        # attach target if present in raw examples
        targets = [ex.get('target') for ex in examples]
        if any(t is not None for t in targets):
            df['target'] = [float(t) if t is not None else float('nan') for t in targets]
        out_path = os.path.join(out_dir, f'features_k{i}.parquet')
        write_parquet(df, out_path)
        manifest.append({'file': out_path, 'families': families, 'n_features': df.shape[1]})
    # write manifest
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Wrote', len(manifest), 'feature parquet files to', out_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=False, help='input raw JSONL (if omitted, generate toy data)')
    parser.add_argument('--out', dest='outdir', default='data/features', help='output folder for parquets')
    parser.add_argument('--pca', dest='pca', type=int, default=2, help='pca components for derived features (0 to disable)')
    args = parser.parse_args()

    if args.infile is None:
        print('No input provided; generating toy JSONL into memory')
        raw_file = 'data/raw/toy_generated.jsonl'
        os.makedirs(os.path.dirname(raw_file), exist_ok=True)
        generate_toy_jsonl(raw_file, n_samples=300)
        infile = raw_file
    else:
        infile = args.infile

    pca = args.pca if args.pca > 0 else None
    prepare_parquets(infile, args.outdir, pca_components=pca)


if __name__ == '__main__':
    main()
