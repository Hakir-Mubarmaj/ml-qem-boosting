# Placeholder for split_methods.py

"""Feature-splitting helper functions.

Provides grouping strategies used by the group-ensemble experiments.
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def group_by_family(feature_names: List[str]) -> Dict[str, List[str]]:
    groups = {}
    for name in feature_names:
        if '_' in name:
            prefix = name.split('_')[0]
        else:
            prefix = 'misc'
        groups.setdefault(prefix, []).append(name)
    return groups

def group_by_block(feature_names: List[str], n_blocks: int) -> Dict[str, List[str]]:
    groups = {}
    n = len(feature_names)
    block_size = max(1, n // n_blocks)
    for i in range(n_blocks):
        start = i * block_size
        end = None if i == n_blocks - 1 else (i + 1) * block_size
        sel = feature_names[start:end]
        groups[f'block_{i}'] = sel
    return groups

def group_by_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, List[str]]:
    X = df.values
    Xf = X.T
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-12)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Xf)
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(f'k_{lab}', []).append(df.columns[i])
    return groups
