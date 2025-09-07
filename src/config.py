"""Central experiment configuration for ML-QEM project."""

from typing import List

# Feature families (order matters for incremental sweeps)
DEFAULT_FAMILIES: List[str] = [
    'global',
    'per_qubit_agg',
    'local_density',
    'graph_spectral',
    'noise_stats',
    'derived_interactions',
]

# PCA components applied to derived interaction block (None to disable)
PCA_COMPONENTS = 2

# Default models to evaluate
MODELS = ['rf', 'xgb', 'lgbm', 'catboost']

# Training / tuning settings
DEFAULT_TRIALS = 30
TUNE_WITH_OPTUNA = True

# Reproducibility
RANDOM_SEED = 0

# Paths
RAW_DATA_DIR = 'data/raw'
FEATURES_DIR = 'data/features'
EXPERIMENTS_DIR = 'experiments'
RESULTS_DIR = 'results'

# Quick utility to load overrides from a JSON file
def load_from_json(path: str):
    import json
    with open(path, 'r') as f:
        d = json.load(f)
    # copy keys if present
    globals().update({k: v for k, v in d.items() if k in globals()})
    return globals()
