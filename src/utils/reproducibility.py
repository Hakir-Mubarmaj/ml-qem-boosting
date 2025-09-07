# Placeholder for reproducibility.py

"""Reproducibility helpers: seed setting and RNG suppliers."""
import random
import numpy as np

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

def get_rng(seed: int = None):
    import numpy as _np
    return _np.random.RandomState(seed)
