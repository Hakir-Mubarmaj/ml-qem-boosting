# Placeholder for feature_encoders.py

"""
Feature encoders for ML-QEM experiments

This module provides a modular FeatureEncoder that converts circuit descriptions
and optional hardware metadata into tabular feature vectors for classical ML.

Design principles:
- Feature families are modular and can be enabled/disabled (so you can sweep #features)
- Feature provenance is preserved via feature names
- Fit/transform API to allow PCA/scaler fitting on training set

Expected input (per-example): a dict with keys (examples below):
{
    'id': str or int,
    'n_qubits': int,
    'gates': [ ('H', [0]), ('CNOT', [0,1]), ... ],
    'two_qubit_edges': [(0,1),(1,2),...],  # optional adjacency list
    'noisy_expectations': np.array([...])  # optional measured values
}

This file is a starting template — extend features as needed. Keep feature names
stable when you run experiments so results are reproducible.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Optional heavy imports; try/except to allow graceful errors in CI
try:
    import networkx as nx
except Exception:
    nx = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None


def safe_len(x):
    return 0 if x is None else len(x)


class FeatureEncoder:
    """Modular encoder that produces a pandas.DataFrame of features.

    Parameters
    ----------
    families: list[str] or None
        Which feature families to include. Available families (defaults):
        - 'global' : global circuit-level features (n_qubits, total_gates, depth estimate)
        - 'per_qubit_agg' : aggregated per-qubit features (mean gate count, max gate count)
        - 'local_density' : sliding-window local two-qubit gate density features
        - 'graph_spectral' : graph Laplacian eigenvalue features (requires networkx)
        - 'noise_stats' : statistics of noisy_expectations
        - 'derived_interactions' : simple pairwise products / PCA components

    pca_components: int or None
        If not None, apply PCA with that many components to the derived interaction block.
    """

    DEFAULT_FAMILIES = [
        'global',
        'per_qubit_agg',
        'local_density',
        'graph_spectral',
        'noise_stats',
        'derived_interactions',
    ]

    def __init__(self, families: Optional[List[str]] = None, pca_components: Optional[int] = None):
        self.families = families if families is not None else self.DEFAULT_FAMILIES.copy()
        self.pca_components = pca_components
        self._fitted = False

        # internal fitted objects
        self.scaler: Optional[Any] = None
        self.pca: Optional[Any] = None
        self.feature_names_: List[str] = []

    def fit(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]] = None):
        """Fit internal transformations (scaler, PCA) using dataset.

        dataset: list of example dicts (see module docstring)
        hardware_meta: optional global hardware calibration info (dict)
        """
        # build raw features first to discover dimensionality
        df = self._build_dataframe(dataset, hardware_meta, apply_pca=False)

        # fit scaler on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if StandardScaler is not None and len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(df[numeric_cols].values)
        else:
            self.scaler = None

        # fit PCA on derived interaction block if requested
        if self.pca_components is not None and 'derived_interactions' in self.families:
            if PCA is None:
                raise RuntimeError('sklearn not available: cannot run PCA')
            # find columns that are derived interactions by name prefix
            derived_cols = [c for c in df.columns if c.startswith('DI_')]
            if len(derived_cols) == 0:
                self.pca = None
            else:
                self.pca = PCA(n_components=min(self.pca_components, len(derived_cols)))
                self.pca.fit(df[derived_cols].values)
        else:
            self.pca = None

        # save feature names in post-transform order
        df2 = self._build_dataframe(dataset, hardware_meta, apply_pca=True)
        self.feature_names_ = df2.columns.tolist()
        self._fitted = True
        return self

    def transform(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Transform dataset into features. Must call fit() first to capture scaler/PCA state."""
        if not self._fitted:
            raise RuntimeError('FeatureEncoder must be fitted before transform()')
        df = self._build_dataframe(dataset, hardware_meta, apply_pca=True)

        # apply scaler if present (only to numeric cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.scaler is not None and len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols].values)

        # reorder to the fitted column ordering (for reproducibility)
        df = df.reindex(columns=self.feature_names_)
        return df

    def fit_transform(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.fit(dataset, hardware_meta)
        return self.transform(dataset, hardware_meta)

    def _build_dataframe(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]], apply_pca: bool) -> pd.DataFrame:
        rows = []
        col_names = None
        for ex in dataset:
            feat_vec = self._encode_example(ex, hardware_meta)
            if col_names is None:
                col_names = list(feat_vec.keys())
            rows.append(feat_vec)
        df = pd.DataFrame(rows, columns=col_names)

        # optionally apply PCA to derived interaction columns
        if apply_pca and self.pca is not None:
            derived_cols = [c for c in df.columns if c.startswith('DI_')]
            if len(derived_cols) > 0:
                pcs = self.pca.transform(df[derived_cols].values)
                # drop original derived columns and add PCA columns in place
                df.drop(columns=derived_cols, inplace=True)
                for i in range(pcs.shape[1]):
                    df[f'PCA_DI_{i}'] = pcs[:, i]
        return df

    def _encode_example(self, ex: Dict[str, Any], hardware_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Encode a single example into a flat dict of features."""
        feats: Dict[str, Any] = {}
        if 'global' in self.families:
            feats.update(self._global_features(ex, hardware_meta))
        if 'per_qubit_agg' in self.families:
            feats.update(self._per_qubit_aggregates(ex))
        if 'local_density' in self.families:
            feats.update(self._local_density_features(ex))
        if 'graph_spectral' in self.families:
            feats.update(self._graph_spectral_features(ex))
        if 'noise_stats' in self.families:
            feats.update(self._noise_stats(ex))
        if 'derived_interactions' in self.families:
            feats.update(self._derived_interactions(ex))
        return feats

    # ----------------------------- Feature families -----------------------------
    def _global_features(self, ex: Dict[str, Any], hardware_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        n_qubits = ex.get('n_qubits', 0)
        total_gates = safe_len(ex.get('gates'))
        two_q = safe_len(ex.get('two_qubit_edges'))
        # naive depth estimate ~ total_gates / n_qubits
        depth_est = float(total_gates) / max(1, n_qubits)
        res = {
            'G_n_qubits': n_qubits,
            'G_total_gates': total_gates,
            'G_two_qubit_count': two_q,
            'G_depth_est': depth_est,
        }
        # hardware meta could add avg_t1 etc.
        if hardware_meta is not None:
            for k, v in (hardware_meta.items() if isinstance(hardware_meta, dict) else []):
                res[f'HW_{k}'] = v
        return res

    def _per_qubit_aggregates(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        # build per-qubit gate counts
        n_qubits = ex.get('n_qubits', 0)
        gate_list = ex.get('gates', [])
        counts = np.zeros(n_qubits, dtype=float)
        for g, qs in gate_list:
            for q in qs:
                if 0 <= q < n_qubits:
                    counts[q] += 1
        if counts.size == 0:
            return {'PQ_mean': 0.0, 'PQ_std': 0.0, 'PQ_max': 0.0}
        return {
            'PQ_mean': float(np.mean(counts)),
            'PQ_std': float(np.std(counts)),
            'PQ_max': float(np.max(counts)),
        }

    def _local_density_features(self, ex: Dict[str, Any], window: int = 3) -> Dict[str, Any]:
        # local two-qubit density along linear ordering of qubits
        n_qubits = ex.get('n_qubits', 0)
        edges = ex.get('two_qubit_edges', [])
        adjacency = np.zeros(n_qubits, dtype=float)
        for u, v in edges:
            if 0 <= u < n_qubits:
                adjacency[u] += 1
            if 0 <= v < n_qubits:
                adjacency[v] += 1
        # sliding window statistics (mean count in windows)
        vals = []
        for i in range(0, max(1, n_qubits), window):
            w = adjacency[i:i + window]
            vals.append(float(np.mean(w)))
        out = {}
        # name them LD_0, LD_1, ... to allow variable-length features for sweeps
        for i, v in enumerate(vals[:20]):  # cap to 20 windows to keep dims bounded
            out[f'LD_{i}'] = v
        # also global mean
        out['LD_mean'] = float(np.mean(adjacency) if adjacency.size > 0 else 0.0)
        return out

    def _graph_spectral_features(self, ex: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        if nx is None:
            # networkx not available; return zeros
            return {f'GS_ev_{i}': 0.0 for i in range(k)}
        edges = ex.get('two_qubit_edges', [])
        n = ex.get('n_qubits', 0)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        try:
            L = nx.normalized_laplacian_matrix(G).astype(float)
            # compute smallest k eigenvalues via numpy (dense) — fine for small n
            w = np.linalg.eigvalsh(L.todense())
            w = np.sort(w)
            vals = list(w[:k]) + [0.0] * max(0, k - len(w))
            return {f'GS_ev_{i}': float(vals[i]) for i in range(k)}
        except Exception:
            return {f'GS_ev_{i}': 0.0 for i in range(k)}

    def _noise_stats(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        arr = np.asarray(ex.get('noisy_expectations', []), dtype=float)
        if arr.size == 0:
            return {'NS_mean': 0.0, 'NS_std': 0.0, 'NS_skew': 0.0}
        # safe low-level stats
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        # simple skew estimate
        skew = float(np.mean(((arr - mean) / (std + 1e-12)) ** 3))
        return {'NS_mean': mean, 'NS_std': std, 'NS_skew': skew}

    def _derived_interactions(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        # create simple derived features: pairwise products of a few base nums
        base = []
        base.append(ex.get('n_qubits', 0))
        base.append(safe_len(ex.get('gates')))
        base.append(safe_len(ex.get('two_qubit_edges')))
        base = [float(x) for x in base]
        out = {}
        idx = 0
        for i in range(len(base)):
            for j in range(i, len(base)):
                out[f'DI_{idx}'] = base[i] * base[j]
                idx += 1
        # optionally add simple PCA later
        return out

    # ----------------------------- utility ------------------------------------
    def get_feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError('Call fit() before get_feature_names()')
        return self.feature_names_


# ----------------------------- quick demo -----------------------------------
if __name__ == '__main__':
    # create a tiny toy dataset
    toy = []
    for q in [4, 6]:
        gates = []
        for i in range(q):
            gates.append(('H', [i]))
        # add some two-qubit gates
        two = [(i, (i + 1) % q) for i in range(q - 1)]
        noisy = np.random.randn(10)
        toy.append({'id': f'c{q}', 'n_qubits': q, 'gates': gates, 'two_qubit_edges': two, 'noisy_expectations': noisy})

    fe = FeatureEncoder(families=None, pca_components=2)
    df = fe.fit_transform(toy)
    print('Feature columns:', df.columns.tolist())
    print(df)
