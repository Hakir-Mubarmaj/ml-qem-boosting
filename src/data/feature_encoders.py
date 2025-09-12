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
import os
import pathlib
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
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
        self.scaler = None
        self.numeric_cols = None
        self.pca = None
        self.pca_cols = None
        self.feature_names_: List[str] = []
        

    def _build_df(self, dataset, hardware_meta=None):
        """
        Build a pandas DataFrame from the provided dataset input.

        Accepts:
        - dataset: list of dicts (typical), or a path to a JSONL file, or a pandas.DataFrame.
        - hardware_meta: optional, currently ignored here (kept for API compatibility).

        Returns:
        - DataFrame with deterministic column ordering and numeric NaNs filled with 0.0.
        Notes:
        - If the class already implements a higher-level encoder method like `encode_dataset`
            or `encode_examples`, this method will try to call it first.
        """
        # If the class provides a dedicated encoder, prefer that (keeps existing logic)
        if hasattr(self, "encode_dataset") and callable(getattr(self, "encode_dataset")):
            try:
                df = self.encode_dataset(dataset, hardware_meta=hardware_meta)
                if isinstance(df, pd.DataFrame):
                    # ensure deterministic column order
                    df = df.reindex(sorted(df.columns), axis=1)
                    return df
            except Exception:
                # fall back to generic processing below
                pass

        # If dataset is already a DataFrame, copy and normalize columns
        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            # If dataset is a path to a file, try to read JSONL or JSON
            if isinstance(dataset, (str, pathlib.Path)) and os.path.exists(str(dataset)):
                p = str(dataset)
                # try JSONL first
                try:
                    rows = []
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line))
                    df = json_normalize(rows)
                except Exception:
                    # fallback to generic json read
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            obj = json.load(f)
                        if isinstance(obj, list):
                            df = json_normalize(obj)
                        else:
                            df = json_normalize([obj])
                    except Exception:
                        # final fallback: empty df
                        df = pd.DataFrame()
            else:
                # assume it's an iterable of dict-like examples
                try:
                    df = json_normalize(list(dataset))
                except Exception:
                    # last resort: try constructing DataFrame directly
                    try:
                        df = pd.DataFrame(dataset)
                    except Exception:
                        df = pd.DataFrame()

        # Ensure stable column order (sorted) to avoid scaler/order mismatch
        if not df.empty:
            df = df.reindex(sorted(df.columns), axis=1)

        # Fill numeric NaNs with 0.0 to keep shapes stable for scaler/PCA transforms
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].fillna(0.0)
        except Exception:
            # be defensive: ignore if select_dtypes fails
            pass

        # For non-numeric columns, replace NaN with empty string to avoid unexpected objects
        try:
            obj_cols = df.select_dtypes(include=['object', 'string']).columns
            if len(obj_cols) > 0:
                df[obj_cols] = df[obj_cols].fillna('')
        except Exception:
            pass

        return df
    

    def fit(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]] = None):
        """
        Fit internal transformations (scaler, PCA) using the _encoded_ feature dataframe.
        We call _build_dataframe(..., apply_pca=False) to produce raw features (DI_* etc.),
        fit scaler and PCA (if requested), and record feature_names_ after applying PCA.
        """
        # Build encoded feature DataFrame (without PCA applied yet)
        df_feat = self._build_dataframe(dataset, hardware_meta, apply_pca=False)

        # numeric columns to be used for scaling
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = numeric_cols

        # fit scaler
        if StandardScaler is not None and len(self.numeric_cols) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(df_feat[self.numeric_cols].values)
        else:
            self.scaler = None

        # Fit PCA on derived interaction columns if requested
        if self.pca_components is not None and 'derived_interactions' in self.families:
            derived_cols = [c for c in df_feat.columns if c.startswith('DI_')]
            self.pca_cols = derived_cols
            if PCA is None:
                raise RuntimeError('sklearn PCA not available: cannot run PCA')
            if len(self.pca_cols) > 0:
                self.pca = PCA(n_components=self.pca_components)
                # defensive: fill NaN -> 0.0
                self.pca.fit(df_feat[self.pca_cols].fillna(0.0).values)
            else:
                self.pca = None
                self.pca_cols = []
        else:
            self.pca = None
            self.pca_cols = []

        # Save full post-transform feature order (apply_pca=True constructs final layout)
        df_post = self._build_dataframe(dataset, hardware_meta, apply_pca=True)
        # Ensure deterministic column ordering
        try:
            df_post = df_post.reindex(sorted(df_post.columns), axis=1)
        except Exception:
            pass
        self.feature_names_ = df_post.columns.tolist()
        self._fitted = True
        return self

    def transform(self, dataset, hardware_meta=None):
        """
        Transform dataset using the previously fitted scaler and PCA.
        Uses _build_dataframe to produce encoded features, then applies scaler & PCA
        consistently with what was fitted.
        """
        # Build encoded features (no PCA applied here yet)
        df = self._build_dataframe(dataset, hardware_meta, apply_pca=False)

        # Ensure numeric_cols exist in df (create missing with zeros)
        if self.numeric_cols:
            missing = [c for c in self.numeric_cols if c not in df.columns]
            for c in missing:
                df[c] = 0.0

            if self.scaler is not None:
                # scaler.transform expects the same order of columns used at fit time
                df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols].values)
            else:
                # ensure numeric dtype
                df[self.numeric_cols] = df[self.numeric_cols].astype(float, errors='ignore')

        # Apply PCA if it was fitted during fit()
        if self.pca is not None and self.pca_cols:
            # add missing pca cols with zeros
            missing_pca = [c for c in self.pca_cols if c not in df.columns]
            for c in missing_pca:
                df[c] = 0.0
            # transform
            pca_vals = self.pca.transform(df[self.pca_cols].fillna(0.0).values)
            for i in range(pca_vals.shape[1]):
                df[f'DI_PCA_{i}'] = pca_vals[:, i]
            # drop original DI_ cols
            df.drop(columns=self.pca_cols, inplace=True, errors='ignore')

        # If fit() was called earlier, enforce final feature ordering & fill missing with zeros
        if self._fitted and self.feature_names_:
            for c in self.feature_names_:
                if c not in df.columns:
                    df[c] = 0.0
            # reorder
            df = df[self.feature_names_]

        # Final sanity: drop any remaining non-numeric columns that accidentally slipped in
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            # convert if possible (e.g., JSON-strings of lists can't convert -> drop)
            for c in non_numeric:
                try:
                    df[c] = pd.to_numeric(df[c], errors='raise')
                except Exception:
                    # last resort: drop these columns (they are not useful for numeric training)
                    df.drop(columns=[c], inplace=True)

        return df

    def fit_transform(self, dataset, hardware_meta=None):
        """
        Fit then transform convenience: ensures fit is performed on encoded features
        and returns transformed numeric DataFrame.
        """
        self.fit(dataset, hardware_meta=hardware_meta)
        return self.transform(dataset, hardware_meta=hardware_meta)

    
    def _build_dataframe(self, dataset: List[Dict[str, Any]], hardware_meta: Optional[Dict[str, Any]], apply_pca: bool) -> pd.DataFrame:
        """
        Build DataFrame by encoding each example. Optionally apply PCA to DI_ columns.
        This function ensures PCA fit/transform is handled safely and avoids double-dropping columns.
        """
        rows = []
        col_names = None
        for ex in dataset:
            feat_vec = self._encode_example(ex, hardware_meta)
            if col_names is None:
                col_names = list(feat_vec.keys())
            rows.append(feat_vec)

        # If no columns (empty dataset), return empty DataFrame
        if col_names is None:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=col_names)

        # Ensure deterministic column order (helps reproducibility)
        try:
            df = df.reindex(sorted(df.columns), axis=1)
        except Exception:
            pass

        # Optionally apply PCA to derived interaction columns (DI_*)
        if apply_pca and (self.pca is not None or getattr(self, "pca_components", None)):
            derived_cols = [c for c in df.columns if c.startswith('DI_')]
            if len(derived_cols) > 0:
                # Lazy-init PCA if not yet created
                if self.pca is None and getattr(self, "pca_components", None):
                    from sklearn.decomposition import PCA as _PCA
                    self.pca = _PCA(n_components=self.pca_components)

                # Prepare numeric data for PCA
                col_data = df[derived_cols].fillna(0.0).values

                # Fit PCA if it hasn't been fitted yet
                if self.pca is not None and not hasattr(self.pca, "components_"):
                    try:
                        self.pca.fit(col_data)
                    except Exception as e:
                        raise RuntimeError(f"PCA fit failed on derived columns ({len(derived_cols)} cols): {e}")

                # If PCA is ready, transform and insert new PCA columns
                if self.pca is not None and hasattr(self.pca, "components_"):
                    pcs = self.pca.transform(col_data)
                    for i in range(pcs.shape[1]):
                        df[f'DI_PCA_{i}'] = pcs[:, i]
                    # Drop original derived columns once (errors='ignore' to avoid KeyError)
                    df.drop(columns=derived_cols, inplace=True, errors='ignore')

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
