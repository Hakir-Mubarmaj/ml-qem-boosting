# Placeholder for make_dataset.py

"""
Dataset creation utilities for ML-QEM project

This script helps convert raw experimental outputs (e.g. Qiskit Job/Result JSON dumps,
measurement counts, or simple user JSONL files) into the canonical raw-example schema
expected by the FeatureEncoder:

Per-example dict fields expected by FeatureEncoder:
{
  'id': str,
  'n_qubits': int,
  'gates': list of (op, [qubits]) tuples,
  'two_qubit_edges': list of (u,v),
  'noisy_expectations': list of floats,
  'target': float (optional; for supervised training)
}

This module provides convenience functions and a small CLI to:
- convert a folder of Qiskit result JSONs to a JSONL of examples (best-effort)
- read a user-specified JSONL and validate/normalize to our schema
- generate a synthetic dataset for quick experiments

Notes:
- Qiskit "result.to_dict()" objects have variable structure depending on how results were
  saved. This parser tries a few common patterns. If your exported result format differs,
  adapt the `parse_qiskit_result_dict` function.
- The script does not require Qiskit; if Qiskit is installed we use it for safer parsing.

Usage examples:
  # generate toy dataset
  python src/data/make_dataset.py --mode toy --out data/raw/toy_dataset.jsonl --n 300

  # convert a directory of qiskit result json files
  python src/data/make_dataset.py --mode qiskit_dir --in_dir qiskit_results/ --out data/raw/ibmq_examples.jsonl

  # normalize an existing JSONL already close to our schema
  python src/data/make_dataset.py --mode normalize --in_file data/raw/raw_examples.jsonl --out data/raw/normalized.jsonl

"""

from typing import List, Dict, Any, Optional
import os
import json
import argparse
import math
import random

try:
    # optional for better compatibility if user has qiskit installed
    import qiskit
    from qiskit import QuantumCircuit
except Exception:
    qiskit = None

# ------------------ canonical schema helpers --------------------------------

def _ensure_schema(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a user-provided dict into the canonical example schema (best-effort).

    This will try to fill missing fields with safe defaults and convert gate lists to
    the expected (op, [qubits]) tuples.
    """
    ex = dict(example)  # shallow copy
    if 'id' not in ex:
        ex['id'] = ex.get('name', ex.get('job_id', str(random.getrandbits(64))))
    if 'n_qubits' not in ex:
        # try to infer from gates or edges
        if 'gates' in ex and isinstance(ex['gates'], (list, tuple)) and len(ex['gates']) > 0:
            # find max qubit index
            maxq = 0
            for g in ex['gates']:
                try:
                    qs = g[1]
                    maxq = max(maxq, max(qs))
                except Exception:
                    continue
            ex['n_qubits'] = int(maxq + 1)
        elif 'two_qubit_edges' in ex and len(ex['two_qubit_edges']) > 0:
            maxq = max(max(u, v) for u, v in ex['two_qubit_edges'])
            ex['n_qubits'] = int(maxq + 1)
        else:
            ex['n_qubits'] = int(ex.get('num_qubits', 1))

    # normalize gate representation to tuples
    if 'gates' in ex and isinstance(ex['gates'], list):
        norm = []
        for g in ex['gates']:
            if isinstance(g, dict):
                op = g.get('name') or g.get('op') or g.get('type')
                qs = g.get('qubits') or g.get('targets') or g.get('wires') or []
                norm.append((op, list(qs)))
            elif isinstance(g, (list, tuple)) and len(g) >= 2:
                op = g[0]
                qs = list(g[1])
                norm.append((op, qs))
            else:
                # unknown format: skip
                continue
        ex['gates'] = norm

    # normalize two_qubit_edges to list of tuples
    if 'two_qubit_edges' in ex and isinstance(ex['two_qubit_edges'], list):
        edges = []
        for e in ex['two_qubit_edges']:
            if isinstance(e, dict):
                u = e.get('u') or e.get('a') or e.get('from')
                v = e.get('v') or e.get('b') or e.get('to')
                edges.append((int(u), int(v)))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append((int(e[0]), int(e[1])))
        ex['two_qubit_edges'] = edges

    # noisy_expectations: ensure list of floats
    if 'noisy_expectations' in ex:
        ex['noisy_expectations'] = [float(x) for x in ex['noisy_expectations']]
    else:
        # maybe user provided counts; attempt to compute simple expectation from counts
        if 'counts' in ex:
            ex['noisy_expectations'] = [counts_to_expectation(ex['counts'])]
        else:
            ex['noisy_expectations'] = []

    return ex


# ------------------ small utility functions --------------------------------

def counts_to_expectation(counts: Dict[str, int], observable='Z') -> float:
    """Convert measurement counts (bitstring -> counts) to a simple expectation value.

    Default observable: Z on the first qubit (parity of first bit). This is naive but a
    reasonable default when user-supplied counts are available but no observable info.
    """
    total = sum(counts.values()) if isinstance(counts, dict) else 0
    if total == 0:
        return 0.0
    # define parity of left-most bit as measurement of Z on qubit 0 (assuming ordering)
    s = 0
    for bitstr, c in counts.items():
        # handle bitstrings like '0101' — leftmost is MSB
        bit0 = 0
        if len(bitstr) > 0:
            try:
                bit0 = int(bitstr[0])
            except Exception:
                bit0 = 0
        val = 1 if bit0 == 0 else -1
        s += val * c
    return float(s / total)


# ------------------ Qiskit result parsing (best-effort) --------------------

def parse_qiskit_result_dict(res_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attempt to extract examples from a qiskit Result.to_dict() structure.

    The function returns a list of canonical examples. This is intentionally permissive
    and may not capture all details; inspect outputs to ensure correctness.
    """
    examples = []
    # common keys: 'results' is a list
    results = res_dict.get('results') or res_dict.get('jobs') or []
    for idx, r in enumerate(results):
        ex = {}
        # Try to get the circuit name / header
        header = r.get('header') if isinstance(r, dict) else None
        name = None
        if header is not None:
            name = header.get('name') or header.get('compiled_circuit_qasm', None)
        ex['id'] = name or r.get('name') or f'qiskit_{idx}'

        # try counts
        counts = None
        if 'counts' in r:
            counts = r['counts']
        else:
            # often measurement data is inside 'data' dict
            data = r.get('data')
            if isinstance(data, dict):
                if 'counts' in data:
                    counts = data['counts']
                elif 'measurement_counts' in data:
                    counts = data['measurement_counts']
        if counts is not None:
            try:
                # ensure dict of str->int
                counts = {str(k): int(v) for k, v in counts.items()}
                ex['noisy_expectations'] = [counts_to_expectation(counts)]
            except Exception:
                ex['noisy_expectations'] = []

        # try to infer number of qubits
        if 'metadata' in r and isinstance(r['metadata'], dict) and 'n_qubits' in r['metadata']:
            ex['n_qubits'] = int(r['metadata']['n_qubits'])
        else:
            # fall back: try to infer from header or from bitstrings
            if counts is not None and len(next(iter(counts)))>0:
                ex['n_qubits'] = len(next(iter(counts)))
            else:
                ex['n_qubits'] = int(res_dict.get('header', {}).get('n_qubits', 1))

        # gates and two_qubit_edges are hard to recover from results; set empty lists
        ex['gates'] = []
        ex['two_qubit_edges'] = []

        examples.append(_ensure_schema(ex))
    return examples


# ------------------ JSONL normalization -----------------------------------

def normalize_jsonl(in_file: str, out_file: str):
    out = []
    with open(in_file, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            ex = _ensure_schema(obj)
            out.append(ex)
    with open(out_file, 'w') as f:
        for ex in out:
            f.write(json.dumps(ex) + '\n')
    return out_file


# ------------------ bulk convert qiskit result JSONs -----------------------

def convert_qiskit_dir(in_dir: str, out_file: str):
    all_examples = []
    for fname in os.listdir(in_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(in_dir, fname)
        with open(path, 'r') as f:
            try:
                obj = json.load(f)
            except Exception:
                continue
        examples = []
        # heuristic: if dict contains 'qobj' or 'results'
        if isinstance(obj, dict) and ('results' in obj or 'jobs' in obj):
            examples = parse_qiskit_result_dict(obj)
        else:
            # try to interpret as a single job result dict
            try:
                examples = parse_qiskit_result_dict({'results': [obj]})
            except Exception:
                examples = []
        all_examples.extend(examples)
    # write jsonl
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    return out_file


# ------------------ toy dataset generator ---------------------------------

def generate_toy_jsonl(out_file: str, n_samples: int = 200, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n_samples):
        q = rng.choice([4, 5, 6, 8])
        gates = []
        two = []
        # random local connectivity edges
        for a in range(q - 1):
            two.append((a, a + 1))
        # random gate list
        ng = rng.randint(q * 2, q * 6)
        for _ in range(ng):
            op = rng.choice(['H', 'X', 'Y', 'Z', 'CNOT'])
            if op == 'CNOT':
                a = rng.randint(0, q - 1)
                b = (a + rng.randint(1, q - 1)) % q
                gates.append((op, [a, b]))
            else:
                gates.append((op, [rng.randint(0, q - 1)]))
        noisy = [rng.gauss(0.0, 1.0) for _ in range(20)]
        # toy target function
        target = q * 0.1 + ng * 0.01 + rng.gauss(0.0, 0.05)
        ex = {'id': f'toy_{i}', 'n_qubits': q, 'gates': gates, 'two_qubit_edges': two, 'noisy_expectations': noisy, 'target': target}
        out.append(ex)
    # write jsonl
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        for ex in out:
            f.write(json.dumps(ex) + '\n')
    return out_file


# ------------------ CLI ---------------------------------------------------

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'normalize', 'qiskit_dir'], required=True)
    parser.add_argument('--in_file', help='input JSONL file (for normalize)')
    parser.add_argument('--in_dir', help='input directory (for qiskit_dir)')
    parser.add_argument('--out', required=True, help='output JSONL file')
    parser.add_argument('--n', type=int, default=200, help='number of toy samples')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'toy':
        p = generate_toy_jsonl(args.out, n_samples=args.n, seed=args.seed)
        print('Wrote toy dataset to', p)
    elif args.mode == 'normalize':
        if args.in_file is None:
            raise SystemExit('normalize mode requires --in_file')
        p = normalize_jsonl(args.in_file, args.out)
        print('Wrote normalized JSONL to', p)
    elif args.mode == 'qiskit_dir':
        if args.in_dir is None:
            raise SystemExit('qiskit_dir requires --in_dir')
        p = convert_qiskit_dir(args.in_dir, args.out)
        print('Converted Qiskit results to', p)


if __name__ == '__main__':
    main_cli()
