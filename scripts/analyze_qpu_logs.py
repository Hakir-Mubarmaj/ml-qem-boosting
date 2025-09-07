# Placeholder for analyze_qpu_logs.py

"""Analyze QPU logs / Result JSONs and produce normalized JSONL for pipeline.

This wraps src.data.make_dataset.parse_qiskit_result_dict and convert_qiskit_dir.
"""
import argparse
import os
from src.data.make_dataset import convert_qiskit_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', required=True, help='directory with qiskit result JSONs')
    parser.add_argument('--out-jsonl', required=True, help='output normalized jsonl')
    args = parser.parse_args()
    convert_qiskit_dir(args.in_dir, args.out_jsonl)
    print('Converted Qiskit JSONs in', args.in_dir, 'to', args.out_jsonl)

if __name__ == '__main__':
    main()
