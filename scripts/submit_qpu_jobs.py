# Placeholder for submit_qpu_jobs.py

"""Submit QPU jobs helper (template).

This script is a safe template to submit jobs to IBMQ using Qiskit. It
supports a dry-run mode that does not contact IBM, and a real mode where
you must have Qiskit installed and be logged in (via qiskit.IBMQ.save_account).
"""
import argparse
import os
import json
from typing import List

def list_qasm_files(path: str) -> List[str]:
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n.lower().endswith('.qasm') or n.lower().endswith('.json'):
                files.append(os.path.join(root, n))
    return files

def dry_run_report(files: List[str], out_json: str):
    report = {'n_jobs': len(files), 'files': files[:100]}
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print('Dry-run report written to', out_json)

def submit_to_ibmq(circuits: List[str], backend_name: str, shots: int = 8192):
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile
        from qiskit import IBMQ
        from qiskit.providers.jobstatus import JobStatus
    except Exception as e:
        raise RuntimeError('Qiskit not available or not configured: ' + str(e))

    provider = IBMQ.get_provider(hub='ibm-q')
    backend = provider.get_backend(backend_name)
    job_ids = []
    for path in circuits:
        try:
            qc = QuantumCircuit.from_qasm_file(path)
        except Exception:
            print('Skipping (cannot parse as QASM):', path)
            continue
        tqc = transpile(qc, backend=backend)
        job = backend.run(tqc, shots=shots)
        job_ids.append(job.job_id())
        print('Submitted job', job.job_id(), 'for', path)
    return job_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--circuits', type=str, required=True, help='file or directory of circuits (qasm)')
    parser.add_argument('--out', type=str, default='qpu_submit_report.json')
    parser.add_argument('--backend', type=str, default='ibmq_guadalupe')
    parser.add_argument('--shots', type=int, default=8192)
    args = parser.parse_args()

    if os.path.isdir(args.circuits):
        files = list_qasm_files(args.circuits)
    else:
        files = [args.circuits]

    if args.dry_run:
        dry_run_report(files, args.out)
        return

    job_ids = submit_to_ibmq(files, args.backend, shots=args.shots)
    with open(args.out, 'w') as f:
        json.dump({'job_ids': job_ids}, f)
    print('Submitted', len(job_ids), 'jobs; metadata written to', args.out)

if __name__ == '__main__':
    main()
