"""
export_results_csv.py
=====================
Export rejection rates from cached *-stored-data/ dirs into a single combined CSV.

Usage:
  python scripts/export_results_csv.py \
      --null    "results/goftest-null-...-stored-data" \
      --varperturb "results/goftest-variance_perturb-...-stored-data" \
      --student "results/goftest-student-t-...-stored-data" \
      --laplace "results/goftest-laplace-...-stored-data" \
      --out results/popular_methods_results.csv

Output schema:
  experiment, test, dim, reject_rate, n, rounds, alpha, df
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfsd.util import restore_object


EXPERIMENT_LABELS = {
    'null':       'Standard Gaussian',
    'varperturb': 'Perturbed Gaussian',
    'student':    'Student-t',
    'laplace':    'Laplace',
}


def _load(stored_dir: Path):
    results = restore_object(str(stored_dir), 'results')
    params  = restore_object(str(stored_dir), 'params')
    J = next(iter(results.keys()))   # only one J in our runs
    return params, results[J]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--null',       required=True)
    p.add_argument('--varperturb', required=True)
    p.add_argument('--student',    required=True)
    p.add_argument('--laplace',    required=True)
    p.add_argument('--out', default='results/popular_methods_results.csv')
    args = p.parse_args()

    rows = []
    for key, label in EXPERIMENT_LABELS.items():
        d = Path(getattr(args, key))
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")
        params, rates = _load(d)
        dims = list(params['variable_values'])
        n = params.get('n', '')
        rounds = params.get('rounds', '')
        alpha = params.get('test_alpha', '')
        df = params.get('df', '')   # only present for student-t

        for test_name, rrs in rates.items():
            for dim, rr in zip(dims, rrs):
                rows.append({
                    'experiment': label,
                    'test':       test_name,
                    'dim':        dim,
                    'reject_rate': float(rr),
                    'n':          n,
                    'rounds':     rounds,
                    'alpha':      alpha,
                    'df':         df,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'experiment', 'test', 'dim', 'reject_rate',
            'n', 'rounds', 'alpha', 'df',
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved → {out}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
