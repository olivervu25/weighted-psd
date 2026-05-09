"""
run_runtime_table.py
====================
Benchmarks wall-clock time for PSD, tPSD (CE), and tPSD (SVD) across
(n, d, Q) configurations.  Prints a console table and a LaTeX table.

Usage
-----
python run_runtime_table.py
python run_runtime_table.py --n_reps 20
"""

from __future__ import annotations
import argparse
import time
import numpy as np

try:
    from wpsd import psd, tpsd
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from rfsd.wpsd import psd, tpsd


CONFIGS = [
    # (n,    d,   Q)
    ( 500,   4,   1),
    ( 500,   4,   4),
    ( 500,  16,   1),
    ( 500,  16,   4),
    (1000,  16,   1),
    (1000,  16,   4),
    (1000,  64,   1),
    (1000,  64,   4),
]


def _median_time(fn, *args, n_reps=10, **kwargs):
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _latex_table(rows):
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Median wall-clock time (seconds) per test call}',
        r'\begin{tabular}{rrrrrr}',
        r'\toprule',
        (r'$n$ & $d$ & $Q$'
         r' & $\mathrm{PSD}_{L_1}$ (s)'
         r' & $\lambda$-SPSD CE (s)'
         r' & $\lambda$-SPSD SVD (s) \\'),
        r'\midrule',
    ]
    for n, d, Q, t_psd, t_ce, t_svd in rows:
        lines.append(f"{n} & {d} & {Q} & {t_psd:.4f} & {t_ce:.4f} & {t_svd:.4f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main(n_reps=10, save_csv=None, save_latex=None):
    header = f"{'n':>6} {'d':>6} {'Q':>4}  {'PSD (s)':>10} {'CE (s)':>10} {'SVD (s)':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for n, d, Q in CONFIGS:
        rng = np.random.default_rng(36)
        X = rng.standard_normal((n, d))
        S = -X   # N(0,I) score

        t_psd = _median_time(psd,  X, S, Q=Q, seed=36,              n_reps=n_reps)
        t_ce  = _median_time(tpsd, X, S, Q=Q, seed=36, lambda_method='ce',  n_reps=n_reps)
        t_svd = _median_time(tpsd, X, S, Q=Q, seed=36, lambda_method='svd', n_reps=n_reps)

        rows.append((n, d, Q, t_psd, t_ce, t_svd))
        print(f"{n:>6} {d:>6} {Q:>4}  {t_psd:>10.4f} {t_ce:>10.4f} {t_svd:>10.4f}")

    latex_str = _latex_table(rows)

    print()
    print("% ── LaTeX table ────────────────────────────────────────────────")
    print(latex_str)

    if save_csv:
        import csv as csv_mod
        with open(save_csv, 'w', newline='') as f:
            w = csv_mod.writer(f)
            w.writerow(['n', 'd', 'Q', 'psd_s', 'ce_s', 'svd_s'])
            w.writerows(rows)
        print(f"CSV   → {save_csv}")

    if save_latex:
        with open(save_latex, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX → {save_latex}")


def _parse_args():
    p = argparse.ArgumentParser(description="Runtime benchmark for PSD / tPSD")
    p.add_argument("--n_reps",     type=int, default=10)
    p.add_argument("--save_csv",   type=str, default=None)
    p.add_argument("--save_latex", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_reps=args.n_reps, save_csv=args.save_csv, save_latex=args.save_latex)
