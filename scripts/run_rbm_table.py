"""
run_rbm_table.py
================
Reproduce Table 1-style "null rejection rates for testing methods at different
perturbation levels in the RBM example", with wPSD added.

Methods included by default (matching the paper plus wPSD L=0..4):
    RFSD (RBM), Gauss FSSD-opt, IMQ KSD, Gauss KSD,
    PSD r1, PSD r2, PSD r3,
    wPSD L=0, wPSD L=1, wPSD L=2, wPSD L=3, wPSD L=4

Outputs (under --out_dir):
    rbm_table.csv        - long-format table (method x sigma_per)
    rbm_table.tex        - paper-style booktabs LaTeX
    rbm_table.md         - markdown table for quick inspection
    rbm_results.pkl      - raw {J: {test_name: [rate per sigma]}} dict

Usage
-----
    python scripts/run_rbm_table.py \\
        --rounds 100 --J 10 --dx 50 --dh 40 \\
        --sigmas 0,0.02,0.04,0.06 \\
        --out_dir results
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Silence harmless autograd "divide by zero in log" from kgof's RBM density
# (log(exp(z)+exp(-z)) hits transient log(0) for extreme |z|; the resulting
# score is still correct, see comments in conversation log).
warnings.filterwarnings(
    'ignore',
    message='divide by zero encountered in log',
    category=RuntimeWarning,
)
np.seterr(divide='ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rfsd.experiments.gof_testing_experiments as goft_exp


PAPER_METHODS = [
    ('RFSD (RBM)',     'RFSD'),
    ('Gauss FSSD-opt', 'FSSD-opt'),
    ('IMQ KSD',        'IMQ KSD'),
    ('Gauss KSD',      'Gauss KSD'),
]
PSD_METHODS = [
    ('PSD L=0', 'PSD ($Q=0$)'),
    ('PSD L=1', 'PSD ($Q=1$)'),
    ('PSD L=2', 'PSD ($Q=2$)'),
    ('PSD L=3', 'PSD ($Q=3$)'),
    ('PSD L=4', 'PSD ($Q=4$)'),
]
WPSD_METHODS = [
    ('wPSD L=0', r'$\lambda$-PSD ($Q=0$)'),
    ('wPSD L=1', r'$\lambda$-PSD ($Q=1$)'),
    ('wPSD L=2', r'$\lambda$-PSD ($Q=2$)'),
    ('wPSD L=3', r'$\lambda$-PSD ($Q=3$)'),
    ('wPSD L=4', r'$\lambda$-PSD ($Q=4$)'),
]


def _format_csv(out_path, methods, display_labels, sigmas, rates):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as csv_mod
    with open(out_path, 'w', newline='') as f:
        w = csv_mod.writer(f)
        w.writerow(['method'] + [f'sigma={s:g}' for s in sigmas])
        for tn, label in zip(methods, display_labels):
            row = [label] + [f'{r:.2f}' for r in rates[tn]]
            w.writerow(row)
    print(f'CSV   -> {out_path}')


def _format_md(out_path, methods, display_labels, sigmas, rates):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ['Method'] + [f'sigma={s:g}' for s in sigmas]
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    for tn, label in zip(methods, display_labels):
        row = [label] + [f'{r:.2f}' for r in rates[tn]]
        lines.append('| ' + ' | '.join(row) + ' |')
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'MD    -> {out_path}')


def _format_latex(out_path, methods, display_labels, sigmas, rates,
                  rounds, dx, dh, J):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(sigmas)
    col_spec = 'l' + ' c' * n_cols
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        (r'\caption{Null rejection rates at perturbation levels '
         r'in the RBM example '
         rf'($d_x={dx}$, $d_h={dh}$, $J={J}$, {rounds} rounds)' + r'}'),
        r'\label{tab:rbm_with_wpsd}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        'PERTURBATION: & ' + ' & '.join(f'{s:g}' for s in sigmas) + r' \\',
        r'\midrule',
    ]
    for tn, label in zip(methods, display_labels):
        cells = [label] + [f'{r:.2f}' for r in rates[tn]]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    out_path.write_text('\n'.join(lines))
    print(f'LaTeX -> {out_path}')


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds',  type=int, default=100)
    p.add_argument('--J',       type=int, default=10)
    p.add_argument('--dx',      type=int, default=50)
    p.add_argument('--dh',      type=int, default=40)
    p.add_argument('--gamma',   type=float, default=0.25)
    p.add_argument('--sigmas',  type=str,
                   default='0,0.02,0.04,0.06',
                   help='Comma-separated perturbation levels.')
    p.add_argument('--alpha',   type=float, default=0.05)
    p.add_argument('--out_dir', type=str, default='results')
    p.add_argument('--cache',   type=str,
                   default='results/rbm_table_cache.pkl',
                   help='Pickled raw results; reused if present.')
    p.add_argument('--force',   action='store_true',
                   help='Ignore cache and re-run.')
    return p.parse_args()


def main():
    args   = _parse()
    sigmas = [float(x) for x in args.sigmas.split(',')]

    all_groups = PAPER_METHODS + PSD_METHODS + WPSD_METHODS
    methods        = [m for m, _ in all_groups]
    display_labels = [d for _, d in all_groups]

    cache_path = Path(args.cache)

    if cache_path.exists() and not args.force:
        print(f'Cache hit -> {cache_path}')
        with open(cache_path, 'rb') as f:
            blob = pickle.load(f)
        all_results = blob['all_results']
        all_params  = blob['all_params']
    else:
        print(f'Running RBM experiment: dx={args.dx}, dh={args.dh}, '
              f'rounds={args.rounds}, J={args.J}, sigmas={sigmas}')
        t0 = time.time()
        all_results, all_params = goft_exp.run_goft_rbm_experiment_group(
            test_names=methods,
            experiment=goft_exp.run_rbm_fssd_experiment,
            sigmaPers=sigmas,
            rounds=args.rounds,
            test_alpha=args.alpha,
            plot_results=False,
            Js=[args.J],
            gamma=args.gamma,
            dx=args.dx,
            dh=args.dh,
        )
        print(f'Done in {(time.time() - t0) / 60:.1f} min')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'all_results': all_results,
                         'all_params':  all_params}, f)
        print(f'Cached -> {cache_path}')

    rates = all_results[args.J]
    missing = [m for m in methods if m not in rates]
    if missing:
        sys.exit(f'ERROR: missing methods in results dict: {missing}')

    out_dir = Path(args.out_dir)
    _format_csv  (out_dir / 'rbm_table.csv', methods, display_labels, sigmas, rates)
    _format_md   (out_dir / 'rbm_table.md',  methods, display_labels, sigmas, rates)
    _format_latex(out_dir / 'rbm_table.tex', methods, display_labels, sigmas, rates,
                  rounds=args.rounds, dx=args.dx, dh=args.dh, J=args.J)


if __name__ == '__main__':
    main()
