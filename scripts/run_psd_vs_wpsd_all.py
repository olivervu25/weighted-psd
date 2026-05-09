"""
run_psd_vs_wpsd_all.py
======================
Sweep d in {2,4,8,16,32,64,128} and Q in {0..4} for plain PSD vs wPSD
across four scenarios (Standard Gaussian, Perturbed Gaussian, Student-t,
Laplace) and assemble one 2x4 combined figure:

    Top row    -> rejection rate vs d
    Bottom row -> log10(MSNR) vs d

One column per scenario. Color = Q. Solid+circle = PSD, dashed+triangle = wPSD.
Per-scenario results are pickled to a cache directory so re-running this
script just reloads them and rebuilds the figure.

Usage
-----
    python scripts/run_psd_vs_wpsd_all.py \\
        --n 1000 --n_reps 100 \\
        --d_list 2,4,8,16,32,64,128 \\
        --q_list 0,1,2,3,4 \\
        --out results/psd_vs_wpsd_combined.png

    # Force a rerun (ignore cache):
    python scripts/run_psd_vs_wpsd_all.py --force
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dim_experiment import run_dim_experiment  # noqa: E402


# (key, display_label, n_override)
# n_override=None -> use the global --n flag.
# Student-t uses 2000 to match the popular-methods convention
# (see scripts/run_gof_experiment.py: dict(ds=args.d, df=10, n=2000)).
SCENARIOS = [
    ('null',            'Standard Gaussian', None),
    ('perturbed_gauss', 'Perturbed Gaussian', None),
    ('gauss_t',         'Student-t',          2000),
    ('laplace',         'Laplace',            None),
]


def _arrays(results, Q_list, d_list):
    """Pack per-(Q,d) results dict into per-Q numpy arrays for plotting."""
    out = {}
    for Q in Q_list:
        out[Q] = {
            'psd_rej':    np.array([results[Q][d]['psd']['reject_mean'] for d in d_list]),
            'psd_rej_e':  np.array([results[Q][d]['psd']['reject_std']  for d in d_list]),
            'wpsd_rej':   np.array([results[Q][d]['wpsd']['reject_mean']for d in d_list]),
            'wpsd_rej_e': np.array([results[Q][d]['wpsd']['reject_std'] for d in d_list]),
            'psd_msnr':   np.array([results[Q][d]['psd']['msnr_mean']   for d in d_list]),
            'psd_msnr_e': np.array([results[Q][d]['psd']['msnr_std']    for d in d_list]),
            'wpsd_msnr':  np.array([results[Q][d]['wpsd']['msnr_mean']  for d in d_list]),
            'wpsd_msnr_e':np.array([results[Q][d]['wpsd']['msnr_std']   for d in d_list]),
        }
    return out


def make_combined_figure(per_scenario, Q_list, d_list, alpha, n_reps,
                         out_path):
    """per_scenario: list of (title, packed_arrays, n_for_panel) tuples."""
    n_panels = len(per_scenario)
    q_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Q_list)))
    log10_ref = np.log10(2)

    fig, axes = plt.subplots(2, n_panels, figsize=(5 * n_panels, 8.5),
                             sharex=True)

    for ci, (title, packed, n_panel) in enumerate(per_scenario):
        ax_rej  = axes[0, ci]
        ax_msnr = axes[1, ci]

        for qi, Q in enumerate(Q_list):
            col = q_colors[qi]
            r = packed[Q]
            ax_rej.errorbar(d_list, r['psd_rej'],  yerr=r['psd_rej_e'],
                            color=col, linestyle='-',  marker='o',
                            markersize=5, linewidth=1.6, capsize=2.5,
                            markeredgecolor=col, markerfacecolor=col)
            ax_rej.errorbar(d_list, r['wpsd_rej'], yerr=r['wpsd_rej_e'],
                            color=col, linestyle='--', marker='^',
                            markersize=6, linewidth=1.6, capsize=2.5,
                            markeredgecolor=col, markerfacecolor=col)
            ax_msnr.errorbar(d_list, r['psd_msnr'],  yerr=r['psd_msnr_e'],
                             color=col, linestyle='-',  marker='o',
                             markersize=5, linewidth=1.6, capsize=2.5,
                             markeredgecolor=col, markerfacecolor=col)
            ax_msnr.errorbar(d_list, r['wpsd_msnr'], yerr=r['wpsd_msnr_e'],
                             color=col, linestyle='--', marker='^',
                             markersize=6, linewidth=1.6, capsize=2.5,
                             markeredgecolor=col, markerfacecolor=col)

        ax_rej.axhline(alpha,      linestyle=':', color='grey', linewidth=0.9)
        ax_msnr.axhline(log10_ref, linestyle=':', color='grey', linewidth=0.9)

        # Tight x-limits with a small symmetric pad on log scale so markers
        # at the endpoints don't get clipped.
        d_min, d_max = min(d_list), max(d_list)
        pad = (d_max / d_min) ** 0.04   # ~4% pad on each side in log-space
        for ax in (ax_rej, ax_msnr):
            ax.set_xscale('log')
            ax.set_xticks(d_list)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlim(d_min / pad, d_max * pad)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.25, linewidth=0.5)

        ax_rej.set_title(f'{title}  (n={n_panel})', fontsize=13)
        ax_rej.set_ylim(0, 1.05)
        ax_msnr.set_xlabel('$d$', fontsize=11)

        if ci == 0:
            ax_rej.set_ylabel('rejection rate', fontsize=11)
            ax_msnr.set_ylabel(r'$\log_{10}(\mathrm{MSNR})$', fontsize=11)

    q_handles = [
        Line2D([0], [0], color=c, linestyle='-', linewidth=1.8,
               marker='o', markersize=6, markerfacecolor=c, markeredgecolor=c,
               label=f'$Q={q}$')
        for q, c in zip(Q_list, q_colors)
    ]
    method_handles = [
        Line2D([0], [0], color='black', linestyle='-',  linewidth=1.8,
               marker='o', markersize=6, label=r'PSD (uniform $\lambda$)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.8,
               marker='^', markersize=7, label=r'$\lambda$-PSD (adaptive)'),
    ]
    fig.legend(handles=q_handles, loc='lower center', ncol=len(Q_list),
               frameon=False, fontsize=10, bbox_to_anchor=(0.32, -0.02))
    fig.legend(handles=method_handles, loc='lower center', ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.78, -0.02))

    fig.suptitle(
        f'PSD vs $\\lambda$-PSD: rejection rate (top) and '
        f'$\\log_{{10}}$(MSNR) (bottom) — {n_reps} reps',
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved figure -> {out_path}')


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--n',      type=int,   default=1000)
    p.add_argument('--n_reps', type=int,   default=100)
    p.add_argument('--alpha',  type=float, default=0.05)
    p.add_argument('--q_list', type=str,   default='0,1,2,3,4')
    p.add_argument('--d_list', type=str,   default='2,4,8,16,32,64,128')
    p.add_argument('--seed',   type=int,   default=36)
    p.add_argument('--split_ratio', type=float, default=None,
                   help='Fraction of data used for training; passed to '
                        'both psd and tpsd. None -> per-method defaults '
                        '(psd=0.2, tpsd=0.3). 0.7 -> 35/35/30 split for wPSD.')
    p.add_argument('--out',    type=str,
                   default='results/psd_vs_wpsd_combined.png')
    p.add_argument('--cache_dir', type=str,
                   default='results/psd_vs_wpsd_cache',
                   help='Per-scenario pickle cache; rerunning reuses it.')
    p.add_argument('--csv_dir',   type=str,
                   default='results/psd_vs_wpsd_cache',
                   help='Where to write per-scenario CSVs alongside pickles.')
    p.add_argument('--force', action='store_true',
                   help='Ignore cache and re-run every scenario.')
    return p.parse_args()


def _save_scenario_csv(csv_path, results, Q_list, d_list):
    """Write a long-format CSV: one row per (Q, d), both methods side by side."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['Q', 'd',
              'psd_rej_mean',  'psd_rej_se',
              'wpsd_rej_mean', 'wpsd_rej_se',
              'psd_msnr_mean', 'psd_msnr_se',
              'wpsd_msnr_mean','wpsd_msnr_se']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for Q in Q_list:
            for d in d_list:
                rp = results[Q][d]['psd']
                rw = results[Q][d]['wpsd']
                w.writerow([Q, d,
                            rp['reject_mean'], rp['reject_std'],
                            rw['reject_mean'], rw['reject_std'],
                            rp['msnr_mean'],   rp['msnr_std'],
                            rw['msnr_mean'],   rw['msnr_std']])
    print(f'  wrote CSV -> {csv_path}')


def main():
    args = _parse()
    Q_list = [int(q) for q in args.q_list.split(',')]
    d_list = [int(d) for d in args.d_list.split(',')]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    split_tag = ('default' if args.split_ratio is None
                 else f'sr{args.split_ratio:g}')

    def _cache_tag(n_for_scenario):
        return (
            f"n{n_for_scenario}_reps{args.n_reps}"
            f"_d-{'-'.join(map(str, d_list))}"
            f"_Q-{'-'.join(map(str, Q_list))}"
            f"_{split_tag}"
        )

    per_scenario = []
    for key, label, n_override in SCENARIOS:
        n_scen = args.n if n_override is None else n_override
        tag = _cache_tag(n_scen)
        cache_path = cache_dir / f'{key}_{tag}.pkl'
        csv_path   = csv_dir   / f'{key}_{tag}.csv'

        if cache_path.exists() and not args.force:
            print(f'[{label}] cache hit (n={n_scen}) -> {cache_path}')
            with open(cache_path, 'rb') as f:
                results = pickle.load(f)
        else:
            print(f'[{label}] running ({args.n_reps} reps, n={n_scen}, '
                  f'split_ratio={args.split_ratio}) ...')
            t0 = time.time()
            results = run_dim_experiment(
                experiment=key,
                n=n_scen,
                d_list=d_list,
                Q_list=Q_list,
                n_reps=args.n_reps,
                alpha=args.alpha,
                base_seed=args.seed,
                verbose=True,
                split_ratio=args.split_ratio,
            )
            print(f'[{label}] done in {time.time() - t0:.1f}s')
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)

        # Always (re)write the CSV so manual edits / new fields are picked up.
        _save_scenario_csv(csv_path, results, Q_list, d_list)
        per_scenario.append((label, _arrays(results, Q_list, d_list), n_scen))

    make_combined_figure(per_scenario, Q_list, d_list,
                         alpha=args.alpha, n_reps=args.n_reps,
                         out_path=args.out)


if __name__ == '__main__':
    main()
