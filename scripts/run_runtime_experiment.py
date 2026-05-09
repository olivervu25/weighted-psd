"""
run_runtime_experiment.py
=========================
Wall-clock benchmark for lambda-PSD_{L1} (tpsd with the default CE lambda
search) under the standard Gaussian setting. Reproduces paper Figures 3 & 4.

  Figure 3: runtime vs dimension d  (n = 1000 fixed)
  Figure 4: runtime vs sample size n (d = 16 fixed)

Both figures are produced from a single invocation. Each curve is one
polynomial order Q in {0, 1, 2, 3, 4}; both axes are on a log scale.

Methodology
-----------
For each (Q, sweep_value) cell:
  * one warmup tpsd call to absorb JIT / cache effects,
  * n_reps timed calls of tpsd,
  * the median wall-clock time is reported (median rather than mean is more
    robust to occasional GC/IO pauses).

Outputs
-------
Caches under --cache_dir (default results/runtime/):
  runtime_vs_d_<tag>.{pkl,csv}
  runtime_vs_n_<tag>.{pkl,csv}

Figures under --out_dir (default results/):
  figure_runtime_vs_d.png
  figure_runtime_vs_n.png

Usage
-----
  python scripts/run_runtime_experiment.py
  python scripts/run_runtime_experiment.py --n_reps 30 --force
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kgof.util import NumpySeedContext

from rfsd.wpsd import tpsd


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

D_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_LIST = [100, 300, 1000, 3000, 10000]
Q_LIST = [0, 1, 2, 3, 4]

D_SWEEP_N = 1000   # fixed sample size n used in the runtime-vs-d sweep (Fig 3)
N_SWEEP_D = 16     # fixed dimension d used in the runtime-vs-n sweep (Fig 4)


# Discrete ggplot2-style palette so the figures visually match the paper.
Q_COLORS = {
    0: '#F8766D',  # salmon
    1: '#A3A500',  # olive
    2: '#00BF7D',  # green-teal
    3: '#00B0F6',  # cyan-blue
    4: '#E76BF3',  # pink
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_tpsd(X, S, Q, seed, n_reps, warmup=1):
    """Time tpsd n_reps times and return (mean, median, std) in seconds.

    The first call always pays JIT / library-caching costs, so we discard
    `warmup` initial calls. Each timed call is wrapped in NumpySeedContext
    so the CE lambda search inside tpsd consumes deterministic global
    randomness (same trick as in run_merged_experiment.py); this does not
    affect timings noticeably but keeps the script reproducible.
    """
    for _ in range(warmup):
        with NumpySeedContext(seed=seed):
            tpsd(X, S, Q=Q, seed=seed)

    times = np.empty(n_reps)
    for i in range(n_reps):
        with NumpySeedContext(seed=seed + i):
            t0 = time.perf_counter()
            tpsd(X, S, Q=Q, seed=seed + i)
            times[i] = time.perf_counter() - t0
    return (float(times.mean()),
            float(np.median(times)),
            float(times.std(ddof=1) if n_reps > 1 else 0.0))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_dim(n_fixed, d_list, q_list, n_reps, base_seed, verbose=True):
    """Sweep dimension d at fixed n; produces data for paper Figure 3."""
    results = {
        'sweep':  'd',
        'n':      n_fixed,
        'd_list': list(d_list),
        'Q_list': list(q_list),
        'n_reps': n_reps,
        'by_Qd':  {Q: {} for Q in q_list},
    }

    for Q in q_list:
        if verbose:
            print(f'  Q={Q}', end='  ', flush=True)
        for d in d_list:
            rng = np.random.default_rng(base_seed + d)
            X = rng.standard_normal((n_fixed, d))
            S = -X
            mean, median, std = _time_tpsd(X, S, Q=Q, seed=base_seed + Q * 10_000 + d,
                                            n_reps=n_reps)
            results['by_Qd'][Q][d] = {'mean': mean, 'median': median, 'std': std}
            if verbose:
                print(f'd={d}:{median*1000:.2f}ms', end='  ', flush=True)
        if verbose:
            print()
    return results


def sweep_n(d_fixed, n_list, q_list, n_reps, base_seed, verbose=True):
    """Sweep sample size n at fixed d; produces data for paper Figure 4."""
    results = {
        'sweep':  'n',
        'd':      d_fixed,
        'n_list': list(n_list),
        'Q_list': list(q_list),
        'n_reps': n_reps,
        'by_Qn':  {Q: {} for Q in q_list},
    }

    for Q in q_list:
        if verbose:
            print(f'  Q={Q}', end='  ', flush=True)
        for n in n_list:
            rng = np.random.default_rng(base_seed + n)
            X = rng.standard_normal((n, d_fixed))
            S = -X
            mean, median, std = _time_tpsd(X, S, Q=Q, seed=base_seed + Q * 10_000 + n,
                                            n_reps=n_reps)
            results['by_Qn'][Q][n] = {'mean': mean, 'median': median, 'std': std}
            if verbose:
                print(f'n={n}:{median*1000:.2f}ms', end='  ', flush=True)
        if verbose:
            print()
    return results


# ---------------------------------------------------------------------------
# Plotting (log-log, matching paper figures)
# ---------------------------------------------------------------------------

def _plot_runtime(results, x_key, x_label, title, out_path):
    """Render a single log-log runtime figure (one curve per Q).

    Parameters
    ----------
    results : dict
        Output of sweep_dim() (use x_key='d_list') or sweep_n()
        (use x_key='n_list').
    x_key : str
        Name of the sweep-variable list inside `results`.
    x_label, title : str
        Axis label and (optional) figure title.
    out_path : str | Path
        Where to write the PNG.
    """
    xs = results[x_key]
    by = results['by_Qd' if x_key == 'd_list' else 'by_Qn']

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for Q in results['Q_list']:
        ys = np.array([by[Q][x]['median'] for x in xs])
        ax.plot(xs, ys,
                color=Q_COLORS.get(Q, None),
                marker='o', markersize=5, linewidth=1.6,
                label=f'{Q}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    leg = ax.legend(title='Q', loc='upper center',
                    bbox_to_anchor=(0.5, 1.12),
                    ncol=len(results['Q_list']),
                    frameon=False, fontsize=10)
    leg.get_title().set_fontsize(10)

    if title:
        ax.set_title(title, fontsize=11, pad=24)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _save_csv_d(csv_path, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    Q_list = results['Q_list']
    header = ['d'] + [f'Q{Q}_median_s' for Q in Q_list] + [f'Q{Q}_mean_s' for Q in Q_list]
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in results['d_list']:
            row = [d]
            row += [results['by_Qd'][Q][d]['median'] for Q in Q_list]
            row += [results['by_Qd'][Q][d]['mean']   for Q in Q_list]
            w.writerow(row)


def _save_csv_n(csv_path, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    Q_list = results['Q_list']
    header = ['n'] + [f'Q{Q}_median_s' for Q in Q_list] + [f'Q{Q}_mean_s' for Q in Q_list]
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for n in results['n_list']:
            row = [n]
            row += [results['by_Qn'][Q][n]['median'] for Q in Q_list]
            row += [results['by_Qn'][Q][n]['mean']   for Q in Q_list]
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--n_reps',    type=int,   default=20,
                   help='Number of repetitions per (Q, sweep-value).')
    p.add_argument('--seed',      type=int,   default=36)
    p.add_argument('--d_list',    type=str,
                   default=','.join(map(str, D_LIST)))
    p.add_argument('--n_list',    type=str,
                   default=','.join(map(str, N_LIST)))
    p.add_argument('--q_list',    type=str,
                   default=','.join(map(str, Q_LIST)))
    p.add_argument('--n_for_d',   type=int, default=D_SWEEP_N,
                   help='Fixed n used for the runtime-vs-d sweep.')
    p.add_argument('--d_for_n',   type=int, default=N_SWEEP_D,
                   help='Fixed d used for the runtime-vs-n sweep.')
    p.add_argument('--cache_dir', type=str, default='results/runtime')
    p.add_argument('--out_dir',   type=str, default='results')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = _parse()
    d_list = [int(x) for x in args.d_list.split(',')]
    n_list = [int(x) for x in args.n_list.split(',')]
    q_list = [int(x) for x in args.q_list.split(',')]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (f"reps{args.n_reps}_Q-{'-'.join(map(str, q_list))}_s{args.seed}")

    # --- Figure 3: runtime vs d (fixed n) --------------------------------
    pkl_d = cache_dir / f'runtime_vs_d_n{args.n_for_d}_d-{"-".join(map(str,d_list))}_{tag}.pkl'
    csv_d = pkl_d.with_suffix('.csv')

    if pkl_d.exists() and not args.force:
        print(f'\n[runtime vs d] cache hit -> {pkl_d}')
        with open(pkl_d, 'rb') as f:
            res_d = pickle.load(f)
    else:
        print(f'\n[runtime vs d] sweeping d in {d_list}, n={args.n_for_d}, '
              f'reps={args.n_reps}')
        t0 = time.time()
        res_d = sweep_dim(n_fixed=args.n_for_d, d_list=d_list, q_list=q_list,
                          n_reps=args.n_reps, base_seed=args.seed)
        print(f'[runtime vs d] total {time.time()-t0:.1f}s')
        with open(pkl_d, 'wb') as f:
            pickle.dump(res_d, f)
    _save_csv_d(csv_d, res_d)
    print(f'  pickle -> {pkl_d}')
    print(f'  csv    -> {csv_d}')

    _plot_runtime(
        res_d, x_key='d_list', x_label='Dimension d',
        title=None,
        out_path=out_dir / 'figure_runtime_vs_d.png',
    )

    # --- Figure 4: runtime vs n (fixed d) --------------------------------
    pkl_n = cache_dir / f'runtime_vs_n_d{args.d_for_n}_n-{"-".join(map(str,n_list))}_{tag}.pkl'
    csv_n = pkl_n.with_suffix('.csv')

    if pkl_n.exists() and not args.force:
        print(f'\n[runtime vs n] cache hit -> {pkl_n}')
        with open(pkl_n, 'rb') as f:
            res_n = pickle.load(f)
    else:
        print(f'\n[runtime vs n] sweeping n in {n_list}, d={args.d_for_n}, '
              f'reps={args.n_reps}')
        t0 = time.time()
        res_n = sweep_n(d_fixed=args.d_for_n, n_list=n_list, q_list=q_list,
                        n_reps=args.n_reps, base_seed=args.seed)
        print(f'[runtime vs n] total {time.time()-t0:.1f}s')
        with open(pkl_n, 'wb') as f:
            pickle.dump(res_n, f)
    _save_csv_n(csv_n, res_n)
    print(f'  pickle -> {pkl_n}')
    print(f'  csv    -> {csv_n}')

    _plot_runtime(
        res_n, x_key='n_list', x_label='Sample size n',
        title=None,
        out_path=out_dir / 'figure_runtime_vs_n.png',
    )


if __name__ == '__main__':
    main()
