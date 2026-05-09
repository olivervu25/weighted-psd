"""
plot_psd_vs_wpsd_merged.py
==========================
Render the 2x4 PSD-vs-lambda-PSD figure (rejection rate on top, log10(MSNR)
on bottom) from the merged cache produced by run_merged_experiment.py.

Layout
------
  Top row     rejection rate vs d   (alpha line at 0.05)
  Bottom row  log10(MSNR) vs d      (reference line at log10(2), the
                                     expected median of log10(z^2) under H0)

Panels: Standard Gaussian | Perturbed Gaussian | Student-t | Laplace
Style:
  solid  + circle   = PSD          (uniform lambda)
  dashed + triangle = lambda-PSD   (adaptive lambda from the CE search)
  colour            = Q value (viridis, graded)

Caches read
-----------
  <cache_dir>/<scenario>_n*_reps*.pkl  (required, merged-runner output)

When multiple matching pickles exist, the most recently modified one wins,
or pass explicit --null/--varperturb/--student/--laplace paths to override.

Usage
-----
  python scripts/plot_psd_vs_wpsd_merged.py
  python scripts/plot_psd_vs_wpsd_merged.py \
      --cache_dir results/merged \
      --out results/figure_psd_vs_wpsd_merged.png
  python scripts/plot_psd_vs_wpsd_merged.py --q_list 0,2,4 --no_errorbars
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


SCENARIOS = [
    ('null',            'Standard Gaussian'),
    ('perturbed_gauss', 'Perturbed Gaussian'),
    ('gauss_t',         'Student-t'),
    ('laplace',         'Laplace'),
]


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def _find_pkl(cache_dir: Path, key: str) -> Path:
    """Most-recent merged-runner pickle for a scenario (rfsd_ excluded)."""
    pkls = sorted(
        (p for p in cache_dir.glob(f'{key}_n*_reps*.pkl')
         if not p.name.startswith('rfsd_')),
        key=lambda p: p.stat().st_mtime,
    )
    if not pkls:
        sys.exit(f'ERROR: no pickle found for scenario "{key}" in {cache_dir}')
    return pkls[-1]


def _load(pkl_path: Path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _pack(results, Q_list):
    """Pack per-(Q, d) results into per-Q numpy arrays for vectorised plotting."""
    d_list = results['d_list']
    out = {}
    for Q in Q_list:
        out[Q] = {
            'psd_rej':    np.array([results['by_d'][d]['psd' ][Q]['reject_mean'] for d in d_list]),
            'psd_rej_e':  np.array([results['by_d'][d]['psd' ][Q]['reject_se']   for d in d_list]),
            'wpsd_rej':   np.array([results['by_d'][d]['wpsd'][Q]['reject_mean'] for d in d_list]),
            'wpsd_rej_e': np.array([results['by_d'][d]['wpsd'][Q]['reject_se']   for d in d_list]),
            'psd_msnr':   np.array([results['by_d'][d]['psd' ][Q]['msnr_mean']   for d in d_list]),
            'psd_msnr_e': np.array([results['by_d'][d]['psd' ][Q]['msnr_se']     for d in d_list]),
            'wpsd_msnr':  np.array([results['by_d'][d]['wpsd'][Q]['msnr_mean']   for d in d_list]),
            'wpsd_msnr_e':np.array([results['by_d'][d]['wpsd'][Q]['msnr_se']     for d in d_list]),
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(panels, Q_list, alpha, n_reps, out_path,
                with_errorbars=True):
    """Render the 2 x len(panels) figure to disk.

    Parameters
    ----------
    panels : list of (title, packed_arrays, n_for_panel, d_list)
        One per scenario. `packed_arrays` comes from _pack().
    Q_list : list[int]
        Polynomial orders to draw (one colour per Q).
    alpha : float
        Significance level; drawn as a dotted reference line on the
        rejection-rate row.
    n_reps : int
        Repetitions; appears in the suptitle.
    out_path : str | Path
        PNG destination.
    with_errorbars : bool
        If True, error bars are SE bars from the merged cache; otherwise
        plain lines.

    Notes
    -----
    The dotted reference line on the bottom row is at log10(2): under H0,
    z^2 ~ chi^2(1), whose median is ~0.45, hence log10(median(z^2)) ~ -0.35.
    The exact reference is conventional; the visual purpose is to give a
    constant null baseline across panels.
    """
    n_panels = len(panels)
    q_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Q_list)))
    log10_ref = np.log10(2)

    fig, axes = plt.subplots(2, n_panels, figsize=(5 * n_panels, 8.5),
                             sharex=True)
    if n_panels == 1:
        axes = axes[:, None]

    for ci, (title, packed, n_panel, d_list) in enumerate(panels):
        ax_rej  = axes[0, ci]
        ax_msnr = axes[1, ci]

        for qi, Q in enumerate(Q_list):
            col = q_colors[qi]
            r = packed[Q]
            psd_kw  = dict(color=col, linestyle='-',  marker='o',
                           markersize=5, linewidth=1.6,
                           markeredgecolor=col, markerfacecolor=col)
            wpsd_kw = dict(color=col, linestyle='--', marker='^',
                           markersize=6, linewidth=1.6,
                           markeredgecolor=col, markerfacecolor=col)
            if with_errorbars:
                ax_rej.errorbar(d_list,  r['psd_rej'],   yerr=r['psd_rej_e'],
                                capsize=2.5, **psd_kw)
                ax_rej.errorbar(d_list,  r['wpsd_rej'],  yerr=r['wpsd_rej_e'],
                                capsize=2.5, **wpsd_kw)
                ax_msnr.errorbar(d_list, r['psd_msnr'],  yerr=r['psd_msnr_e'],
                                 capsize=2.5, **psd_kw)
                ax_msnr.errorbar(d_list, r['wpsd_msnr'], yerr=r['wpsd_msnr_e'],
                                 capsize=2.5, **wpsd_kw)
            else:
                ax_rej.plot(d_list,  r['psd_rej'],  **psd_kw)
                ax_rej.plot(d_list,  r['wpsd_rej'], **wpsd_kw)
                ax_msnr.plot(d_list, r['psd_msnr'], **psd_kw)
                ax_msnr.plot(d_list, r['wpsd_msnr'],**wpsd_kw)

        ax_rej.axhline(alpha,      linestyle=':', color='grey', linewidth=0.9)
        ax_msnr.axhline(log10_ref, linestyle=':', color='grey', linewidth=0.9)

        d_min, d_max = min(d_list), max(d_list)
        pad = (d_max / d_min) ** 0.04
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
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', type=str, default='results/merged')
    p.add_argument('--null',       type=str, default=None)
    p.add_argument('--varperturb', type=str, default=None)
    p.add_argument('--student',    type=str, default=None)
    p.add_argument('--laplace',    type=str, default=None)
    p.add_argument('--out', type=str,
                   default='results/figure_psd_vs_wpsd_merged.png')
    p.add_argument('--no_errorbars', action='store_true')
    p.add_argument('--q_list', type=str, default=None,
                   help='Comma-separated subset of Q values to plot. '
                        'Default: all Q values present in the cache.')
    return p.parse_args()


def main():
    args = _parse()
    cache_dir = Path(args.cache_dir)

    overrides = {
        'null':            args.null,
        'perturbed_gauss': args.varperturb,
        'gauss_t':         args.student,
        'laplace':         args.laplace,
    }

    loaded = []
    for key, label in SCENARIOS:
        path = overrides.get(key)
        path = Path(path) if path else _find_pkl(cache_dir, key)
        print(f'[{label}] using {path}')
        loaded.append((label, _load(path)))

    Q_list_cache = loaded[0][1]['Q_list']
    if args.q_list is None:
        Q_list = Q_list_cache
    else:
        Q_list = [int(q) for q in args.q_list.split(',')]
        missing = [q for q in Q_list if q not in Q_list_cache]
        if missing:
            sys.exit(f'ERROR: requested Q values not in cache: {missing} '
                     f'(cache has {Q_list_cache})')

    n_reps_set = {r['n_reps'] for _, r in loaded}
    n_reps = next(iter(n_reps_set)) if len(n_reps_set) == 1 else max(n_reps_set)
    alpha = loaded[0][1].get('alpha', 0.05)

    panels = []
    for label, results in loaded:
        panels.append((label, _pack(results, Q_list),
                       results['n'], results['d_list']))

    make_figure(panels, Q_list, alpha, n_reps, args.out,
                with_errorbars=not args.no_errorbars)


if __name__ == '__main__':
    main()
