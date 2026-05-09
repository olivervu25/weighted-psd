"""
plot_popular_methods_merged.py
==============================
Render the 1x4 popular-methods figure (rejection rate vs dimension d) from
the merged cache produced by run_merged_experiment.py. Optionally overlays
the RFSD curve from run_rfsd_experiment.py if its pickle is present.

Layout
------
Panels (left to right): Standard Gaussian | Perturbed Gaussian | Student-t | Laplace
Lines per panel:
  IMQ KSD, Gauss KSD, Gauss FSSD-opt, RFSD   solid + square (tab10 palette)
  lambda-PSD L=0..4                          dashed + triangle (viridis palette)

Caches read
-----------
  <cache_dir>/<scenario>_n*_reps*.pkl       (required, from merged runner)
  <cache_dir>/rfsd_<scenario>_n*_reps*.pkl  (optional, adds RFSD line)

When multiple matching pickles exist, the most recently modified one wins,
or pass explicit --null/--varperturb/--student/--laplace paths to override.

Usage
-----
  python scripts/plot_popular_methods_merged.py
  python scripts/plot_popular_methods_merged.py \
      --cache_dir results/merged \
      --out results/figure_popular_methods_merged.png
  python scripts/plot_popular_methods_merged.py --no_errorbars
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


# ---------------------------------------------------------------------------
# Styling
#
# BASE_TESTS share a 'solid line + square marker' style (tab10 palette);
# the lambda-PSD family uses 'dashed line + triangle marker' (viridis,
# graded by Q). PLOT_ORDER fixes draw order so the legend is consistent
# across panels.
# ---------------------------------------------------------------------------

BASE_TESTS = ['IMQ KSD', 'Gauss KSD', 'Gauss FSSD-opt', 'RFSD']
WPSD_QS    = [0, 1, 2, 3, 4]

PLOT_ORDER = (
    BASE_TESTS
    + [f'wPSD L={Q}' for Q in WPSD_QS]
)

LEGEND_LABELS = {
    'IMQ KSD':        'IMQ KSD',
    'Gauss KSD':      'Gauss KSD',
    'Gauss FSSD-opt': 'Gauss FSSD-opt',
    'RFSD':           'RFSD',
    **{f'wPSD L={Q}': rf'$\lambda$-PSD$_{{L_1}}$ ($Q={Q}$)' for Q in WPSD_QS},
}

SCENARIOS = [
    ('null',            'Standard Gaussian'),
    ('perturbed_gauss', 'Perturbed Gaussian'),
    ('gauss_t',         'Student-t'),
    ('laplace',         'Laplace'),
]


def _style_dict():
    base_palette = plt.cm.tab10(np.linspace(0, 1, 10))[:len(BASE_TESTS)]
    wpsd_palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(WPSD_QS)))
    style = {}
    for tn, c in zip(BASE_TESTS, base_palette):
        style[tn] = dict(color=c, linestyle='-',  marker='s')
    for Q, c in zip(WPSD_QS, wpsd_palette):
        style[f'wPSD L={Q}'] = dict(color=c, linestyle='--', marker='^')
    return style


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def _find_pkl(cache_dir: Path, key: str) -> Path:
    """Locate the most-recent merged-runner pickle for a scenario.

    Excludes the rfsd_ prefix which lives in the same directory but has
    a different schema (single test rather than the full method bundle).
    """
    pkls = sorted(
        (p for p in cache_dir.glob(f'{key}_n*_reps*.pkl')
         if not p.name.startswith('rfsd_')),
        key=lambda p: p.stat().st_mtime,
    )
    if not pkls:
        sys.exit(f'ERROR: no pickle found for scenario "{key}" in {cache_dir}')
    return pkls[-1]


def _find_rfsd_pkl(cache_dir: Path, key: str):
    """Optional RFSD cache lookup; returns Path or None if absent."""
    pkls = sorted(cache_dir.glob(f'rfsd_{key}_n*_reps*.pkl'),
                  key=lambda p: p.stat().st_mtime)
    return pkls[-1] if pkls else None


def _load(pkl_path: Path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _series(results, key, Q=None):
    """Return (means, ses) arrays over results['d_list'] for a given series.

    'RFSD' is loaded from a separate cache file; results['rfsd'] (if present)
    is the dict-by-d from rfsd_<scenario>_*.pkl.
    """
    d_list = results['d_list']
    if key == 'RFSD':
        rfsd = results.get('rfsd')
        if rfsd is None:
            return None, None
        means = np.array([rfsd[d]['reject_mean'] for d in d_list])
        ses   = np.array([rfsd[d]['reject_se']   for d in d_list])
        return means, ses
    if key in BASE_TESTS:
        means = np.array([results['by_d'][d]['pop'][key]['reject_mean'] for d in d_list])
        ses   = np.array([results['by_d'][d]['pop'][key]['reject_se']   for d in d_list])
    elif key == 'wpsd':
        means = np.array([results['by_d'][d]['wpsd'][Q]['reject_mean'] for d in d_list])
        ses   = np.array([results['by_d'][d]['wpsd'][Q]['reject_se']   for d in d_list])
    else:
        raise KeyError(key)
    return means, ses


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(panels, out_path, ymax=1.05, with_errorbars=True):
    """panels: list of (title, results_dict)."""
    style = _style_dict()
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    seen = []

    for i, (title, results) in enumerate(panels):
        ax = axes[i]
        d_list = results['d_list']
        alpha  = results.get('alpha', 0.05)

        for tn in PLOT_ORDER:
            if tn in BASE_TESTS:
                means, ses = _series(results, tn)
                if means is None:           # RFSD cache missing
                    continue
            else:
                Q = int(tn.split('=')[1])
                if Q not in results['Q_list']:
                    continue
                means, ses = _series(results, 'wpsd', Q=Q)

            s = style[tn]
            kwargs = dict(
                label=tn, linewidth=2.0, markersize=8,
                markeredgecolor=s['color'], markerfacecolor=s['color'],
                **s,
            )
            if with_errorbars:
                ax.errorbar(d_list, means, yerr=ses, capsize=2.5, **kwargs)
            else:
                ax.plot(d_list, means, **kwargs)
            if tn not in seen:
                seen.append(tn)

        ax.axhline(alpha, linestyle=':', color='grey', linewidth=0.9)
        ax.set_xscale('log')
        ax.set_xticks(d_list)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel('$d$', fontsize=11)
        if i == 0:
            ax.set_ylabel('rejection rate', fontsize=11)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{title}  (n={results['n']})", fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # Shared legend strip
    handles, labels_ = [], []
    for tn in PLOT_ORDER:
        if tn not in seen:
            continue
        s = style[tn]
        handles.append(Line2D(
            [0], [0],
            color=s['color'], linestyle=s['linestyle'], marker=s['marker'],
            linewidth=2.0, markersize=8,
            markeredgecolor=s['color'], markerfacecolor=s['color'],
        ))
        labels_.append(LEGEND_LABELS.get(tn, tn))
    fig.legend(handles, labels_,
               loc='lower center', ncol=len(handles),
               frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', type=str, default='results/merged',
                   help='Directory containing per-scenario pickles.')
    p.add_argument('--null',       type=str, default=None)
    p.add_argument('--varperturb', type=str, default=None)
    p.add_argument('--student',    type=str, default=None)
    p.add_argument('--laplace',    type=str, default=None)
    p.add_argument('--out', type=str,
                   default='results/figure_popular_methods_merged.png')
    p.add_argument('--no_errorbars', action='store_true',
                   help='Plot lines only, suppress error bars.')
    p.add_argument('--ymax', type=float, default=1.05)
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

    panels = []
    for key, label in SCENARIOS:
        path = overrides.get(key)
        path = Path(path) if path else _find_pkl(cache_dir, key)
        print(f'[{label}] using {path}')
        results = _load(path)

        rfsd_path = _find_rfsd_pkl(cache_dir, key)
        if rfsd_path is not None:
            print(f'  + RFSD overlay: {rfsd_path.name}')
            rfsd_results = _load(rfsd_path)
            # Attach the by_d dict; _series consumes it under results['rfsd'].
            # Filter to dims present in the main cache so plot lines align.
            rfsd_by_d = rfsd_results['by_d']
            results['rfsd'] = {d: rfsd_by_d[d]
                               for d in results['d_list']
                               if d in rfsd_by_d}

        panels.append((label, results))

    make_figure(panels, args.out, ymax=args.ymax,
                with_errorbars=not args.no_errorbars)


if __name__ == '__main__':
    main()
