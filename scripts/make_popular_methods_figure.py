"""
make_popular_methods_figure.py
==============================
Build the final 1xN combined figure (panels + shared legend) for the
"popular methods" comparison directly from cached results.pck/params.pck.
Does NOT re-run experiments and does NOT touch the existing plotter.

Styling:
  wPSD L=*  → dashed line, triangle marker (viridis colour gradient)
  others    → solid  line, square   marker (tab10 palette)

Usage:
  python scripts/make_popular_methods_figure.py \
      --null    "results/goftest-null-...-stored-data" \
      --varperturb "results/goftest-variance_perturb-...-stored-data" \
      --student "results/goftest-student-t-...-stored-data" \
      --laplace "results/goftest-laplace-...-stored-data" \
      --out results/figure_popular_methods.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfsd.util import restore_object


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

BASE_TESTS = ['IMQ KSD', 'Gauss KSD', 'Gauss FSSD-opt']
WPSD_TESTS = ['wPSD L=0', 'wPSD L=1', 'wPSD L=2', 'wPSD L=3', 'wPSD L=4']
PLOT_ORDER = BASE_TESTS + WPSD_TESTS

LEGEND_LABELS = {
    'IMQ KSD':        'IMQ KSD',
    'Gauss KSD':      'Gauss KSD',
    'Gauss FSSD-opt': 'Gauss FSSD-opt',
    'wPSD L=0': r'$\lambda$-PSD$_{L_1}$ ($Q=0$)',
    'wPSD L=1': r'$\lambda$-PSD$_{L_1}$ ($Q=1$)',
    'wPSD L=2': r'$\lambda$-PSD$_{L_1}$ ($Q=2$)',
    'wPSD L=3': r'$\lambda$-PSD$_{L_1}$ ($Q=3$)',
    'wPSD L=4': r'$\lambda$-PSD$_{L_1}$ ($Q=4$)',
}


def _style_dict():
    base_palette = plt.cm.tab10(np.linspace(0, 1, 10))[:len(BASE_TESTS)]
    wpsd_palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(WPSD_TESTS)))
    style = {}
    for tn, c in zip(BASE_TESTS, base_palette):
        style[tn] = dict(color=c, linestyle='-',  marker='s')
    for tn, c in zip(WPSD_TESTS, wpsd_palette):
        style[tn] = dict(color=c, linestyle='--', marker='^')
    return style


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load(stored_dir: Path):
    results = restore_object(str(stored_dir), 'results')
    params  = restore_object(str(stored_dir), 'params')
    # results is {J: {test_name: [rate_per_dim]}}; pick the only J present
    J = next(iter(results.keys()))
    rates = results[J]
    dims  = list(params['variable_values'])
    alpha = params.get('test_alpha', 0.05)
    return dims, rates, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(panels, out_path, ymax_per_panel=None):
    """
    panels: list of (title, dims, rates, alpha)
    """
    style = _style_dict()
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    seen_tests = []

    for i, (title, dims, rates, alpha) in enumerate(panels):
        ax = axes[i]
        ymax = (ymax_per_panel or {}).get(title, 1.05)

        for tn in PLOT_ORDER:
            if tn not in rates:
                continue
            s = style[tn]
            ax.plot(
                dims, np.array(rates[tn]),
                label=tn,
                linewidth=2.0, markersize=8,
                markeredgecolor=s['color'], markerfacecolor=s['color'],
                **s,
            )
            if tn not in seen_tests:
                seen_tests.append(tn)

        ax.axhline(alpha, linestyle=':', color='grey', linewidth=0.9)
        ax.set_xscale('log')
        ax.set_xticks(dims)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel('$d$', fontsize=11)
        if i == 0:
            ax.set_ylabel('rejection rate', fontsize=11)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # Shared legend strip below — build explicit Line2D handles so the
    # wPSD dashed style is preserved in the legend.
    ordered_handles = []
    ordered_labels = []
    for tn in PLOT_ORDER:
        if tn not in seen_tests:
            continue
        s = style[tn]
        ordered_handles.append(Line2D(
            [0], [0],
            color=s['color'], linestyle=s['linestyle'], marker=s['marker'],
            linewidth=2.0, markersize=8,
            markeredgecolor=s['color'], markerfacecolor=s['color'],
        ))
        ordered_labels.append(LEGEND_LABELS.get(tn, tn))
    ncol = len(ordered_handles)
    fig.legend(ordered_handles, ordered_labels,
               loc='lower center', ncol=ncol,
               frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--null',       required=True, help='stored-data dir for null')
    p.add_argument('--varperturb', required=True, help='stored-data dir for variance_perturb')
    p.add_argument('--student',    required=True, help='stored-data dir for student-t')
    p.add_argument('--laplace',    required=True, help='stored-data dir for laplace')
    p.add_argument('--out', default='results/figure_popular_methods.png')
    return p.parse_args()


def main():
    args = _parse()

    panels = []
    for label, key in [
        ('Standard Gaussian',   'null'),
        ('Perturbed Gaussian',  'varperturb'),
        ('Student-t',           'student'),
        ('Laplace',             'laplace'),
    ]:
        d = Path(getattr(args, key))
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")
        dims, rates, alpha = _load(d)
        panels.append((label, dims, rates, alpha))

    make_figure(panels, args.out)


if __name__ == '__main__':
    main()
