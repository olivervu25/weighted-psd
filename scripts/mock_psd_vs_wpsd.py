"""
Mock figure: wPSD vs plain PSD across d in {2,4,8,16,32,64,128} and Q in {0..4}.
FAKE DATA — for discussion / draft only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


DIMS = np.array([2, 4, 8, 16, 32, 64, 128])
QS = [0, 1, 2, 3, 4]
ALPHA = 0.05
RNG = np.random.default_rng(20260506)

PANELS = ['Standard Gaussian', 'Perturbed Gaussian', 'Student-t', 'Laplace']


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _jitter(shape, scale=0.015):
    return RNG.normal(0, scale, size=shape)


def fake_curve(panel, Q, method):
    """Plausible-looking rejection rate vs d for (panel, Q, method)."""
    log_d = np.log2(DIMS)
    bonus_Q = 0.10 * Q
    wpsd_advantage = 0.18 if method == 'wpsd' else 0.0

    if panel == 'Standard Gaussian':
        base = np.full_like(DIMS, ALPHA, dtype=float)
        return np.clip(base + _jitter(DIMS.shape, 0.012), 0, 1)

    if panel == 'Perturbed Gaussian':
        x0 = 4.5 - 0.4 * Q - (0.6 if method == 'wpsd' else 0.0)
        k = 0.9 + 0.1 * Q + (0.2 if method == 'wpsd' else 0.0)
        y = _logistic(log_d, x0, k)
        y *= 0.92 + 0.04 * (method == 'wpsd')
        return np.clip(y + _jitter(DIMS.shape), 0, 1)

    if panel == 'Student-t':
        x0 = 5.5 - 0.3 * Q - (1.0 if method == 'wpsd' else 0.0)
        k = 0.7 + 0.08 * Q + (0.25 if method == 'wpsd' else 0.0)
        y = _logistic(log_d, x0, k)
        y *= 0.85 + 0.08 * (method == 'wpsd') + 0.02 * Q
        return np.clip(y + _jitter(DIMS.shape), 0, 1)

    if panel == 'Laplace':
        x0 = 4.0 - 0.5 * Q - (0.8 if method == 'wpsd' else 0.0)
        k = 0.85 + 0.1 * Q + (0.2 if method == 'wpsd' else 0.0)
        y = _logistic(log_d, x0, k)
        y = np.clip(y + bonus_Q * 0.05 + wpsd_advantage * 0.05, 0, 1)
        return np.clip(y + _jitter(DIMS.shape), 0, 1)

    raise ValueError(panel)


def main():
    q_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(QS)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)

    for ax, panel in zip(axes, PANELS):
        ymax = 0.2 if panel == 'Standard Gaussian' else 1.05

        for q, color in zip(QS, q_colors):
            psd  = fake_curve(panel, q, 'psd')
            wpsd = fake_curve(panel, q, 'wpsd')
            ax.plot(DIMS, psd,  color=color, linestyle='-',  marker='s',
                    linewidth=2.0, markersize=7,
                    markeredgecolor=color, markerfacecolor=color)
            ax.plot(DIMS, wpsd, color=color, linestyle='--', marker='^',
                    linewidth=2.0, markersize=7,
                    markeredgecolor=color, markerfacecolor=color)

        ax.axhline(ALPHA, linestyle=':', color='grey', linewidth=0.9)
        ax.set_xscale('log')
        ax.set_xticks(DIMS)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel('dimension $d$', fontsize=11)
        ax.set_title(panel, fontsize=13)
        ax.set_ylim(0, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel('rejection rate', fontsize=11)

    q_handles = [
        Line2D([0], [0], color=c, linestyle='-', linewidth=2.0,
               marker='s', markersize=7, markerfacecolor=c, markeredgecolor=c,
               label=f'$Q={q}$')
        for q, c in zip(QS, q_colors)
    ]
    method_handles = [
        Line2D([0], [0], color='black', linestyle='-',  linewidth=2.0,
               marker='s', markersize=7, label='PSD (uniform $\\lambda$)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2.0,
               marker='^', markersize=7, label=r'$\lambda$-PSD (adaptive)'),
    ]

    fig.legend(handles=q_handles, loc='lower center', ncol=len(QS),
               frameon=False, fontsize=10, bbox_to_anchor=(0.35, -0.04))
    fig.legend(handles=method_handles, loc='lower center', ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.78, -0.04))

    fig.suptitle('MOCK / DRAFT — fake data for layout review',
                 fontsize=11, color='crimson', y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = Path('/Users/olivervu25/PSD/results/mock_psd_vs_wpsd.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
