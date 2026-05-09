"""
run_psd_comparison.py
=====================
Compares PSD_L2 (PolynomialSteinTest, Narayan's method) vs
λ-SPSD_L1 (WPSDTest, our method) across dimensions.

Q pairing (from paper — L1 has one fewer moment; bhat is the Q=0 term):
    λ-SPSD_L1  Q = 0, 1, 2, 3
    PSD_L2     Q = 1, 2, 3, 4

Usage
-----
python run_psd_comparison.py --experiment perturbed_gauss \\
    --n 1000 --n_reps 100 --save results/fig_compare.png \\
    --save_csv results/compare.csv --save_latex results/compare.tex
"""

from __future__ import annotations
import argparse
import csv as csv_mod
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kgof.density import IsotropicNormal, Normal
from kgof.data import DSLaplace
from rfsd.goftest import PolynomialSteinTest, WPSDTest
from rfsd.data import DSStudentsT


# ─────────────────────────────────────────────────────────────────────────────
# Q pairing
# ─────────────────────────────────────────────────────────────────────────────

Q_LIST_L1_DEFAULT = [0, 1, 2, 3]
Q_LIST_L2_DEFAULT = [1, 2, 3, 4]

EXPERIMENT_TITLES = {
    "null":            "StandardGaussian",
    "perturbed_gauss": "PerturbedGaussian",
    "gauss_t":         "Student-$t$",
    "laplace":         "Laplace",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def make_case(experiment, n, d, seed):
    """Return (p, dat): p = UnnormalizedDensity, dat = Data object."""
    if experiment == "null":
        p   = IsotropicNormal(np.zeros(d), 1)
        dat = p.get_datasource().sample(n, seed=seed)

    elif experiment == "perturbed_gauss":
        p         = IsotropicNormal(np.zeros(d), 1)
        var       = np.ones(d)
        var[0]   += 0.7
        q         = Normal(np.zeros(d), np.diag(var))
        dat       = q.get_datasource().sample(n, seed=seed)

    elif experiment == "gauss_t":
        df    = 5
        sigma2 = df / (df - 2.0)
        p     = IsotropicNormal(np.zeros(d), sigma2)
        dat   = DSStudentsT(d, df).sample(n, seed=seed)

    elif experiment == "laplace":
        p   = IsotropicNormal(np.zeros(d), 1)
        dat = DSLaplace(d, 0, 1.0 / np.sqrt(2)).sample(n, seed=seed)

    else:
        raise ValueError(f"Unknown experiment: {experiment!r}")

    return p, dat


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_psd_comparison(experiment="perturbed_gauss", n=1000,
                       d_list=None, Q_list_l1=None, Q_list_l2=None,
                       n_reps=100, alpha=0.05,
                       base_seed=36, verbose=True):
    """
    Sweep over (d, Q-pair) combinations, n_reps repetitions each.

    MSNR convention:
      λ-SPSD_L1 test_stat is a z-score  → store log10(z²)
      PSD_L2    test_stat is chi²-like   → store log10(test_stat)

    Returns
    -------
    results_l1 : {Q_l1: {d: {'msnr_mean', 'msnr_std', 'reject_mean', 'reject_std'}}}
    results_l2 : {Q_l2: {d: {'msnr_mean', 'msnr_std', 'reject_mean', 'reject_std'}}}
    """
    if d_list    is None: d_list    = [2, 4, 8, 16, 32, 64]
    if Q_list_l1 is None: Q_list_l1 = Q_LIST_L1_DEFAULT
    if Q_list_l2 is None: Q_list_l2 = Q_LIST_L2_DEFAULT

    results_l1 = {Q: {} for Q in Q_list_l1}
    results_l2 = {Q: {} for Q in Q_list_l2}

    for d in d_list:
        for Q_l1, Q_l2 in zip(Q_list_l1, Q_list_l2):
            if verbose:
                print(f"  d={d:4d}  Q_L1={Q_l1}  Q_L2={Q_l2}", end="", flush=True)
            t0 = time.time()

            l1_msnr = np.zeros(n_reps)
            l2_msnr = np.zeros(n_reps)
            l1_rejs = np.zeros(n_reps)
            l2_rejs = np.zeros(n_reps)

            for rep in range(n_reps):
                seed = base_seed + d * 10_000 + Q_l1 * 1_000 + rep
                p, dat = make_case(experiment, n, d, seed)

                r_l1 = WPSDTest(p, Q=Q_l1, alpha=alpha, seed=seed,
                                method='tpsd').perform_test(dat)
                r_l2 = PolynomialSteinTest(p, polyorder=Q_l2, alpha=alpha,
                                           seed=seed,
                                           bootstrap=True).perform_test(dat)

                z  = r_l1['test_stat']
                ts = r_l2['test_stat']

                l1_msnr[rep] = np.log10(max(z  ** 2, 1e-12))
                l2_msnr[rep] = np.log10(max(ts,      1e-12))
                l1_rejs[rep] = float(r_l1['h0_rejected'])
                l2_rejs[rep] = float(r_l2['h0_rejected'])

            def _agg(msnr, rejs):
                return {
                    'msnr_mean':   float(np.mean(msnr)),
                    'msnr_std':    float(np.std(msnr,  ddof=1) / np.sqrt(n_reps)),
                    'reject_mean': float(np.mean(rejs)),
                    'reject_std':  float(np.std(rejs, ddof=1) / np.sqrt(n_reps)),
                }

            results_l1[Q_l1][d] = _agg(l1_msnr, l1_rejs)
            results_l2[Q_l2][d] = _agg(l2_msnr, l2_rejs)

            elapsed = time.time() - t0
            if verbose:
                rl1 = results_l1[Q_l1][d]
                rl2 = results_l2[Q_l2][d]
                print(f"  [{elapsed:.1f}s]"
                      f"  L1 rej={rl1['reject_mean']:.3f}"
                      f"  L2 rej={rl2['reject_mean']:.3f}")

    return results_l1, results_l2


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_psd_comparison(results_l1, results_l2,
                        Q_list_l1, Q_list_l2, d_list,
                        experiment="perturbed_gauss",
                        n=1000, alpha=0.05, save_path=None):
    """
    Two-panel plot.
    Solid  = λ-SPSD_L1 (our method).
    Dashed = PSD_L2 (Narayan's baseline).
    One colour per Q pair.
    """
    title_str = EXPERIMENT_TITLES.get(experiment, experiment)
    colors    = plt.cm.viridis(np.linspace(0.1, 0.85, len(Q_list_l1)))
    ds        = np.array(d_list)
    log10_ref = np.log10(2)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.45)

    for qi, (Q_l1, Q_l2) in enumerate(zip(Q_list_l1, Q_list_l2)):
        col    = colors[qi]
        lbl_l1 = rf"$Q={Q_l1}$  $\lambda$-SPSD$_{{L_1}}$"
        lbl_l2 = rf"$Q={Q_l2}$  PSD$_{{L_2}}$"

        l1_msnr   = np.array([results_l1[Q_l1][d]['msnr_mean']   for d in d_list])
        l1_msnr_e = np.array([results_l1[Q_l1][d]['msnr_std']    for d in d_list])
        l2_msnr   = np.array([results_l2[Q_l2][d]['msnr_mean']   for d in d_list])
        l2_msnr_e = np.array([results_l2[Q_l2][d]['msnr_std']    for d in d_list])

        l1_rej    = np.array([results_l1[Q_l1][d]['reject_mean'] for d in d_list])
        l1_rej_e  = np.array([results_l1[Q_l1][d]['reject_std']  for d in d_list])
        l2_rej    = np.array([results_l2[Q_l2][d]['reject_mean'] for d in d_list])
        l2_rej_e  = np.array([results_l2[Q_l2][d]['reject_std']  for d in d_list])

        kw_l1 = dict(color=col, linestyle="-",  marker="o", markersize=4,
                     linewidth=1.4, capsize=3, label=lbl_l1)
        kw_l2 = dict(color=col, linestyle="--", marker="^", markersize=4,
                     linewidth=1.4, capsize=3, label=lbl_l2)

        axes[0].errorbar(ds, l1_msnr, yerr=l1_msnr_e, **kw_l1)
        axes[0].errorbar(ds, l2_msnr, yerr=l2_msnr_e, **kw_l2)
        axes[1].errorbar(ds, l1_rej,  yerr=l1_rej_e,  **kw_l1)
        axes[1].errorbar(ds, l2_rej,  yerr=l2_rej_e,  **kw_l2)

    ax = axes[0]
    ax.axhline(log10_ref, linestyle=":", color="grey", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks(d_list)
    ax.set_xticklabels([str(d) for d in d_list], rotation=45, ha="right")
    ax.set_xlabel("$d$", fontsize=11)
    ax.set_ylabel("log10(MSNR)", fontsize=11)
    ax.set_title(f"{title_str}: log10(MSNR) vs $d$  (n={n})", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2, handlelength=2.5)

    ax = axes[1]
    ax.axhline(alpha, linestyle=":", color="grey", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks(d_list)
    ax.set_xticklabels([str(d) for d in d_list], rotation=45, ha="right")
    ax.set_xlabel("$d$", fontsize=11)
    ax.set_ylabel("Rejection rate", fontsize=11)
    ax.set_title(f"{title_str}: Rejection rate vs $d$  (n={n})", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2, handlelength=2.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Table export
# ─────────────────────────────────────────────────────────────────────────────

def save_results_table(results_l1, results_l2,
                       Q_list_l1, Q_list_l2, d_list,
                       experiment, n, alpha,
                       csv_path=None, latex_path=None):
    """
    Save rejection rate and log10(MSNR) for both methods.
    CSV: one row per (d, Q-pair).
    LaTeX: two booktabs tables (rejection rate + log10 MSNR),
           rows = d, columns = Q-pairs × (L1, L2).
    """
    title = EXPERIMENT_TITLES.get(experiment, experiment)

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv_mod.writer(f)
            header = ['d', 'Q_l1', 'Q_l2',
                      'l1_rej', 'l1_rej_se', 'l2_rej', 'l2_rej_se',
                      'l1_msnr', 'l1_msnr_se', 'l2_msnr', 'l2_msnr_se']
            w.writerow(header)
            for d in d_list:
                for Q_l1, Q_l2 in zip(Q_list_l1, Q_list_l2):
                    rl1 = results_l1[Q_l1][d]
                    rl2 = results_l2[Q_l2][d]
                    w.writerow([d, Q_l1, Q_l2,
                                rl1['reject_mean'], rl1['reject_std'],
                                rl2['reject_mean'], rl2['reject_std'],
                                rl1['msnr_mean'],   rl1['msnr_std'],
                                rl2['msnr_mean'],   rl2['msnr_std']])
        print(f"CSV   → {csv_path}")

    if latex_path:
        def _bold(s): return r'\textbf{' + s + '}'
        n_pairs  = len(Q_list_l1)
        col_spec = 'r' + ' rr' * n_pairs

        def _make_table(metric_key, caption_metric, fmt):
            lines = [
                r'\begin{table}[ht]',
                r'\centering',
                (r'\caption{' + caption_metric + r' vs $d$: '
                 r'$\lambda$-SPSD$_{L_1}$ vs PSD$_{L_2}$ --- ' + title +
                 rf' ($n={n}$, $\alpha={alpha}$)' + r'}'),
                r'\begin{tabular}{' + col_spec + r'}',
                r'\toprule',
            ]
            # Q-pair spanning headers
            q_header = '$d$'
            for Q_l1, Q_l2 in zip(Q_list_l1, Q_list_l2):
                q_header += (r' & \multicolumn{2}{c}{$Q_{L_1}=' + str(Q_l1) +
                             r',\,Q_{L_2}=' + str(Q_l2) + r'$}')
            lines.append(q_header + r' \\')

            cmidrules = ''
            for i in range(n_pairs):
                s = 2 + 2 * i
                cmidrules += r'\cmidrule(lr){' + f'{s}-{s+1}' + '}'
            lines.append(cmidrules)

            sub = ''
            for _ in range(n_pairs):
                sub += r' & $\lambda$-SPSD$_{L_1}$ & PSD$_{L_2}$'
            lines += [sub + r' \\', r'\midrule']

            for d in d_list:
                row_str = str(d)
                for Q_l1, Q_l2 in zip(Q_list_l1, Q_list_l2):
                    rl1 = results_l1[Q_l1][d]
                    rl2 = results_l2[Q_l2][d]
                    v1  = rl1[metric_key]
                    v2  = rl2[metric_key]
                    s1  = fmt.format(v1)
                    s2  = fmt.format(v2)
                    if   v1 > v2: s1 = _bold(s1)
                    elif v2 > v1: s2 = _bold(s2)
                    row_str += f' & {s1} & {s2}'
                lines.append(row_str + r' \\')

            lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
            return '\n'.join(lines) + '\n'

        rej_table  = _make_table('reject_mean', 'Rejection rate', '{:.3f}')
        msnr_table = _make_table('msnr_mean',   r'$\log_{10}$(MSNR)', '{:.3f}')

        os.makedirs(os.path.dirname(os.path.abspath(latex_path)), exist_ok=True)
        with open(latex_path, 'w') as f:
            f.write(rej_table + '\n' + msnr_table)
        print(f"LaTeX → {latex_path}  (2 tables: rejection rate + log10 MSNR)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="PSD_L2 vs λ-SPSD_L1 comparison across dimensions")
    p.add_argument("--experiment", default="perturbed_gauss",
                   choices=list(EXPERIMENT_TITLES))
    p.add_argument("--n",      type=int,   default=1000)
    p.add_argument("--n_reps", type=int,   default=100)
    p.add_argument("--alpha",  type=float, default=0.05)
    p.add_argument("--d_list", type=str,   default="2,4,8,16,32,64",
                   help="Comma-separated d values")
    p.add_argument("--q_list_l1", type=str, default="0,1,2,3",
                   help="Comma-separated Q values for λ-SPSD_L1")
    p.add_argument("--q_list_l2", type=str, default="1,2,3,4",
                   help="Comma-separated Q values for PSD_L2 (paired with q_list_l1)")
    p.add_argument("--seed",       type=int, default=36)
    p.add_argument("--save",       type=str, default=None)
    p.add_argument("--save_csv",   type=str, default=None)
    p.add_argument("--save_latex", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args      = _parse_args()
    d_list    = [int(x) for x in args.d_list.split(",")]
    Q_list_l1 = [int(x) for x in args.q_list_l1.split(",")]
    Q_list_l2 = [int(x) for x in args.q_list_l2.split(",")]

    if len(Q_list_l1) != len(Q_list_l2):
        raise ValueError("q_list_l1 and q_list_l2 must have the same length")

    print(f"Experiment : {args.experiment}")
    print(f"n={args.n}, reps={args.n_reps}, alpha={args.alpha}, seed={args.seed}")
    print(f"d values   : {d_list}")
    print(f"Q pairs    : L1={Q_list_l1}  ↔  L2={Q_list_l2}")
    print()

    results_l1, results_l2 = run_psd_comparison(
        experiment=args.experiment,
        n=args.n,
        d_list=d_list,
        Q_list_l1=Q_list_l1,
        Q_list_l2=Q_list_l2,
        n_reps=args.n_reps,
        alpha=args.alpha,
        base_seed=args.seed,
        verbose=True,
    )

    plot_psd_comparison(
        results_l1, results_l2,
        Q_list_l1, Q_list_l2, d_list,
        experiment=args.experiment,
        n=args.n,
        alpha=args.alpha,
        save_path=args.save,
    )

    if args.save_csv or args.save_latex:
        save_results_table(
            results_l1, results_l2,
            Q_list_l1, Q_list_l2, d_list,
            experiment=args.experiment,
            n=args.n, alpha=args.alpha,
            csv_path=args.save_csv,
            latex_path=args.save_latex,
        )
