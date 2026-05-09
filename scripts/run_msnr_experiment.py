"""
run_msnr_experiment.py
======================
Reproduces the two-panel plot:
    - MSNR vs Q           (top)
    - Rejection rate vs Q (bottom)

for PSD_L1 (uniform lambda) and λ-SPSD_L1 (CE-optimised lambda = wPSD/tPSD),
under a given null/alternative distribution.

Usage
-----
# Null experiment (StandardGaussian):
python run_msnr_experiment.py --experiment null --n 1000 --d 16 --n_reps 100

# Gaussian mean-shift alternative:
python run_msnr_experiment.py --experiment gauss_mean --n 1000 --d 16 --n_reps 100

Standalone — only requires numpy, scipy, matplotlib, and wpsd.py in the same
directory (or on PYTHONPATH).  No kgof dependency needed.

MSNR definition
---------------
For each repetition, record z = √n · mean(u) / sd(u).
MSNR per rep = z²  (equals 1 in expectation under H0 for a correctly-sized test;
larger under H1).  Report mean and 1-SD error bars across reps.
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── import your math module ───────────────────────────────────────────────────
try:
    from wpsd import tpsd, psd
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from rfsd.wpsd import tpsd, psd


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def _standard_gaussian(n, d, rng):
    """H0: X ~ N(0, I_d),  score s(x) = -x."""
    X = rng.standard_normal((n, d))
    S = -X
    return X, S


def _gauss_mean(n, d, rng, shift=0.3):
    """H1: X ~ N(shift·e1, I_d),  model score s(x) = -x  (mismatch)."""
    X = rng.standard_normal((n, d))
    X[:, 0] += shift
    S = -X          # score of N(0, I) evaluated at shifted samples
    return X, S


def _gauss_variance(n, d, rng, extra_var=0.7):
    """H1: X ~ N(0, diag(1+extra_var, 1,...,1)),  model score s(x) = -x."""
    X = rng.standard_normal((n, d))
    X[:, 0] *= np.sqrt(1 + extra_var)
    S = -X
    return X, S


def _gauss_t(n, d, rng, df=5):
    """H1: X from Student-t_df(0, sigma²·I) where sigma² = df/(df-2)."""
    if df > 2:
        sigma2 = df / (df - 2.0)
    else:
        sigma2 = 1.0
    # sample from t, score from N(0, sigma²·I)
    X = rng.standard_t(df, size=(n, d))
    S = -X / sigma2
    return X, S


def _perturbed_gaussian(n, d, rng, extra_var=0.7):
    """
    PerturbedGaussian: X ~ N(0, diag(1+extra_var, 1,...,1))
    Model score: s(x) = -x  (N(0,I) — variance mismatch in first coordinate)
    Q=0 is blind here (mean score = 0). Signal lives in polynomial terms only.
    """
    X = rng.standard_normal((n, d))
    X[:, 0] *= np.sqrt(1 + extra_var)
    S = -X
    return X, S


def _laplace(n, d, rng):
    """
    Laplace: X ~ Laplace(0, 1/sqrt(2)) per coordinate
    scale=1/sqrt(2) makes Var(X)=1, matching N(0,1).
    Model score: s(x) = -x  (N(0,I) — tail mismatch)
    """
    X = rng.laplace(loc=0.0, scale=1.0/np.sqrt(2), size=(n, d))
    S = -X
    return X, S


GENERATORS = {
    "null":            _standard_gaussian,
    "gauss_mean":      _gauss_mean,
    "gauss_variance":  _gauss_variance,
    "gauss_t":         _gauss_t,
    "perturbed_gauss": _perturbed_gaussian,
    "laplace":         _laplace,
}

EXPERIMENT_TITLES = {
    "null":            "StandardGaussian",
    "gauss_mean":      "Gaussian Mean-Shift",
    "gauss_variance":  "Gaussian Variance-Shift",
    "gauss_t":         "Gaussian vs Student-t",
    "perturbed_gauss": "PerturbedGaussian",
    "laplace":         "Laplace",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-repetition runner
# ─────────────────────────────────────────────────────────────────────────────

def _extract_stat(r):
    """
    Pull z-statistic and p-value out of a wpsd result dict.

    Handles two interface variants:
      - Flat keys: {'statistic': z, 'p_value': p, ...}          (wpsd.py v1)
      - Nested keys: {'studentised': {'statistic': z, ...}, ...} (wpsd.py v2)
        or           {'unweighted':  {'statistic': z, ...}, ...}
    """
    if 'statistic' in r:
        return float(r['statistic']), float(r['p_value'])
    for key in ('studentised', 'unweighted'):
        if key in r and isinstance(r[key], dict):
            return float(r[key]['statistic']), float(r[key]['p_value'])
    raise KeyError(f"Cannot find statistic in result dict with keys: {list(r.keys())}")


def _run_one_rep(gen, n, d, Q, alpha, seed):
    """
    Run one repetition for a given Q.

    Returns
    -------
    dict with keys:
        psd_z, psd_rejected,
        wpsd_z, wpsd_rejected
    """
    rng = np.random.default_rng(seed)
    X, S = gen(n, d, rng)

    r_psd  = psd( X, S, Q=Q, seed=seed)
    r_wpsd = tpsd(X, S, Q=Q, seed=seed)

    z_psd,  p_psd  = _extract_stat(r_psd)
    z_wpsd, p_wpsd = _extract_stat(r_wpsd)

    return {
        "psd_z":         z_psd,
        "psd_rejected":  float(p_psd  < alpha),
        "wpsd_z":        z_wpsd,
        "wpsd_rejected": float(p_wpsd < alpha),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(experiment="null", n=1000, d=16,
                   Q_list=None, n_reps=100, alpha=0.05,
                   base_seed=0, verbose=True):
    """
    Sweep over Q values, collecting MSNR and rejection rate for both methods.

    Returns
    -------
    results : dict  Q -> {'psd': {...}, 'wpsd': {...}}
        Each inner dict has keys: msnr_mean, msnr_std, reject_mean, reject_std
    """
    if Q_list is None:
        Q_list = list(range(9))   # 0..8 matching the target plot

    gen = GENERATORS[experiment]
    results = {}

    for Q in Q_list:
        if verbose:
            print(f"  Q={Q}", end="", flush=True)
        t0 = time.time()

        psd_z2s   = np.zeros(n_reps)
        wpsd_z2s  = np.zeros(n_reps)
        psd_rejs  = np.zeros(n_reps)
        wpsd_rejs = np.zeros(n_reps)

        for rep in range(n_reps):
            seed = base_seed + Q * 10_000 + rep
            r = _run_one_rep(gen, n, d, Q, alpha, seed)

            # MSNR = z²  (expected value 1 under H0, larger under H1)
            psd_z2s [rep] = r["psd_z"]  ** 2
            wpsd_z2s[rep] = r["wpsd_z"] ** 2
            psd_rejs [rep] = r["psd_rejected"]
            wpsd_rejs[rep] = r["wpsd_rejected"]

        results[Q] = {
            "psd":  {
                "msnr_mean":   float(np.mean(psd_z2s)),
                "msnr_std":    float(np.std(psd_z2s,  ddof=1) / np.sqrt(n_reps)),
                "reject_mean": float(np.mean(psd_rejs)),
                "reject_std":  float(np.std(psd_rejs, ddof=1) / np.sqrt(n_reps)),
            },
            "wpsd": {
                "msnr_mean":   float(np.mean(wpsd_z2s)),
                "msnr_std":    float(np.std(wpsd_z2s,  ddof=1) / np.sqrt(n_reps)),
                "reject_mean": float(np.mean(wpsd_rejs)),
                "reject_std":  float(np.std(wpsd_rejs, ddof=1) / np.sqrt(n_reps)),
            },
        }

        elapsed = time.time() - t0
        if verbose:
            r_psd  = results[Q]["psd"]
            r_wpsd = results[Q]["wpsd"]
            print(f"  [{elapsed:.1f}s]  "
                  f"PSD  MSNR={r_psd['msnr_mean']:.3f}  rej={r_psd['reject_mean']:.3f}  |  "
                  f"wPSD MSNR={r_wpsd['msnr_mean']:.3f}  rej={r_wpsd['reject_mean']:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, experiment="null", n=1000, d=16,
                 alpha=0.05, save_path=None, msnr_ref=2.0):
    """
    Two-panel plot: MSNR (top) and Rejection rate (bottom) vs Q.
    """
    Q_list = sorted(results.keys())
    Qs     = np.array(Q_list)

    # ── extract series ───────────────────────────────────────────────
    psd_msnr_m  = np.array([results[Q]["psd"]["msnr_mean"]   for Q in Q_list])
    psd_msnr_s  = np.array([results[Q]["psd"]["msnr_std"]    for Q in Q_list])
    psd_rej_m   = np.array([results[Q]["psd"]["reject_mean"] for Q in Q_list])
    psd_rej_s   = np.array([results[Q]["psd"]["reject_std"]  for Q in Q_list])

    wpsd_msnr_m = np.array([results[Q]["wpsd"]["msnr_mean"]   for Q in Q_list])
    wpsd_msnr_s = np.array([results[Q]["wpsd"]["msnr_std"]    for Q in Q_list])
    wpsd_rej_m  = np.array([results[Q]["wpsd"]["reject_mean"] for Q in Q_list])
    wpsd_rej_s  = np.array([results[Q]["wpsd"]["reject_std"]  for Q in Q_list])

    title_str = EXPERIMENT_TITLES.get(experiment, experiment)

    # ── figure setup ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    fig.subplots_adjust(hspace=0.45)

    label_psd  = r"$\mathrm{PSD}_{L_1}$"
    label_wpsd = r"$\lambda\text{-}\mathrm{SPSD}_{L_1}$"

    kw_psd  = dict(marker="o", markersize=5, linewidth=1.4,
                   color="black", linestyle="-",  label=label_psd)
    kw_wpsd = dict(marker="^", markersize=5, linewidth=1.4,
                   color="black", linestyle="--", label=label_wpsd)

    # ── panel 1: MSNR ────────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(Qs, psd_msnr_m,  yerr=psd_msnr_s,  capsize=3, **kw_psd)
    ax.errorbar(Qs, wpsd_msnr_m, yerr=wpsd_msnr_s, capsize=3, **kw_wpsd)

    ax.axhline(msnr_ref, linestyle=":", color="black", linewidth=0.9)
    ax.set_xlabel("Q", fontsize=11)
    ax.set_ylabel("MSNR", fontsize=11)
    ax.set_title(f"{title_str}: MSNR vs Q (n = {n}, d = {d})", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=9,
              frameon=False, handlelength=2.5,
              title=r"Method", title_fontsize=9)
    ylo = max(0.0, min(psd_msnr_m.min(), wpsd_msnr_m.min()) - 0.3)
    yhi = max(psd_msnr_m.max(), wpsd_msnr_m.max()) + 0.3
    ax.set_ylim(ylo, yhi)

    # ── panel 2: rejection rate ───────────────────────────────────────
    ax = axes[1]
    ax.errorbar(Qs, psd_rej_m,  yerr=psd_rej_s,  capsize=3, **kw_psd)
    ax.errorbar(Qs, wpsd_rej_m, yerr=wpsd_rej_s, capsize=3, **kw_wpsd)

    ax.axhline(alpha, linestyle=":", color="black", linewidth=0.9)
    ax.set_xlabel("Q", fontsize=11)
    ax.set_ylabel("Rejection rate", fontsize=11)
    ax.set_title(f"{title_str}: Rejection rate vs Q (n = {n}, d = {d})", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(0, min(1.0, max(psd_rej_m.max(), wpsd_rej_m.max()) + 0.15))
    ax.legend(loc="upper right", fontsize=9, frameon=False,
              handlelength=2.5, title="Method", title_fontsize=9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Table export
# ─────────────────────────────────────────────────────────────────────────────

def save_results_table(results, experiment, n, d, alpha,
                       csv_path=None, latex_path=None):
    """Save results as CSV and/or a LaTeX booktabs table."""
    import csv as csv_mod
    Q_list = sorted(results.keys())
    title  = EXPERIMENT_TITLES.get(experiment, experiment)

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csv_mod.writer(f)
            w.writerow(['Q',
                        'psd_msnr_mean', 'psd_msnr_se',
                        'psd_rej_mean',  'psd_rej_se',
                        'wpsd_msnr_mean','wpsd_msnr_se',
                        'wpsd_rej_mean', 'wpsd_rej_se'])
            for Q in Q_list:
                rp = results[Q]['psd']
                rw = results[Q]['wpsd']
                w.writerow([Q,
                            rp['msnr_mean'],   rp['msnr_std'],
                            rp['reject_mean'], rp['reject_std'],
                            rw['msnr_mean'],   rw['msnr_std'],
                            rw['reject_mean'], rw['reject_std']])
        print(f"CSV   → {csv_path}")

    if latex_path:
        def _bold(s): return r'\textbf{' + s + '}'

        lines = [
            r'\begin{table}[ht]',
            r'\centering',
            (r'\caption{Rejection rate and MSNR vs $Q$ --- ' + title +
             rf' ($n={n}$, $d={d}$, $\alpha={alpha}$)' + r'}'),
            r'\begin{tabular}{r rr rr}',
            r'\toprule',
            (r' & \multicolumn{2}{c}{Rejection rate}'
             r' & \multicolumn{2}{c}{MSNR} \\'),
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
            (r'$Q$ & $\mathrm{PSD}_{L_1}$'
             r' & $\lambda\text{-SPSD}_{L_1}$'
             r' & $\mathrm{PSD}_{L_1}$'
             r' & $\lambda\text{-SPSD}_{L_1}$ \\'),
            r'\midrule',
        ]

        for Q in Q_list:
            rp = results[Q]['psd']
            rw = results[Q]['wpsd']

            ps = f"{rp['reject_mean']:.3f} $\\pm$ {rp['reject_std']:.3f}"
            ws = f"{rw['reject_mean']:.3f} $\\pm$ {rw['reject_std']:.3f}"
            if   rw['reject_mean'] > rp['reject_mean']: ws = _bold(ws)
            elif rp['reject_mean'] > rw['reject_mean']: ps = _bold(ps)

            pm = f"{rp['msnr_mean']:.2f} $\\pm$ {rp['msnr_std']:.2f}"
            wm = f"{rw['msnr_mean']:.2f} $\\pm$ {rw['msnr_std']:.2f}"
            if   rw['msnr_mean'] > rp['msnr_mean']: wm = _bold(wm)
            elif rp['msnr_mean'] > rw['msnr_mean']: pm = _bold(pm)

            lines.append(rf'{Q} & {ps} & {ws} & {pm} & {wm} \\')

        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(latex_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"LaTeX → {latex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="MSNR / rejection-rate vs Q sweep")
    p.add_argument("--experiment", default="null",
                   choices=list(GENERATORS),
                   help="Which experiment to run")
    p.add_argument("--n",      type=int,   default=1000)
    p.add_argument("--d",      type=int,   default=16)
    p.add_argument("--n_reps", type=int,   default=100,
                   help="Repetitions per Q value")
    p.add_argument("--alpha",  type=float, default=0.05)
    p.add_argument("--q_max",  type=int,   default=8,
                   help="Q swept from 0 to q_max (inclusive)")
    p.add_argument("--q_list", type=str,   default=None,
                   help="Override q_max with explicit comma-sep list, e.g. 0,1,2,4,8,16")
    p.add_argument("--seed",   type=int,   default=36)
    p.add_argument("--save",   type=str,   default=None,
                   help="Path to save figure (e.g. results.png). "
                        "If omitted, figure is shown interactively.")
    p.add_argument("--save_csv",   type=str, default=None,
                   help="Path to save results as CSV (e.g. results.csv)")
    p.add_argument("--save_latex", type=str, default=None,
                   help="Path to save results as LaTeX table (e.g. table.tex)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.q_list is not None:
        Q_list = [int(q) for q in args.q_list.split(",")]
    else:
        Q_list = list(range(args.q_max + 1))

    print(f"Experiment : {args.experiment}")
    print(f"n={args.n}, d={args.d}, reps={args.n_reps}, alpha={args.alpha}")
    print(f"Q values   : {Q_list}")
    print()

    results = run_experiment(
        experiment=args.experiment,
        n=args.n, d=args.d,
        Q_list=Q_list,
        n_reps=args.n_reps,
        alpha=args.alpha,
        base_seed=args.seed,
        verbose=True,
    )

    plot_results(
        results,
        experiment=args.experiment,
        n=args.n, d=args.d,
        alpha=args.alpha,
        save_path=args.save,
    )

    if args.save_csv or args.save_latex:
        save_results_table(
            results,
            experiment=args.experiment,
            n=args.n, d=args.d, alpha=args.alpha,
            csv_path=args.save_csv,
            latex_path=args.save_latex,
        )
