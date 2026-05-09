"""
run_dim_experiment.py
=====================
Sweeps d ∈ d_list for multiple fixed Q values.
Produces Figure 4-style plot:
    - log10(MSNR) vs d   (top)
    - Rejection rate vs d (bottom)

Each Q gets one colour; solid line = PSD, dashed line = λ-SPSD.

Usage
-----
python run_dim_experiment.py --experiment perturbed_gauss \\
    --q_list 0,1,2,3,4 --n 1000 --n_reps 100 --save dim_sweep.png
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from wpsd import tpsd, psd
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from rfsd.wpsd import tpsd, psd


# ─────────────────────────────────────────────────────────────────────────────
# Data generators  (identical to run_msnr_experiment.py)
# ─────────────────────────────────────────────────────────────────────────────

def _standard_gaussian(n, d, rng):
    X = rng.standard_normal((n, d))
    return X, -X

def _gauss_mean(n, d, rng, shift=0.3):
    X = rng.standard_normal((n, d))
    X[:, 0] += shift
    return X, -X

def _gauss_variance(n, d, rng, extra_var=0.7):
    X = rng.standard_normal((n, d))
    X[:, 0] *= np.sqrt(1 + extra_var)
    return X, -X

def _gauss_t(n, d, rng, df=5):
    X = rng.standard_t(df, size=(n, d))
    # Long's setup: null model is p = N(0, I), data is standard Student-t.
    # Score of N(0, I) is grad log p(x) = -x  (NOT -x/sigma2).
    S = -X
    return X, S

def _perturbed_gaussian(n, d, rng, extra_var=0.7):
    X = rng.standard_normal((n, d))
    X[:, 0] *= np.sqrt(1 + extra_var)
    return X, -X

def _laplace(n, d, rng):
    X = rng.laplace(loc=0.0, scale=1.0/np.sqrt(2), size=(n, d))
    return X, -X

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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_stat(r):
    if 'statistic' in r:
        return float(r['statistic']), float(r['p_value'])
    for key in ('studentised', 'unweighted'):
        if key in r and isinstance(r[key], dict):
            return float(r[key]['statistic']), float(r[key]['p_value'])
    raise KeyError(f"Cannot find statistic in result dict: {list(r.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_dim_experiment(experiment="perturbed_gauss", n=1000,
                       d_list=None, Q_list=None,
                       n_reps=100, alpha=0.05,
                       base_seed=0, verbose=True,
                       split_ratio=None):
    """
    Sweep over (Q, d) pairs.

    Returns
    -------
    results : {Q: {d: {'psd': {...}, 'wpsd': {...}}}}
        Inner dicts have keys: msnr_mean, msnr_std, reject_mean, reject_std.
        MSNR is stored as log10(z²).
    """
    if d_list is None:
        d_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    if Q_list is None:
        Q_list = [0, 1, 2, 3, 4]

    gen = GENERATORS[experiment]
    results = {Q: {} for Q in Q_list}

    for Q in Q_list:
        for d in d_list:
            if verbose:
                print(f"  Q={Q}, d={d}", end="", flush=True)
            t0 = time.time()

            psd_log_z2s  = np.zeros(n_reps)
            wpsd_log_z2s = np.zeros(n_reps)
            psd_rejs     = np.zeros(n_reps)
            wpsd_rejs    = np.zeros(n_reps)

            for rep in range(n_reps):
                seed = base_seed + Q * 1_000_000 + d * 10_000 + rep
                rng  = np.random.default_rng(seed)
                X, S = gen(n, d, rng)

                psd_kwargs  = {} if split_ratio is None else {'split_ratio': split_ratio}
                tpsd_kwargs = {} if split_ratio is None else {'split_ratio': split_ratio}
                r_psd  = psd( X, S, Q=Q, seed=seed, **psd_kwargs)
                r_wpsd = tpsd(X, S, Q=Q, seed=seed, **tpsd_kwargs)

                z_psd,  p_psd  = _extract_stat(r_psd)
                z_wpsd, p_wpsd = _extract_stat(r_wpsd)

                # log10(z²) — matches Figure 4 y-axis convention
                psd_log_z2s [rep] = np.log10(max(z_psd  ** 2, 1e-12))
                wpsd_log_z2s[rep] = np.log10(max(z_wpsd ** 2, 1e-12))
                psd_rejs    [rep] = float(p_psd  < alpha)
                wpsd_rejs   [rep] = float(p_wpsd < alpha)

            results[Q][d] = {
                "psd": {
                    "msnr_mean":   float(np.mean(psd_log_z2s)),
                    "msnr_std":    float(np.std(psd_log_z2s,  ddof=1) / np.sqrt(n_reps)),
                    "reject_mean": float(np.mean(psd_rejs)),
                    "reject_std":  float(np.std(psd_rejs, ddof=1) / np.sqrt(n_reps)),
                },
                "wpsd": {
                    "msnr_mean":   float(np.mean(wpsd_log_z2s)),
                    "msnr_std":    float(np.std(wpsd_log_z2s,  ddof=1) / np.sqrt(n_reps)),
                    "reject_mean": float(np.mean(wpsd_rejs)),
                    "reject_std":  float(np.std(wpsd_rejs, ddof=1) / np.sqrt(n_reps)),
                },
            }

            elapsed = time.time() - t0
            if verbose:
                rp  = results[Q][d]["psd"]
                rw  = results[Q][d]["wpsd"]
                print(f"  [{elapsed:.1f}s]  "
                      f"PSD  log10-MSNR={rp['msnr_mean']:.3f}  rej={rp['reject_mean']:.3f}  |  "
                      f"wPSD log10-MSNR={rw['msnr_mean']:.3f}  rej={rw['reject_mean']:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_dim_results(results, Q_list, d_list, experiment="perturbed_gauss",
                     n=1000, alpha=0.05, save_path=None):
    """
    Two-panel Figure 4-style plot.
    X-axis: d (log scale).  One colour per Q; solid = PSD, dashed = λ-SPSD.
    """
    title_str = EXPERIMENT_TITLES.get(experiment, experiment)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(Q_list)))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.45)

    ds = np.array(d_list)
    log10_ref = np.log10(2)   # MSNR reference under H0 (z²~chi²(1), median≈log10(1))

    for qi, Q in enumerate(Q_list):
        col = colors[qi]
        psd_msnr  = np.array([results[Q][d]["psd"]["msnr_mean"]   for d in d_list])
        psd_msnr_e= np.array([results[Q][d]["psd"]["msnr_std"]    for d in d_list])
        wp_msnr   = np.array([results[Q][d]["wpsd"]["msnr_mean"]  for d in d_list])
        wp_msnr_e = np.array([results[Q][d]["wpsd"]["msnr_std"]   for d in d_list])

        psd_rej   = np.array([results[Q][d]["psd"]["reject_mean"] for d in d_list])
        psd_rej_e = np.array([results[Q][d]["psd"]["reject_std"]  for d in d_list])
        wp_rej    = np.array([results[Q][d]["wpsd"]["reject_mean"]for d in d_list])
        wp_rej_e  = np.array([results[Q][d]["wpsd"]["reject_std"] for d in d_list])

        lbl_psd  = f"Q={Q} PSD"
        lbl_wpsd = f"Q={Q} " + r"$\lambda$-SPSD"

        # top panel: log10(MSNR)
        ax = axes[0]
        ax.errorbar(ds, psd_msnr, yerr=psd_msnr_e, color=col,
                    linestyle="-",  marker="o", markersize=4, linewidth=1.4,
                    capsize=3, label=lbl_psd)
        ax.errorbar(ds, wp_msnr,  yerr=wp_msnr_e,  color=col,
                    linestyle="--", marker="^", markersize=4, linewidth=1.4,
                    capsize=3, label=lbl_wpsd)

        # bottom panel: rejection rate
        ax = axes[1]
        ax.errorbar(ds, psd_rej,  yerr=psd_rej_e,  color=col,
                    linestyle="-",  marker="o", markersize=4, linewidth=1.4,
                    capsize=3, label=lbl_psd)
        ax.errorbar(ds, wp_rej,   yerr=wp_rej_e,   color=col,
                    linestyle="--", marker="^", markersize=4, linewidth=1.4,
                    capsize=3, label=lbl_wpsd)

    # top panel decoration
    ax = axes[0]
    ax.axhline(log10_ref, linestyle=":", color="grey", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks(d_list)
    ax.set_xticklabels([str(d) for d in d_list], rotation=45, ha="right")
    ax.set_xlabel("d", fontsize=11)
    ax.set_ylabel("log10(MSNR)", fontsize=11)
    ax.set_title(f"{title_str}: log10(MSNR) vs d  (n={n})", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, frameon=False,
              ncol=2, handlelength=2.5)

    # bottom panel decoration
    ax = axes[1]
    ax.axhline(alpha, linestyle=":", color="grey", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks(d_list)
    ax.set_xticklabels([str(d) for d in d_list], rotation=45, ha="right")
    ax.set_xlabel("d", fontsize=11)
    ax.set_ylabel("Rejection rate", fontsize=11)
    ax.set_title(f"{title_str}: Rejection rate vs d  (n={n})", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7, frameon=False,
              ncol=2, handlelength=2.5)

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
# Table export  (rejection rate only — cleaner for paper)
# ─────────────────────────────────────────────────────────────────────────────

def save_results_table(results, Q_list, d_list, experiment, n, alpha,
                       csv_path=None, latex_path=None):
    """
    Save rejection rate and log10(MSNR) sweep (rows=d, cols=Q pairs)
    as CSV and/or two LaTeX tables (one per metric).
    """
    import csv as csv_mod
    title = EXPERIMENT_TITLES.get(experiment, experiment)

    if csv_path:
        header = ['d']
        for Q in Q_list:
            header += [f'Q{Q}_psd_rej',      f'Q{Q}_psd_rej_se',
                       f'Q{Q}_wpsd_rej',     f'Q{Q}_wpsd_rej_se',
                       f'Q{Q}_psd_msnr',     f'Q{Q}_psd_msnr_se',
                       f'Q{Q}_wpsd_msnr',    f'Q{Q}_wpsd_msnr_se']
        with open(csv_path, 'w', newline='') as f:
            w = csv_mod.writer(f)
            w.writerow(header)
            for d in d_list:
                row = [d]
                for Q in Q_list:
                    rp = results[Q][d]['psd']
                    rw = results[Q][d]['wpsd']
                    row += [rp['reject_mean'], rp['reject_std'],
                            rw['reject_mean'], rw['reject_std'],
                            rp['msnr_mean'],   rp['msnr_std'],
                            rw['msnr_mean'],   rw['msnr_std']]
                w.writerow(row)
        print(f"CSV   → {csv_path}")

    if latex_path:
        def _bold(s): return r'\textbf{' + s + '}'

        def _make_table(metric_key, caption_metric, fmt):
            col_spec = 'r' + ' rr' * len(Q_list)
            lines = [
                r'\begin{table}[ht]',
                r'\centering',
                (r'\caption{' + caption_metric + r' vs $d$ --- ' + title +
                 rf' ($n={n}$, $\alpha={alpha}$)' + r'}'),
                r'\begin{tabular}{' + col_spec + r'}',
                r'\toprule',
            ]
            q_header = '$d$'
            for Q in Q_list:
                q_header += r' & \multicolumn{2}{c}{$Q=' + str(Q) + r'$}'
            lines.append(q_header + r' \\')

            cmidrules = ''
            for i in range(len(Q_list)):
                s = 2 + 2 * i
                cmidrules += r'\cmidrule(lr){' + f'{s}-{s+1}' + '}'
            lines.append(cmidrules)

            sub = ''
            for _ in Q_list:
                sub += r' & PSD & $\lambda$-SPSD'
            lines += [sub + r' \\', r'\midrule']

            for d in d_list:
                row_str = str(d)
                for Q in Q_list:
                    rp = results[Q][d]['psd']
                    rw = results[Q][d]['wpsd']
                    pv = rp[metric_key]
                    wv = rw[metric_key]
                    ps = fmt.format(pv)
                    ws = fmt.format(wv)
                    if   wv > pv: ws = _bold(ws)
                    elif pv > wv: ps = _bold(ps)
                    row_str += f' & {ps} & {ws}'
                lines.append(row_str + r' \\')

            lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
            return '\n'.join(lines) + '\n'

        rej_table  = _make_table('reject_mean', 'Rejection rate', '{:.3f}')
        msnr_table = _make_table('msnr_mean',   r'$\log_{10}$(MSNR)', '{:.3f}')

        with open(latex_path, 'w') as f:
            f.write(rej_table + '\n' + msnr_table)
        print(f"LaTeX → {latex_path}  (2 tables: rejection rate + log10 MSNR)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="log10(MSNR) / rejection-rate vs d sweep")
    p.add_argument("--experiment", default="perturbed_gauss",
                   choices=list(GENERATORS))
    p.add_argument("--n",      type=int,   default=1000)
    p.add_argument("--n_reps", type=int,   default=100)
    p.add_argument("--alpha",  type=float, default=0.05)
    p.add_argument("--q_list", type=str,   default="0,1,2,3,4",
                   help="Comma-separated Q values, e.g. 0,1,2,3,4")
    p.add_argument("--d_list", type=str,
                   default="2,4,8,16,32,64,128,256,512,1024",
                   help="Comma-separated d values")
    p.add_argument("--seed",   type=int,   default=36)
    p.add_argument("--split_ratio", type=float, default=None,
                   help="Train/test split fraction passed to both psd "
                        "and tpsd. None -> per-method default (psd=0.2, "
                        "tpsd=0.3). 0.7 -> 35/35/30 split for wPSD.")
    p.add_argument("--save",   type=str,   default=None)
    p.add_argument("--save_csv",   type=str, default=None,
                   help="Path to save results as CSV")
    p.add_argument("--save_latex", type=str, default=None,
                   help="Path to save rejection-rate table as LaTeX")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    Q_list = [int(q) for q in args.q_list.split(",")]
    d_list = [int(d) for d in args.d_list.split(",")]

    print(f"Experiment : {args.experiment}")
    print(f"n={args.n}, reps={args.n_reps}, alpha={args.alpha}")
    print(f"Q values   : {Q_list}")
    print(f"d values   : {d_list}")
    print()

    results = run_dim_experiment(
        experiment=args.experiment,
        n=args.n,
        d_list=d_list,
        Q_list=Q_list,
        n_reps=args.n_reps,
        alpha=args.alpha,
        base_seed=args.seed,
        verbose=True,
        split_ratio=args.split_ratio,
    )

    plot_dim_results(
        results, Q_list, d_list,
        experiment=args.experiment,
        n=args.n,
        alpha=args.alpha,
        save_path=args.save,
    )

    if args.save_csv or args.save_latex:
        save_results_table(
            results, Q_list, d_list,
            experiment=args.experiment,
            n=args.n, alpha=args.alpha,
            csv_path=args.save_csv,
            latex_path=args.save_latex,
        )
