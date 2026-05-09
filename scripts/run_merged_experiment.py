"""
run_merged_experiment.py
========================
Unified runner that evaluates every test on the SAME synthetic data per
(scenario, dimension, repetition). The two downstream figures
(popular-methods comparison, PSD vs lambda-PSD) are drawn from this single
cache, which guarantees that lines appearing in both (e.g. lambda-PSD L=0..4)
are bit-identical between figures.

Methods evaluated per (scenario, d, rep)
----------------------------------------
  Rejection rate only      : IMQ KSD, Gauss KSD, Gauss FSSD-opt
  Rejection + log10(MSNR)  : PSD L=0..4   (uniform lambda)
                             lambda-PSD L=0..4 (adaptive lambda via CE)

Sweeps
------
  scenarios : null, perturbed_gauss, gauss_t (Student-t, n=2000), laplace
  d         : 2, 4, 8, 16, 32, 64, 128
  L (== Q)  : 0, 1, 2, 3, 4
  reps      : 100  (configurable via --n_reps)
  n         : 1000  (Student-t panel uses 2000, matching the convention in
                     run_gof_experiment.py and run_psd_vs_wpsd_all.py)
  alpha     : 0.05

Reproducibility
---------------
The per-rep seed is a deterministic function of (scenario index, d, rep):

    seed = base_seed + scen_idx * 1e8 + d * 1e4 + rep * 4

so reruns of this script (same flags) produce bit-identical numbers, and the
RFSD-only runner (run_rfsd_experiment.py) can reuse the exact same X by
deriving its seed the same way.

  Note: rfsd.wpsd.sparse_ce_lambda (the CE lambda search inside tpsd) draws
  from the GLOBAL numpy RNG, so passing seed= to tpsd is not sufficient to
  pin its output. We wrap each tpsd call in NumpySeedContext to pin global
  state per (rep, Q). Without this, lambda-PSD numbers drift between runs
  even with a fixed seed.

Outputs
-------
Under --cache_dir (default results/merged/):
  <scenario>_<tag>.pkl   nested results dict (full)
  <scenario>_<tag>.csv   flat per-d dump for spreadsheet inspection

Downstream plots
----------------
  scripts/plot_popular_methods_merged.py   reads this cache + RFSD overlay
  scripts/plot_psd_vs_wpsd_merged.py       reads this cache

Usage
-----
  python scripts/run_merged_experiment.py
  python scripts/run_merged_experiment.py --force      # ignore cache
  python scripts/run_merged_experiment.py --n_reps 200 # tighter SE bars
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kgof.data import Data as KGOFData
from kgof.density import IsotropicNormal
from kgof.goftest import GaussFSSD, KernelSteinTest
from kgof.kernel import KIMQ
from kgof.util import NumpySeedContext, fit_gaussian_draw

from rfsd.kernel import KGauss2
from rfsd.util import meddistance
from rfsd.wpsd import psd, tpsd


# ---------------------------------------------------------------------------
# Data generators
#
# Mirrors run_dim_experiment.py so PSD/wPSD numbers are directly comparable.
# In every scenario the null model is p = N(0, I_d), so the score
#   grad log p(x) = -x
# regardless of the true sampling distribution. Each generator returns
# (X, S) where S = grad log p(X) under the null.
# ---------------------------------------------------------------------------

def _standard_gaussian(n, d, rng):
    X = rng.standard_normal((n, d))
    return X, -X


def _perturbed_gaussian(n, d, rng, extra_var=0.7):
    """Variance perturbation: X[:, 0] ~ N(0, 1 + extra_var)."""
    X = rng.standard_normal((n, d))
    X[:, 0] *= np.sqrt(1 + extra_var)
    return X, -X


def _gauss_t(n, d, rng, df=5):
    """Standard Student-t (no scaling); score is still -x under the N(0,I) null."""
    X = rng.standard_t(df, size=(n, d))
    return X, -X


def _laplace(n, d, rng):
    """Laplace with scale 1/sqrt(2) so Var = 1, matching the kgof DSLaplace setup."""
    X = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2), size=(n, d))
    return X, -X


# (key, label, generator, n_override). Order is load-bearing: scen_idx in the
# seed formula relies on this list, so reordering would change all seeds.
SCENARIOS = [
    ('null',            'Standard Gaussian',  _standard_gaussian, None),
    ('perturbed_gauss', 'Perturbed Gaussian', _perturbed_gaussian, None),
    ('gauss_t',         'Student-t',          _gauss_t,            2000),
    ('laplace',         'Laplace',            _laplace,            None),
]

POPULAR_TESTS = ['IMQ KSD', 'Gauss KSD', 'Gauss FSSD-opt']


# ---------------------------------------------------------------------------
# Single-rep evaluation
# ---------------------------------------------------------------------------

def _extract_stat(r):
    """Return (statistic, p_value) from a psd/tpsd result dict.

    psd returns {'unweighted': {...}}; tpsd returns {'studentised': {...}};
    older variants flatten into top-level 'statistic'/'p_value'.
    """
    if 'statistic' in r:
        return float(r['statistic']), float(r['p_value'])
    for key in ('studentised', 'unweighted'):
        sub = r.get(key)
        if isinstance(sub, dict):
            return float(sub['statistic']), float(sub['p_value'])
    raise KeyError(f'Cannot find statistic in: {list(r.keys())}')


def run_one_rep(X, S, d, seed, Q_list, alpha, J, split_ratio):
    """Evaluate every method on a single (X, S) realisation.

    Parameters
    ----------
    X, S : ndarray, shape (n, d)
        Sample matrix and the score grad log p(X) under the null p = N(0, I).
    d : int
        Sample dimension (passed for IsotropicNormal construction).
    seed : int
        Per-rep seed; used directly for psd/tpsd splits and offset by small
        constants for kgof tests so the random streams don't alias.
    Q_list : list[int]
        Polynomial orders L for both PSD and lambda-PSD.
    alpha : float
        Significance level for all tests.
    J : int
        Number of FSSD feature locations.
    split_ratio : float
        Fraction of data used for training in psd/tpsd. 0.7 -> 35/35/30 split
        for tpsd (train1/train2/test) and 70/30 for psd; matches the
        convention used in rfsd/experiments/gof_testing_experiments.py.

    Returns
    -------
    {'psd':  {Q: {'reject': 0/1, 'log10_z2': float}},
     'wpsd': {Q: {'reject': 0/1, 'log10_z2': float}},
     'pop':  {test_name: {'reject': 0/1}}}
    """
    out = {'psd': {}, 'wpsd': {}, 'pop': {}}

    # ---- PSD / lambda-PSD ------------------------------------------------
    # tpsd's CE lambda search (sparse_ce_lambda in rfsd/wpsd.py) draws from
    # the GLOBAL numpy RNG, so passing seed= alone does not pin its output.
    # NumpySeedContext saves/restores global state, giving us deterministic
    # tpsd numbers per (rep, Q) without modifying the core library. Without
    # this wrapper, lambda-PSD numbers drift between runs at the same seed.
    for Q in Q_list:
        r_psd  = psd( X, S, Q=Q, seed=seed, split_ratio=split_ratio)
        with NumpySeedContext(seed=seed + 1000 + Q):
            r_wpsd = tpsd(X, S, Q=Q, seed=seed, split_ratio=split_ratio)
        z_psd, p_psd = _extract_stat(r_psd)
        z_wp,  p_wp  = _extract_stat(r_wpsd)
        out['psd'][Q] = {
            'reject':   float(p_psd < alpha),
            'log10_z2': float(np.log10(max(z_psd ** 2, 1e-12))),
        }
        out['wpsd'][Q] = {
            'reject':   float(p_wp < alpha),
            'log10_z2': float(np.log10(max(z_wp ** 2, 1e-12))),
        }

    # ---- Popular methods (kgof) -----------------------------------------
    # Note: kgof uses tr_proportion=0.2 (20% tune / 80% test) for FSSD-opt,
    # which is *different* from the 70/30 split used by PSD/wPSD above.
    # This matches the convention in gof_testing_experiments.py exactly.
    p_model = IsotropicNormal(np.zeros(d), 1)
    dat     = KGOFData(X)
    tr, te  = dat.split_tr_te(tr_proportion=0.2, seed=seed)

    med_l2 = meddistance(X, subsample=1000)
    sigma2 = med_l2 ** 2

    kgauss = KGauss2(sigma2)
    kimq   = KIMQ(c=1)

    # IMQ KSD: full-data test, bootstrap null. Seeded for reproducibility.
    t_imq = KernelSteinTest(p_model, kimq, alpha=alpha, seed=seed + 1)
    r_imq = t_imq.perform_test(dat)
    out['pop']['IMQ KSD'] = {'reject': float(r_imq['h0_rejected'])}

    # Gauss KSD: same as IMQ but with the Gaussian kernel at the median bw.
    t_gks = KernelSteinTest(p_model, kgauss, alpha=alpha, seed=seed + 2)
    r_gks = t_gks.perform_test(dat)
    out['pop']['Gauss KSD'] = {'reject': float(r_gks['h0_rejected'])}

    # Gauss FSSD-opt: tune locs/widths on tr (20%), test on te (80%).
    # Step 1: draw J initial feature locations from a Gaussian fit to tr.
    Vgauss = fit_gaussian_draw(tr.data(), J, reg=1e-6, seed=seed + 3)
    if len(Vgauss.shape) == 0:
        Vgauss = np.array([[Vgauss]])
    if J == 1:
        Vgauss = Vgauss.T

    # Step 2: grid-search the Gaussian bandwidth on a log-spaced ladder
    # around the median heuristic.
    n_gwidth_cand = 5
    gwidth_factors = 2.0 ** np.linspace(-3, 3, n_gwidth_cand)
    list_gwidth = np.hstack((sigma2 * gwidth_factors,))
    besti, _ = GaussFSSD.grid_search_gwidth(p_model, tr, Vgauss, list_gwidth)
    gwidth = list_gwidth[besti]

    # Step 3: jointly optimise locations + bandwidth (gradient-based, on tr).
    ops = dict(reg=1e-2, max_iter=40, tol_fun=1e-4, disp=False,
               locs_bounds_frac=10.0, gwidth_lb=1e-1, gwidth_ub=1e4)
    Vgauss_opt, gwidth_opt, _ = GaussFSSD.optimize_locs_widths(
        p_model, tr, gwidth, Vgauss, **ops)

    # Step 4: evaluate on the held-out 80% test split.
    t_fssd = GaussFSSD(p_model, gwidth_opt, Vgauss_opt, alpha=alpha,
                       n_simulate=2000, seed=seed + 4)
    r_fssd = t_fssd.perform_test(te)
    out['pop']['Gauss FSSD-opt'] = {'reject': float(r_fssd['h0_rejected'])}

    return out


# ---------------------------------------------------------------------------
# Scenario loop
# ---------------------------------------------------------------------------

def run_scenario(key, label, gen, n_for_panel, d_list, Q_list, n_reps,
                 alpha, base_seed, J, split_ratio, scen_idx, verbose=True):
    """Sweep one scenario across (d, rep) and aggregate to means + standard errors.

    The per-rep seed is

        seed = base_seed + scen_idx * 1e8 + d * 1e4 + rep * 4

    The 'rep * 4' stride leaves room for kgof tests to use seed+1..seed+4
    without colliding with neighbouring reps' seeds.

    Returns a results dict with metadata at the top level and per-d
    aggregates under results['by_d'][d]['psd'|'wpsd'|'pop'].
    """
    Q_list_t = list(Q_list)
    results = {
        'scenario':    key,
        'label':       label,
        'n':           n_for_panel,
        'n_reps':      n_reps,
        'alpha':       alpha,
        'd_list':      list(d_list),
        'Q_list':      Q_list_t,
        'split_ratio': split_ratio,
        'J':           J,
        'base_seed':   base_seed,
        'by_d':        {},
    }

    for d in d_list:
        rejs = {'psd':  {Q: [] for Q in Q_list_t},
                'wpsd': {Q: [] for Q in Q_list_t},
                'pop':  {tn: [] for tn in POPULAR_TESTS}}
        msnrs = {'psd':  {Q: [] for Q in Q_list_t},
                 'wpsd': {Q: [] for Q in Q_list_t}}

        if verbose:
            print(f'  [{label}] d={d:>3}  ', end='', flush=True)
        t0 = time.time()

        for rep in range(n_reps):
            seed = (base_seed
                    + scen_idx * 100_000_000
                    + d         * 10_000
                    + rep      * 4)
            rng  = np.random.default_rng(seed)
            X, S = gen(n_for_panel, d, rng)

            try:
                rep_out = run_one_rep(X, S, d, seed, Q_list_t,
                                      alpha=alpha, J=J,
                                      split_ratio=split_ratio)
            except Exception as e:
                print(f'\n    rep {rep} failed ({type(e).__name__}: {e}); '
                      'skipping')
                continue

            for Q in Q_list_t:
                rejs ['psd'] [Q].append(rep_out['psd'] [Q]['reject'])
                rejs ['wpsd'][Q].append(rep_out['wpsd'][Q]['reject'])
                msnrs['psd'] [Q].append(rep_out['psd'] [Q]['log10_z2'])
                msnrs['wpsd'][Q].append(rep_out['wpsd'][Q]['log10_z2'])
            for tn in POPULAR_TESTS:
                rejs['pop'][tn].append(rep_out['pop'][tn]['reject'])

        agg = {'psd': {}, 'wpsd': {}, 'pop': {}}

        def _se(arr):
            arr = np.asarray(arr, dtype=float)
            if len(arr) <= 1:
                return 0.0
            return float(arr.std(ddof=1) / np.sqrt(len(arr)))

        for Q in Q_list_t:
            for m in ('psd', 'wpsd'):
                rj = np.asarray(rejs [m][Q], dtype=float)
                mz = np.asarray(msnrs[m][Q], dtype=float)
                agg[m][Q] = {
                    'reject_mean': float(rj.mean()) if len(rj) else float('nan'),
                    'reject_se':   _se(rj),
                    'msnr_mean':   float(mz.mean()) if len(mz) else float('nan'),
                    'msnr_se':     _se(mz),
                    'n_done':      int(len(rj)),
                }
        for tn in POPULAR_TESTS:
            rj = np.asarray(rejs['pop'][tn], dtype=float)
            agg['pop'][tn] = {
                'reject_mean': float(rj.mean()) if len(rj) else float('nan'),
                'reject_se':   _se(rj),
                'n_done':      int(len(rj)),
            }
        results['by_d'][d] = agg

        if verbose:
            elapsed = time.time() - t0
            r0 = agg['wpsd'][Q_list_t[0]]['reject_mean']
            print(f'done in {elapsed:6.1f}s  '
                  f'(wPSD L={Q_list_t[0]} rej={r0:.2f})')

    return results


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _save_csv(csv_path, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    Q_list = results['Q_list']

    header = ['d']
    for Q in Q_list:
        header += [f'psd_Q{Q}_rej',  f'psd_Q{Q}_rej_se',
                   f'psd_Q{Q}_msnr', f'psd_Q{Q}_msnr_se',
                   f'wpsd_Q{Q}_rej', f'wpsd_Q{Q}_rej_se',
                   f'wpsd_Q{Q}_msnr',f'wpsd_Q{Q}_msnr_se']
    for tn in POPULAR_TESTS:
        header += [f'{tn}_rej', f'{tn}_rej_se']

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in results['d_list']:
            agg = results['by_d'][d]
            row = [d]
            for Q in Q_list:
                rp = agg['psd' ][Q]
                rw = agg['wpsd'][Q]
                row += [rp['reject_mean'], rp['reject_se'],
                        rp['msnr_mean'],   rp['msnr_se'],
                        rw['reject_mean'], rw['reject_se'],
                        rw['msnr_mean'],   rw['msnr_se']]
            for tn in POPULAR_TESTS:
                rj = agg['pop'][tn]
                row += [rj['reject_mean'], rj['reject_se']]
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--n',           type=int,   default=1000)
    p.add_argument('--n_reps',      type=int,   default=100)
    p.add_argument('--alpha',       type=float, default=0.05)
    p.add_argument('--q_list',      type=str,   default='0,1,2,3,4')
    p.add_argument('--d_list',      type=str,   default='2,4,8,16,32,64,128')
    p.add_argument('--seed',        type=int,   default=36)
    p.add_argument('--J',           type=int,   default=10,
                   help='Number of Gauss-FSSD feature locations')
    p.add_argument('--split_ratio', type=float, default=0.7,
                   help='Train/test split for psd & tpsd; 0.7 -> 35/35/30 '
                        'for wPSD, 70/30 for PSD')
    p.add_argument('--cache_dir',   type=str,   default='results/merged')
    p.add_argument('--force', action='store_true',
                   help='Ignore cache and re-run every scenario.')
    return p.parse_args()


def _scen_tag(args, key, n_for_panel, d_list, Q_list):
    return (
        f'{key}'
        f'_n{n_for_panel}_reps{args.n_reps}'
        f"_d-{'-'.join(map(str, d_list))}"
        f"_Q-{'-'.join(map(str, Q_list))}"
        f'_sr{args.split_ratio:g}_J{args.J}_s{args.seed}'
    )


def main():
    args = _parse()
    Q_list = [int(q) for q in args.q_list.split(',')]
    d_list = [int(d) for d in args.d_list.split(',')]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f'sweeps : d={d_list}  Q={Q_list}  reps={args.n_reps}  '
          f'split_ratio={args.split_ratio}  J={args.J}  base_seed={args.seed}')
    print(f'cache  : {cache_dir}')

    for scen_idx, (key, label, gen, n_override) in enumerate(SCENARIOS):
        n_for_panel = args.n if n_override is None else n_override
        tag = _scen_tag(args, key, n_for_panel, d_list, Q_list)
        pkl_path = cache_dir / f'{tag}.pkl'
        csv_path = cache_dir / f'{tag}.csv'

        if pkl_path.exists() and not args.force:
            print(f'\n[{label}] cache hit -> {pkl_path}')
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
        else:
            print(f'\n[{label}] running (n={n_for_panel}, reps={args.n_reps})')
            t0 = time.time()
            results = run_scenario(
                key=key, label=label, gen=gen,
                n_for_panel=n_for_panel,
                d_list=d_list, Q_list=Q_list, n_reps=args.n_reps,
                alpha=args.alpha, base_seed=args.seed,
                J=args.J, split_ratio=args.split_ratio,
                scen_idx=scen_idx,
            )
            print(f'[{label}] total {time.time()-t0:.1f}s')
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)

        _save_csv(csv_path, results)
        print(f'  pickle -> {pkl_path}')
        print(f'  csv    -> {csv_path}')


if __name__ == '__main__':
    main()
