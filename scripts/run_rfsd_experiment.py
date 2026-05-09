"""
run_rfsd_experiment.py
======================
Stand-alone runner for RFSD (L1-IMQ random-features Stein discrepancy,
Huggins & Mackey 2018) that piggybacks on the seed convention of
run_merged_experiment.py. The X data seen by RFSD for each (scenario, d, rep)
is bit-identical to what the merged runner saw, so the RFSD line can be
overlaid on the popular-methods figure without re-running everything.

Why a separate runner?
----------------------
RFSD wasn't part of the original merged sweep (it costs a lot at high d due
to the n_draw=5000+d*500 null simulation). Splitting it off means a user can
add the RFSD curve in ~15-20 minutes without redoing the ~4-hour full sweep.

Calibration caveat
------------------
The nominal-level table L1_IMQ_NOMINAL_LEVELS in
rfsd/experiments/gof_testing_experiments.py only has entries for
d in {1, 3, 5, 7, 10, 15, 20}. Our sweep uses d in {2, 4, 8, 16, 32, 64, 128},
none of which appear in the table, so the runner falls back to the
formulaic correction

    alpha_d = 0.05 / (0.8 + 0.2 * d)

which is very conservative at high d (alpha = 0.0019 at d = 128) and was
not validated for those dimensions. The exact alpha used at each d is
recorded in the CSV/pickle as 'alpha_used' for transparency. Treat the
high-d numbers with this caveat in mind.

Output
------
Under --cache_dir (default results/merged/, alongside the merged caches):
  rfsd_<scenario>_<tag>.pkl
  rfsd_<scenario>_<tag>.csv

The plot script auto-detects the rfsd_ prefix:
  python scripts/plot_popular_methods_merged.py

Usage
-----
  python scripts/run_rfsd_experiment.py
  python scripts/run_rfsd_experiment.py --force
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kgof.data import Data as KGOFData
from kgof.density import IsotropicNormal

from rfsd.experiments.gof_testing_experiments import L1_IMQ_NOMINAL_LEVELS
from rfsd.goftest import RFDGofTest, RFDH0SimCovDrawV
from rfsd.rfsd import L1IMQFastKSD
from rfsd.util import meddistance

# Reuse generators + scenario list so seeds align with run_merged_experiment.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_merged_experiment import SCENARIOS  # noqa: E402


# RFSD hyperparameters: match gof_testing_experiments.py:single_gof_testing_round.
GAMMA      = 0.25  # L1-IMQ kernel exponent parameter
TEST_ALPHA = 0.05  # nominal Type-I error before per-d correction
TARGET_DF  = 0.5   # IMQ target degrees-of-freedom (controls feature scaling)

# Autograd's grad-of-log occasionally evaluates log(0) at IMQ boundary points;
# the resulting nan is masked downstream and the warning is just noise.
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='divide by zero encountered in log')


def _l1_imq_alpha(d, test_alpha=TEST_ALPHA, gamma=GAMMA):
    """Per-d significance level for RFSD with the L1-IMQ kernel.

    Mirrors the logic in gof_testing_experiments.py: prefer the precomputed
    nominal level from L1_IMQ_NOMINAL_LEVELS if available, otherwise fall
    back to the analytic correction alpha / (0.8 + 0.2 * d).
    """
    a = test_alpha / (.8 + .2 * d)
    if test_alpha == 0.05 and gamma in L1_IMQ_NOMINAL_LEVELS:
        a = L1_IMQ_NOMINAL_LEVELS[gamma].get(d, a)
    return a


def run_one_rep(X, d, seed):
    """Run RFSD once on (X, d) with deterministic null simulation.

    The null distribution is approximated by RFDH0SimCovDrawV with
    n_draw = 5000 + d*500 (same as the original popular-methods setup,
    growing with d so the simulated covariance stays well-conditioned).
    """
    p_model = IsotropicNormal(np.zeros(d), 1)
    dat     = KGOFData(X)

    # Median heuristic for the L1-IMQ scale; factor of 4 matches the
    # convention in single_gof_testing_round (c = 4 * med_l2).
    med_l2 = meddistance(X, subsample=1000)
    c      = 4 * med_l2

    rfd_imq  = L1IMQFastKSD(p_model, c=c, gamma=GAMMA, d=d,
                            target_df=TARGET_DF)
    null_sim = RFDH0SimCovDrawV(n_draw=5000 + d * 500, seed=seed + 5,
                                fromp=False)
    alpha    = _l1_imq_alpha(d)
    test     = RFDGofTest(p_model, rfd_imq, null_sim=null_sim, alpha=alpha)

    r = test.perform_test(dat)
    return float(r['h0_rejected'])


def run_scenario(scen_idx, key, label, gen, n_for_panel, d_list, n_reps,
                 base_seed, verbose=True):
    """Sweep RFSD across (d, rep) for one scenario.

    Uses the IDENTICAL seed formula as run_merged_experiment.py:

        seed = base_seed + scen_idx * 1e8 + d * 1e4 + rep * 4

    so the X seen here is bit-identical to the X used by the merged sweep.
    """
    results = {
        'scenario':   key,
        'label':      label,
        'n':          n_for_panel,
        'n_reps':     n_reps,
        'd_list':     list(d_list),
        'base_seed':  base_seed,
        'gamma':      GAMMA,
        'test_alpha': TEST_ALPHA,
        'by_d':       {},
    }

    for d in d_list:
        rejs = []
        if verbose:
            print(f'  [{label}] d={d:>3}  ', end='', flush=True)
        t0 = time.time()

        for rep in range(n_reps):
            seed = (base_seed
                    + scen_idx * 100_000_000
                    + d         * 10_000
                    + rep      * 4)
            rng  = np.random.default_rng(seed)
            X, _ = gen(n_for_panel, d, rng)

            try:
                rejs.append(run_one_rep(X, d, seed))
            except Exception as e:
                print(f'\n    rep {rep} failed ({type(e).__name__}: {e}); '
                      'skipping')
                continue

        rj = np.asarray(rejs, dtype=float)
        results['by_d'][d] = {
            'reject_mean': float(rj.mean()) if len(rj) else float('nan'),
            'reject_se':   (float(rj.std(ddof=1) / np.sqrt(len(rj)))
                            if len(rj) > 1 else 0.0),
            'n_done':      int(len(rj)),
            'alpha_used':  _l1_imq_alpha(d),
        }
        if verbose:
            print(f'done in {time.time()-t0:6.1f}s  '
                  f"(rej={results['by_d'][d]['reject_mean']:.2f}, "
                  f'alpha={_l1_imq_alpha(d):.4f})')
    return results


def _save_csv(csv_path, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['d', 'rfsd_rej', 'rfsd_rej_se', 'alpha_used', 'n_done'])
        for d in results['d_list']:
            r = results['by_d'][d]
            w.writerow([d, r['reject_mean'], r['reject_se'],
                        r['alpha_used'], r['n_done']])


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--n',         type=int, default=1000)
    p.add_argument('--n_reps',    type=int, default=100)
    p.add_argument('--d_list',    type=str, default='2,4,8,16,32,64,128')
    p.add_argument('--seed',      type=int, default=36)
    p.add_argument('--cache_dir', type=str, default='results/merged')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def _scen_tag(args, key, n_for_panel, d_list):
    return (
        f'rfsd_{key}'
        f'_n{n_for_panel}_reps{args.n_reps}'
        f"_d-{'-'.join(map(str, d_list))}"
        f'_s{args.seed}'
    )


def main():
    args = _parse()
    d_list = [int(d) for d in args.d_list.split(',')]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f'sweeps : d={d_list}  reps={args.n_reps}  base_seed={args.seed}')
    print(f'cache  : {cache_dir}')
    print(f'gamma={GAMMA}, target_df={TARGET_DF}, n_draw=5000+d*500')

    for scen_idx, (key, label, gen, n_override) in enumerate(SCENARIOS):
        n_for_panel = args.n if n_override is None else n_override
        tag = _scen_tag(args, key, n_for_panel, d_list)
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
                scen_idx=scen_idx, key=key, label=label, gen=gen,
                n_for_panel=n_for_panel, d_list=d_list,
                n_reps=args.n_reps, base_seed=args.seed,
            )
            print(f'[{label}] total {time.time()-t0:.1f}s')
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)

        _save_csv(csv_path, results)
        print(f'  pickle -> {pkl_path}')
        print(f'  csv    -> {csv_path}')


if __name__ == '__main__':
    main()
