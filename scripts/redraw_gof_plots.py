"""
redraw_gof_plots.py
===================
Re-render GoF experiment plots from cached results.pck/params.pck without
re-running the experiment. Useful when only plotting code (e.g. config.py
test name registration / colors) has changed.

Usage:
    python scripts/redraw_gof_plots.py <stored-data-dir> [<stored-data-dir> ...]

Each <stored-data-dir> is a `*-stored-data/` directory produced by
run_gof_experiment.py (containing results.pck and params.pck).
PNGs are written next to the directory, with the same base name the original
run would have used.
"""
from __future__ import absolute_import, print_function

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rfsd.experiments.gof_testing_experiments as goft_exp
from rfsd.util import restore_object


def _expt_name_from_dir(dir_path: Path) -> str:
    name = dir_path.name
    suffix = "-stored-data"
    if not name.endswith(suffix):
        raise ValueError(f"Expected a *-stored-data directory, got: {dir_path}")
    return name[: -len(suffix)]


def _ymax_for(expt_name: str) -> float:
    return 0.2 if expt_name.startswith("goftest-null-") else 1.05


def redraw(stored_dir: Path):
    expt_name = _expt_name_from_dir(stored_dir)
    parent = stored_dir.parent

    results = restore_object(str(stored_dir), "results")
    params = restore_object(str(stored_dir), "params")

    cwd = os.getcwd()
    os.chdir(parent)
    try:
        goft_exp.show_all_results(
            results, params,
            save=expt_name,
            ymax=_ymax_for(expt_name),
            show=False,
        )
    finally:
        os.chdir(cwd)

    for J in results:
        png = parent / f"{expt_name}-J-{J}.png"
        print(f"  → {png}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sns.set_style("white")
    sns.set_context("notebook", font_scale=3, rc={"lines.linewidth": 3})

    for arg in sys.argv[1:]:
        d = Path(arg).resolve()
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}")
            sys.exit(1)
        print(f"Redrawing {d.name}")
        redraw(d)
        plt.close("all")


if __name__ == "__main__":
    main()
