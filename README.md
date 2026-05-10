# λ-PSD: Scalable Approximate SNR-Optimised Polynomial Stein Discrepancies

Code accompanying the paper *λ-PSD: Scalable Approximate SNR-Optimised Polynomial Stein Discrepancies*.

## Abstract

Assessing sampling quality remains a fundamental problem in modern Bayesian computation. In many settings, the target distribution $p$ is known only up to a normalising constant, while the approximation $q$ is represented through samples. Stein discrepancies provide a principled framework for directly measuring discrepancy between $p$ and $q$ using only score evaluations of the target distribution. Among these, kernel Stein discrepancy (KSD) has strong theoretical guarantees but typically requires quadratic computational complexity in the number of samples $n$, limiting scalability in large-scale settings. Recent work on polynomial Stein discrepancy (PSD) addresses this limitation by replacing kernel witness functions with polynomial Stein features of order $Q$, yielding computationally efficient linear-time discrepancy measures. PSD is particularly effective at detecting moment convergence under the Bernstein-von Mises limit. However, its statistical power is highly sensitive to the choice of polynomial order $Q$, and the statistical mechanisms underlying this behaviour remain poorly understood.

## Reproducing the figures

```bash
# Environment (one-off)
conda create -n PSD_3p11 python==3.11 && conda activate PSD_3p11
pip install numpy==1.23.5 git+https://github.com/wittawatj/kernel-gof.git \
            autograd seaborn scikit-learn matplotlib

# Quick: re-render figures from the cached results shipped in results/ (seconds)
python scripts/plot_popular_methods_merged.py     # popular-methods comparison
python scripts/plot_psd_vs_wpsd_merged.py          # PSD vs λ-PSD
python scripts/run_runtime_experiment.py           # paper Figures 3 and 4

# Full: re-run every experiment from scratch (~5h)
python scripts/run_merged_experiment.py            # main sweep (~3-5h)
python scripts/run_rfsd_experiment.py              # RFSD overlay (~15min)
python scripts/run_runtime_experiment.py           # runtime sweep (~10min)
```

See [`RunInstructions.txt`](./RunInstructions.txt) for the full protocol and CLI flags.
