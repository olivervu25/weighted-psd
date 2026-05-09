"""
Weighted Polynomial Stein Discrepancy (wPSD).

Core math only — no GofTest dependencies.
All functions operate on numpy arrays and can be used independently.

  svd_lambda        — analytical optimal lambda: Cov(g)^{-1} @ mean(g)  [fast, default]
  sparse_ce_lambda  — CE optimisation to find best projection direction   [slower, stochastic]
  tpsd              — test-optimised wPSD (lambda_method selects which)
  psd               — unweighted baseline PSD (uniform lambda)
"""

from __future__ import absolute_import, print_function

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Cross-entropy lambda search
# ---------------------------------------------------------------------------

def sparse_ce_lambda(g_aug, n_iter=50, n_cand=1000, elite_k=100,
                     tol=1e-4, patience=3):
    """
    Find a unit-norm direction lambda that maximises SNR = mean(u)^2 / var(u)
    where u = g_aug @ lambda, using Cross-Entropy optimisation over sparse
    Bernoulli-Gaussian candidates.

    Parameters
    ----------
    g_aug : (n, P) array
    n_iter : int
    n_cand : int  — candidates per iteration
    elite_k : int — elite set size for CE update
    tol : float   — relative improvement threshold for early stopping
    patience : int — consecutive non-improving iterations before stopping

    Returns
    -------
    best_lambda : (P,) array, unit-norm
    """
    n, P = g_aug.shape

    # Precompute once — replaces O(n*P) per candidate with O(P²)
    mu_G    = np.mean(g_aug, axis=0)   # (P,)
    Sigma_G = np.cov(g_aug.T)          # (P, P), ddof=1 matches R cov()
    if Sigma_G.ndim == 0:
        Sigma_G = Sigma_G.reshape(1, 1)

    p_active = np.full(P, 0.5)
    mu = np.zeros(P)
    sigma = np.ones(P)

    best_score = -np.inf
    best_lambda = np.zeros(P)
    prev_best = -np.inf
    no_improve_count = 0

    lambda_cand = np.zeros((n_cand, P))
    scores = np.full(n_cand, -np.inf)

    for _ in range(n_iter):
        scores[:] = -np.inf
        lambda_cand[:] = 0.0

        for j in range(n_cand):
            active = np.random.binomial(1, p_active)
            vals = np.random.normal(mu, sigma)
            lam = active * vals

            norm = np.sqrt(np.sum(lam ** 2))
            if norm < 1e-12:
                continue
            lam /= norm
            lambda_cand[j] = lam

            m = mu_G @ lam                  # scalar mean of projected features
            v = lam @ Sigma_G @ lam         # scalar variance
            if np.isnan(v) or v <= 1e-12:
                continue

            score = m ** 2 / v
            scores[j] = score
            if score > best_score:
                best_score = score
                best_lambda = lam.copy()

        # early stopping
        if np.isfinite(prev_best):
            improvement = abs(best_score - prev_best) / (abs(prev_best) + 1e-12)
            if improvement < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= patience:
                break
        prev_best = best_score

        # CE update on elite candidates
        order = np.argsort(scores)[::-1][:elite_k]
        elite_idx = order[np.isfinite(scores[order])]
        if len(elite_idx) == 0:
            continue

        elite = lambda_cand[elite_idx]
        p_active = np.mean(np.abs(elite) > 1e-6, axis=0)
        mu = np.mean(elite, axis=0)
        sigma = np.std(elite, axis=0, ddof=1)
        sigma = np.where(np.isnan(sigma) | (sigma < 1e-6), 1e-6, sigma)

    return best_lambda


# ---------------------------------------------------------------------------
# SVD lambda — analytical optimal direction
# ---------------------------------------------------------------------------

def svd_lambda(g_aug):
    """
    Analytical optimal lambda: Cov(g_aug)^{-1} @ mean(g_aug).

    Maximises (lambda^T mu)^2 / (lambda^T Sigma lambda) in closed form —
    the same objective CE searches stochastically, but solved exactly in O(P^3).
    No randomness; fully deterministic.

    Matches R:
        svd_C = svd(cov(gAug))
        lambda_aug = svd_C$v %*% (t(svd_C$u) %*% colMeans(gAug) / svd_C$d)

    Parameters
    ----------
    g_aug : (n, P) array

    Returns
    -------
    lam : (P,) array, unit-norm
    """
    mu = np.mean(g_aug, axis=0)           # P
    cov = np.cov(g_aug.T)                 # P x P, ddof=1 matches R cov()
    if cov.ndim == 0:                     # P=1 edge case
        cov = cov.reshape(1, 1)

    U, s, Vh = np.linalg.svd(cov)
    # cov = U @ diag(s) @ Vh  →  cov^{-1} @ mu = Vh.T @ (U.T @ mu / s)
    lam = Vh.T @ (U.T @ mu / s)

    norm = np.sqrt(np.sum(lam ** 2))
    if norm > 1e-12:
        lam /= norm
    return lam


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_stein_moments(samples, derivatives, Q):
    """
    Estimate normalised matrix-valued Stein moments Ahat_1,...,Ahat_Q and bhat.

    For k = 1,...,Q Stein's identity gives:
        E_p[ s(X) (X^{⊙k})^T + k diag(X^{⊙(k-1)}) ] = 0

    Raw empirical moment:
        M_k = (1/n) S^T X^{⊙k}  +  k * diag(mean(X^{⊙(k-1)}))
    with M_1 = (1/n) S^T X + I.

    All moments and bhat are jointly normalised so that
        sum_k ||Ahat_k||_F^2 + ||bhat||^2 = 1.

    Returns
    -------
    Ahat : list of Q arrays, each (d, d)
    bhat : (d,) array
    """
    n, d = samples.shape

    x_pows = [None] * Q          # x_pows[i] = X^{⊙(i+1)} element-wise
    stein_moments = [None] * Q

    x_pows[0] = samples
    stein_moments[0] = (derivatives.T @ x_pows[0]) / n + np.eye(d)

    for k in range(1, Q):       # k in Python = order (k+1) in 1-indexed math
        x_pows[k] = x_pows[k - 1] * samples
        stein_moments[k] = (derivatives.T @ x_pows[k]) / n + \
                           np.diag((k + 1) * np.mean(x_pows[k - 1], axis=0))

    mu_b = np.mean(derivatives, axis=0)

    denom = np.sqrt(
        sum(np.sum(M ** 2) for M in stein_moments) + np.sum(mu_b ** 2)
    )
    if denom < 1e-12:
        denom = 1e-12

    Ahat = [M / denom for M in stein_moments]
    bhat = mu_b / denom
    return Ahat, bhat


def _compute_features(samples, derivatives, Ahat, bhat):
    """
    Compute per-sample projected features g_1(x), ..., g_Q(x), g_b(x).

    For order i (0-indexed, polynomial degree i+1):
        g_{i+1}(x_j) = s_j^T Ahat_i x_j^{⊙(i+1)}
                       + (i+1) * x_j^{⊙i} · diag(Ahat_i)     for i > 0
        g_1(x_j)     = s_j^T Ahat_0 x_j  + trace(Ahat_0)     for i == 0
        g_b(x_j)     = s_j · bhat

    Returns
    -------
    g_aug : (n, Q+1) array
    """
    n, d = samples.shape
    Q = len(Ahat)

    # precompute element-wise powers X^{⊙1}, ..., X^{⊙Q}
    x_pows = [None] * Q
    x_pows[0] = samples
    for k in range(1, Q):
        x_pows[k] = x_pows[k - 1] * samples

    g_mat = np.empty((n, Q))
    for i in range(Q):
        # term1[j] = s_j^T Ahat_i x_j^{⊙(i+1)}
        term1 = np.sum((derivatives @ Ahat[i]) * x_pows[i], axis=1)
        diag_a = np.diag(Ahat[i])
        if i == 0:
            term2 = np.full(n, np.sum(diag_a))
        else:
            term2 = (i + 1) * (x_pows[i - 1] @ diag_a)
        g_mat[:, i] = term1 + term2

    g_b = derivatives @ bhat
    return np.column_stack([g_mat, g_b])


def _studentized_stat(u):
    """Two-sided z-statistic: z = sqrt(n) * mean(u) / std(u, ddof=1)."""
    n = len(u)
    z = np.sqrt(n) * np.mean(u) / np.std(u, ddof=1)
    p = 2 * scipy_stats.norm.sf(np.abs(z))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tpsd(samples, derivatives, Q=1, split_ratio=0.3, seed=None,
         lambda_method='ce'):
    """
    Test-optimised wPSD: adaptively-selected lambda on a held-out training split,
    studentised test statistic on a held-out test split.

    Parameters
    ----------
    samples : (n, d) array
    derivatives : (n, d) array — score ∇log p evaluated at each sample
    Q : int  — polynomial order (number of matrix Stein moments)
    split_ratio : float — fraction of data used for training (train1 + train2)
    seed : int or None
    lambda_method : str — 'svd' (default, fast, deterministic) or 'ce' (stochastic)

    Returns
    -------
    dict with keys:
        'studentised' : {'statistic': float, 'p_value': float}
        'Ahat'        : list of Q (d,d) arrays
        'bhat'        : (d,) array
        'lambda'      : (Q+1,) array  — learned weights
    """
    rng = np.random.RandomState(seed)
    samples = np.asarray(samples, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    n, d = samples.shape

    idx = rng.permutation(n)
    n_train = int(np.floor(split_ratio * n))
    n_test = n - n_train
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]

    # Q=0: score-only direction, no polynomial moments
    if Q == 0:
        deriv_train = derivatives[idx_train]
        deriv_test = derivatives[idx_test]
        mu_b = np.mean(deriv_train, axis=0)
        denom = max(np.sqrt(np.sum(mu_b ** 2)), 1e-12)
        bhat = mu_b / denom
        z, p = _studentized_stat(deriv_test @ bhat)
        return {'unweighted':  {'statistic': z, 'p_value': p},
                'studentised': {'statistic': z, 'p_value': p},
                'sign': int(np.sign(z)),
                'Ahat': [], 'bhat': bhat, 'lambda': np.array([1.0])}

    # Split train into two halves: Ahat estimation vs lambda search
    idx2 = rng.permutation(n_train)
    n_train1 = int(np.floor(0.5 * n_train))
    idx_train1 = idx_train[idx2[:n_train1]]
    idx_train2 = idx_train[idx2[n_train1:]]

    smpl1, deriv1 = samples[idx_train1], derivatives[idx_train1]
    smpl2, deriv2 = samples[idx_train2], derivatives[idx_train2]
    smpl_test = samples[idx_test]
    deriv_test = derivatives[idx_test]

    # Train1: estimate Ahat, bhat
    Ahat, bhat = _build_stein_moments(smpl1, deriv1, Q)

    # Train2: find optimal lambda
    g_aug_train2 = _compute_features(smpl2, deriv2, Ahat, bhat)
    if lambda_method == 'ce':
        lambda_aug = sparse_ce_lambda(g_aug_train2)
    else:
        lambda_aug = svd_lambda(g_aug_train2)

    # Test: studentised statistic with learned lambda
    g_aug_test = _compute_features(smpl_test, deriv_test, Ahat, bhat)
    u = g_aug_test @ lambda_aug
    z, p = _studentized_stat(u)

    return {
        'studentised': {'statistic': z, 'p_value': p},
        'sign': int(np.sign(z)),
        'Ahat': Ahat,
        'bhat': bhat,
        'lambda': lambda_aug,
    }


def psd(samples, derivatives, Q=1, split_ratio=0.2, seed=None):
    """
    Unweighted baseline PSD: uniform lambda, studentised test statistic.

    Same Stein moment features as tpsd, but lambda = ones(Q+1) (no search).

    Parameters
    ----------
    samples : (n, d) array
    derivatives : (n, d) array — score ∇log p evaluated at each sample
    Q : int  — polynomial order
    split_ratio : float — fraction of data used to estimate Ahat
    seed : int or None

    Returns
    -------
    dict with keys:
        'unweighted' : {'statistic': float, 'p_value': float}
        'Ahat'       : list of Q (d,d) arrays
        'bhat'       : (d,) array
        'lambda'     : (Q+1,) array of ones
    """
    rng = np.random.RandomState(seed)
    samples = np.asarray(samples, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    n, d = samples.shape

    idx = rng.permutation(n)
    n_train = int(np.floor(split_ratio * n))
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]

    # Q=0 special case
    if Q == 0:
        deriv_train = derivatives[idx_train]
        deriv_test = derivatives[idx_test]
        mu_b = np.mean(deriv_train, axis=0)
        denom = max(np.sqrt(np.sum(mu_b ** 2)), 1e-12)
        bhat = mu_b / denom
        z, p = _studentized_stat(deriv_test @ bhat)
        return {'unweighted': {'statistic': z, 'p_value': p},
                'Ahat': [], 'bhat': bhat, 'lambda': np.array([1.0])}

    smpl_train = samples[idx_train]
    deriv_train = derivatives[idx_train]
    smpl_test = samples[idx_test]
    deriv_test = derivatives[idx_test]

    # Estimate Ahat, bhat on training set
    Ahat, bhat = _build_stein_moments(smpl_train, deriv_train, Q)

    # Test with uniform lambda
    g_aug_test = _compute_features(smpl_test, deriv_test, Ahat, bhat)
    lambda_unif = np.ones(Q + 1)
    u = g_aug_test @ lambda_unif
    z, p = _studentized_stat(u)

    return {
        'unweighted': {'statistic': z, 'p_value': p},
        'Ahat': Ahat,
        'bhat': bhat,
        'lambda': lambda_unif,
    }
