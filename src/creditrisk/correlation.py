"""Estimate the asset correlation of the one-factor (Vasicek) model from the
loan data itself, instead of assuming a textbook value.

Set-up. Loans are grouped into cohorts by origination period and grade. In
cohort t a single standard normal factor Z_t moves all default probabilities
together:

    p_g(Z) = Phi( (Phi^-1(PD_g) - sqrt(rho) * Z) / sqrt(1 - rho) )

and, given Z_t, the d_gt defaults among n_gt loans are binomial. rho is the
asset correlation. The more default rates swing from cohort to cohort (beyond
what binomial noise explains), the larger rho.

Two estimators are provided:

* ``fit_mle``   maximum likelihood, integrating Z out numerically. Handles
                small cohorts properly because binomial noise is modelled.
* ``fit_mom``   method of moments on joint default frequencies. Simpler and
                noisier; used as an independent cross-check.

Known limits of the approach are discussed in the README: cohorts overlap in
calendar time, grade definitions drift, and the sample window has no deep
recession after 2009.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import config as cfg

Z_LIMIT = 8.0


def z_grid(points: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced grid on [-8, 8] with log weights for integrating against
    the standard normal density (trapezoid rule)."""
    if points < 101 or points % 2 == 0:
        raise ValueError("points must be an odd number >= 101")
    z = np.linspace(-Z_LIMIT, Z_LIMIT, points)
    h = z[1] - z[0]
    w = np.full(points, h)
    w[[0, -1]] = h / 2
    return z, np.log(w) + stats.norm.logpdf(z)


def cohort_table(df: pd.DataFrame, freq: str = "Q",
                 bounds: tuple[str, str] = cfg.RHO_WINDOW) -> pd.DataFrame:
    """Loans and defaults by origination period and grade."""
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    sub = df.loc[(df["issue_date"] >= start) & (df["issue_date"] <= end)]
    period = sub["issue_date"].dt.to_period(freq).astype(str)
    out = (sub.assign(cohort=period.to_numpy())
              .groupby(["cohort", "grade"], observed=True)["default"]
              .agg(n="size", d="sum").reset_index())
    out["rate"] = out["d"] / out["n"]
    return out


def to_matrices(cohorts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Pivot the cohort table into (T x G) arrays of loans and defaults."""
    n = cohorts.pivot(index="cohort", columns="grade", values="n").fillna(0)
    d = cohorts.pivot(index="cohort", columns="grade", values="d").fillna(0)
    return (n.to_numpy(dtype=float), d.to_numpy(dtype=float),
            list(n.index), list(n.columns))


# ---------------------------------------------------------------------------
# Maximum likelihood
# ---------------------------------------------------------------------------
def log_likelihood(c: np.ndarray, rho: float, n: np.ndarray, d: np.ndarray,
                   grid: tuple[np.ndarray, np.ndarray]) -> float:
    """Log-likelihood of cohort default counts under the one-factor model.

    c   (G,)    default thresholds Phi^-1(PD_g)
    n,d (T, G)  loans and defaults per cohort and grade
    """
    z, logw = grid
    x = (c[:, None] - np.sqrt(rho) * z[None, :]) / np.sqrt(1.0 - rho)   # (G, K)
    log_p = special.log_ndtr(x)
    log_q = special.log_ndtr(-x)
    # log binomial likelihood summed over grades, for each cohort and z node
    ll_tk = d @ log_p + (n - d) @ log_q                                   # (T, K)
    const = (special.gammaln(n + 1) - special.gammaln(d + 1)
             - special.gammaln(n - d + 1)).sum()
    return float(special.logsumexp(ll_tk + logw[None, :], axis=1).sum() + const)


def _fit_thresholds(rho: float, n, d, grid, start) -> tuple[np.ndarray, float]:
    res = optimize.minimize(lambda c: -log_likelihood(c, rho, n, d, grid), start,
                            method="L-BFGS-B", options={"maxiter": 2000})
    return res.x, -float(res.fun)


def fit_mle(n: np.ndarray, d: np.ndarray, points: int = 4001,
            confidence: float = 0.95) -> dict:
    """Maximum-likelihood estimate of rho and the grade PDs, with a
    profile-likelihood interval for rho."""
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    if n.shape != d.shape or n.ndim != 2:
        raise ValueError("n and d must be 2-d arrays of the same shape")
    if (d > n).any() or (d < 0).any():
        raise ValueError("defaults must satisfy 0 <= d <= n")
    grid = z_grid(points)
    pooled = np.clip(d.sum(axis=0) / n.sum(axis=0), 1e-4, 1 - 1e-4)
    c0 = stats.norm.ppf(pooled)

    def negative_profile(rho):
        return -_fit_thresholds(rho, n, d, grid, c0)[1]

    opt = optimize.minimize_scalar(negative_profile, bounds=(1e-5, 0.5),
                                   method="bounded", options={"xatol": 1e-7})
    rho_hat = float(opt.x)
    c_hat, ll_hat = _fit_thresholds(rho_hat, n, d, grid, c0)

    # Profile-likelihood interval: all rho whose log-likelihood is within
    # chi2(1)/2 of the maximum.
    drop = stats.chi2.ppf(confidence, df=1) / 2.0

    def gap(rho):
        return ll_hat - _fit_thresholds(rho, n, d, grid, c_hat)[1] - drop

    lo = optimize.brentq(gap, 1e-6, rho_hat, xtol=1e-7) if gap(1e-6) > 0 else 0.0
    hi = optimize.brentq(gap, rho_hat, 0.6, xtol=1e-7) if gap(0.6) > 0 else 0.6

    return {
        "rho": rho_hat,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "pd_by_grade": stats.norm.cdf(c_hat).tolist(),
        "log_likelihood": ll_hat,
        "cohorts": int(n.shape[0]),
        "grid_points": points,
    }


# ---------------------------------------------------------------------------
# Method of moments (cross-check)
# ---------------------------------------------------------------------------
def joint_default_probability(pd_: float, rho: float,
                              grid: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    """P(two loans with the same PD both default) under the one-factor model."""
    z, logw = grid or z_grid()
    x = (stats.norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho)
    return float(np.exp(special.logsumexp(2 * special.log_ndtr(x) + logw)))


def fit_mom(n: np.ndarray, d: np.ndarray) -> dict:
    """Method-of-moments rho for each grade and a loan-weighted average.

    For each grade, d/n estimates PD and d(d-1)/(n(n-1)) is an unbiased
    estimate of the joint default probability of two loans in the same
    cohort. rho is the value that reproduces that joint probability.
    """
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    grid = z_grid()
    rhos, weights = [], []
    for g in range(n.shape[1]):
        ok = n[:, g] >= 2
        if ok.sum() < 2:
            rhos.append(np.nan)
            weights.append(0.0)
            continue
        ng, dg = n[ok, g], d[ok, g]
        pd_hat = float(np.mean(dg / ng))
        jdp_hat = float(np.mean(dg * (dg - 1) / (ng * (ng - 1))))
        if not 0 < pd_hat < 1 or jdp_hat <= pd_hat ** 2:
            rhos.append(0.0)
        else:
            def f(rho):
                return joint_default_probability(pd_hat, rho, grid) - jdp_hat
            rhos.append(float(optimize.brentq(f, 1e-9, 0.99)) if f(0.99) > 0 else 0.99)
        weights.append(float(ng.sum()))
    rhos, weights = np.array(rhos), np.array(weights)
    valid = ~np.isnan(rhos)
    pooled = float(np.average(rhos[valid], weights=weights[valid]))
    return {"rho": pooled, "rho_by_grade": rhos.tolist()}


# ---------------------------------------------------------------------------
# Regulatory reference value
# ---------------------------------------------------------------------------
def basel_other_retail_rho(pd_: np.ndarray | float) -> np.ndarray:
    """Asset correlation the Basel IRB framework prescribes for 'other
    retail' exposures: 16% for the safest borrowers falling to 3% for the
    riskiest. Basel defines it on a one-year PD."""
    pd_ = np.asarray(pd_, dtype=float)
    k = (1 - np.exp(-35.0 * pd_)) / (1 - np.exp(-35.0))
    return 0.03 * k + 0.16 * (1 - k)


def simulate_cohorts(pd_by_grade: np.ndarray, rho: float, n: np.ndarray,
                     seed: int = cfg.SEED) -> np.ndarray:
    """Draw default counts from the model. Used by the tests to confirm the
    estimators recover a known rho."""
    rng = np.random.default_rng(seed)
    n = np.asarray(n, dtype=np.int64)
    z = rng.standard_normal(n.shape[0])
    c = stats.norm.ppf(np.asarray(pd_by_grade, dtype=float))
    p = stats.norm.cdf((c[None, :] - np.sqrt(rho) * z[:, None]) / np.sqrt(1 - rho))
    return rng.binomial(n, p).astype(float)
