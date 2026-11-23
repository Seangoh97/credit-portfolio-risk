"""Portfolio credit loss distribution under the one-factor (Vasicek) model.

Each loan i has a default probability PD_i and a dollar loss if it defaults,
L_i. A loan defaults when its latent "asset value"

    A_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i

falls below Phi^-1(PD_i). Z is a standard normal factor shared by every loan
(the state of the economy); eps_i is standard normal and specific to loan i.
rho, the asset correlation, controls how strongly loans default together.

Three independent ways of getting the loss distribution are implemented so
that they can be checked against one another:

* ``simulate``              Monte Carlo, loan by loan. Exact up to sampling
                            error.
* ``semi_analytic``         integrates over Z and treats the loan-specific
                            noise as normal (central limit theorem). Fast.
* ``asrf_quantile``         the textbook large-portfolio limit that ignores
                            loan-specific noise altogether (the formula
                            behind Basel capital).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, optimize, special, stats

from . import config as cfg


def _validate(pd_: np.ndarray, loss_if_default: np.ndarray, rho: float):
    pd_ = np.asarray(pd_, dtype=float)
    lid = np.asarray(loss_if_default, dtype=float)
    if pd_.ndim != 1 or pd_.shape != lid.shape:
        raise ValueError("pd and loss_if_default must be 1-d arrays of equal length")
    if not ((pd_ > 0) & (pd_ < 1)).all():
        raise ValueError("every PD must lie strictly between 0 and 1")
    if not np.isfinite(lid).all() or (lid < 0).any():
        raise ValueError("loss_if_default must be finite and non-negative")
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")
    return pd_, lid


def conditional_pd(pd_: np.ndarray, rho: float, z: np.ndarray | float) -> np.ndarray:
    """Default probability given the common factor Z = z.

    Z below zero is a bad economy, so the conditional PD rises as z falls.
    Returns an array of shape (len(z), len(pd)) for array inputs.
    """
    c = stats.norm.ppf(np.asarray(pd_, dtype=float))
    z = np.asarray(z, dtype=float)
    return special.ndtr((c - np.sqrt(rho) * z[..., None]) / np.sqrt(1.0 - rho))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class Simulation:
    losses: np.ndarray            # (S,) portfolio loss per scenario
    z: np.ndarray                 # (S,) common factor per scenario
    segment_losses: np.ndarray    # (S, G) loss per scenario and segment
    segment_labels: list
    n_loans: int
    rho: float


def stratified_normals(n: int, rng: np.random.Generator) -> np.ndarray:
    """n standard normal draws, one from each of n equal-probability strata.

    With a single common factor this removes most of the sampling error in
    the tail: every part of the Z distribution, including the worst 0.1%, is
    represented in proportion.
    """
    u = (np.arange(n) + rng.random(n)) / n
    return stats.norm.ppf(np.clip(u, 1e-15, 1 - 1e-15))


def simulate(pd_: np.ndarray, loss_if_default: np.ndarray, rho: float,
             n_scenarios: int = 20_000, seed: int = cfg.SEED,
             segments: np.ndarray | None = None, stratified: bool = True,
             max_cells: int = 12_000_000) -> Simulation:
    """Simulate the portfolio loss distribution loan by loan.

    segments    optional label per loan; losses are also tracked per segment
                so that tail losses can be attributed afterwards.
    max_cells   memory cap: scenarios are processed in chunks of at most
                max_cells / n_loans. The chunk size changes memory use
                only; the random draws are the same for any chunk size.
    """
    pd_, lid = _validate(pd_, loss_if_default, rho)
    n = len(pd_)
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be positive")

    if segments is None:
        labels, codes = ["all"], np.zeros(n, dtype=np.int64)
    else:
        segments = np.asarray(segments)
        if segments.shape != pd_.shape:
            raise ValueError("segments must have one label per loan")
        labels, codes = np.unique(segments, return_inverse=True)
        labels = labels.tolist()
    # Sort loans by segment so each segment is one contiguous slice.
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(labels) + 1))
    c_sorted = stats.norm.ppf(pd_[order])
    lid_sorted = lid[order]

    z_seed, eps_seed = np.random.SeedSequence(seed).spawn(2)
    z_rng, eps_rng = np.random.default_rng(z_seed), np.random.default_rng(eps_seed)
    z = stratified_normals(n_scenarios, z_rng) if stratified else z_rng.standard_normal(n_scenarios)

    sqrt_rho, sqrt_1m = np.sqrt(rho), np.sqrt(1.0 - rho)
    threshold_base = (c_sorted / sqrt_1m).astype(np.float32)
    seg_losses = np.empty((n_scenarios, len(labels)))
    chunk = max(1, int(max_cells // n))
    for start in range(0, n_scenarios, chunk):
        stop = min(start + chunk, n_scenarios)
        shift = (sqrt_rho / sqrt_1m * z[start:stop]).astype(np.float32)
        eps = eps_rng.standard_normal((stop - start, n), dtype=np.float32)
        defaulted = eps < (threshold_base[None, :] - shift[:, None])
        for g in range(len(labels)):
            lo, hi = bounds[g], bounds[g + 1]
            seg_losses[start:stop, g] = defaulted[:, lo:hi] @ lid_sorted[lo:hi]
    return Simulation(losses=seg_losses.sum(axis=1), z=z, segment_losses=seg_losses,
                      segment_labels=labels, n_loans=n, rho=rho)


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------
def tail_count(n_scenarios: int, level: float) -> int:
    """Number of scenarios in the worst (1 - level) share of outcomes."""
    return max(1, int(round(n_scenarios * (1.0 - level))))


def value_at_risk(losses: np.ndarray, level: float) -> float:
    """Loss exceeded in only a (1 - level) share of scenarios."""
    return float(np.quantile(np.asarray(losses), level, method="inverted_cdf"))


def expected_shortfall(losses: np.ndarray, level: float) -> float:
    """Average loss in the worst (1 - level) share of scenarios."""
    losses = np.asarray(losses)
    k = tail_count(len(losses), level)
    return float(np.sort(losses)[-k:].mean())


def risk_measures(losses: np.ndarray, levels=(0.95, 0.99, 0.999)) -> dict:
    losses = np.asarray(losses, dtype=float)
    out = {"expected_loss": float(losses.mean()), "loss_std": float(losses.std(ddof=1))}
    for q in levels:
        var = value_at_risk(losses, q)
        out[f"var_{q}"] = var
        out[f"es_{q}"] = expected_shortfall(losses, q)
        out[f"capital_{q}"] = var - out["expected_loss"]
    return out


def tail_contributions(sim: Simulation, level: float) -> dict:
    """Split expected shortfall across segments: the average loss of each
    segment in the worst (1 - level) share of scenarios. The pieces add up to
    the portfolio's expected shortfall exactly."""
    k = tail_count(len(sim.losses), level)
    worst = np.argsort(sim.losses, kind="stable")[-k:]
    contrib = sim.segment_losses[worst].mean(axis=0)
    return dict(zip(sim.segment_labels, contrib.tolist()))


# ---------------------------------------------------------------------------
# Analytic cross-checks
# ---------------------------------------------------------------------------
def asrf_quantile(pd_: np.ndarray, loss_if_default: np.ndarray, rho: float,
                  level: float) -> float:
    """Loss quantile for an infinitely fine-grained portfolio (the
    'asymptotic single risk factor' result used in Basel capital rules)."""
    pd_, lid = _validate(pd_, loss_if_default, rho)
    z_bad = -stats.norm.ppf(level)
    return float(lid @ conditional_pd(pd_, rho, np.array([z_bad]))[0])


@dataclass
class SemiAnalytic:
    """Loss distribution from numerical integration over Z, with the
    loan-specific noise at each Z approximated by a normal distribution."""
    z: np.ndarray
    weight: np.ndarray
    mean: np.ndarray      # E[loss | Z = z]
    std: np.ndarray       # sd[loss | Z = z]

    @property
    def expected_loss(self) -> float:
        return float(self.weight @ self.mean)

    def cdf(self, x: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(self.std > 0, (x - self.mean) / self.std,
                         np.where(x >= self.mean, np.inf, -np.inf))
        return float(self.weight @ special.ndtr(a))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density of the portfolio loss at each value in x."""
        x = np.asarray(x, dtype=float)
        std = np.maximum(self.std, 1e-12)
        a = (x[:, None] - self.mean[None, :]) / std[None, :]
        return (stats.norm.pdf(a) / std[None, :]) @ self.weight

    def quantile(self, level: float) -> float:
        lo = float(self.mean.min() - 10 * self.std.max() - 1.0)
        hi = float(self.mean.max() + 10 * self.std.max() + 1.0)
        return float(optimize.brentq(lambda x: self.cdf(x) - level, lo, hi, xtol=1e-6 * max(hi, 1.0)))

    def expected_shortfall(self, level: float) -> float:
        var = self.quantile(level)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(self.std > 0, (var - self.mean) / self.std,
                         np.where(var >= self.mean, np.inf, -np.inf))
        partial = self.mean * special.ndtr(-a) + self.std * stats.norm.pdf(a)
        return float(self.weight @ partial / (1.0 - level))

    def risk_measures(self, levels=(0.95, 0.99, 0.999)) -> dict:
        out = {"expected_loss": self.expected_loss}
        for q in levels:
            out[f"var_{q}"] = self.quantile(q)
            out[f"es_{q}"] = self.expected_shortfall(q)
            out[f"capital_{q}"] = out[f"var_{q}"] - out["expected_loss"]
        return out


def semi_analytic(pd_: np.ndarray, loss_if_default: np.ndarray, rho: float,
                  coarse_points: int = 161, fine_points: int = 6401) -> SemiAnalytic:
    """Build the semi-analytic loss distribution.

    The conditional mean and variance of the loss are smooth in Z, so they
    are computed exactly on a coarse grid and interpolated onto a fine one.
    """
    pd_, lid = _validate(pd_, loss_if_default, rho)
    z_coarse = np.linspace(-8.0, 8.0, coarse_points)
    mean = np.empty(coarse_points)
    var = np.empty(coarse_points)
    lid_sq = lid ** 2
    block = max(1, int(4_000_000 // len(pd_)))
    for start in range(0, coarse_points, block):
        p = conditional_pd(pd_, rho, z_coarse[start:start + block])
        mean[start:start + block] = p @ lid
        var[start:start + block] = (p * (1 - p)) @ lid_sq

    z = np.linspace(-8.0, 8.0, fine_points)
    h = z[1] - z[0]
    weight = stats.norm.pdf(z) * h
    weight[[0, -1]] *= 0.5
    weight /= weight.sum()
    mean_f = interpolate.CubicSpline(z_coarse, mean)(z)
    var_f = np.clip(interpolate.CubicSpline(z_coarse, var)(z), 0.0, None)
    return SemiAnalytic(z=z, weight=weight, mean=mean_f, std=np.sqrt(var_f))


def default_correlation(pd_: float, rho: float) -> float:
    """Correlation between the default indicators of two loans with the same
    PD. Much smaller than the asset correlation rho that drives it."""
    from .correlation import joint_default_probability
    if rho == 0:
        return 0.0
    jdp = joint_default_probability(pd_, rho)
    return float((jdp - pd_ ** 2) / (pd_ * (1 - pd_)))
