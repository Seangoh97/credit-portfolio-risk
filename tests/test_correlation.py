"""Checks on the asset-correlation estimators, using data simulated from the
model with a known rho."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from creditrisk import correlation as corr


def simulated_panel(rho, cohorts, seed, loans=(400, 3_000)):
    rng = np.random.default_rng(seed)
    pds = np.array([0.05, 0.12, 0.25])
    n = rng.integers(loans[0], loans[1], size=(cohorts, len(pds)))
    d = corr.simulate_cohorts(pds, rho, n, seed=seed + 1)
    return pds, n.astype(float), d


@pytest.mark.parametrize("rho", [0.01, 0.06])
def test_mle_recovers_known_rho(rho):
    pds, n, d = simulated_panel(rho, cohorts=150, seed=10)
    fit = corr.fit_mle(n, d, points=2001)
    assert fit["rho"] == pytest.approx(rho, rel=0.30)
    assert fit["ci_low"] <= rho <= fit["ci_high"]
    assert fit["ci_low"] < fit["rho"] < fit["ci_high"]
    np.testing.assert_allclose(fit["pd_by_grade"], pds, rtol=0.08)


def test_mle_reports_near_zero_rho_when_defaults_are_independent():
    rng = np.random.default_rng(3)
    n = rng.integers(500, 3_000, size=(60, 3))
    d = rng.binomial(n, np.array([0.05, 0.12, 0.25])).astype(float)
    fit = corr.fit_mle(n.astype(float), d, points=2001)
    assert fit["rho"] < 0.001
    assert fit["ci_low"] == 0.0


def test_mom_recovers_known_rho():
    _, n, d = simulated_panel(0.06, cohorts=400, seed=20)
    assert corr.fit_mom(n, d)["rho"] == pytest.approx(0.06, rel=0.30)


def test_likelihood_is_stable_when_the_integration_grid_is_refined():
    pds, n, d = simulated_panel(0.005, cohorts=30, seed=30, loans=(5_000, 40_000))
    c = stats.norm.ppf(pds)
    coarse = corr.log_likelihood(c, 0.005, n, d, corr.z_grid(4001))
    fine = corr.log_likelihood(c, 0.005, n, d, corr.z_grid(16001))
    assert coarse == pytest.approx(fine, abs=1e-6)


def test_likelihood_reduces_to_binomial_when_rho_is_tiny():
    n = np.array([[1_000.0, 500.0]])
    d = np.array([[80.0, 90.0]])
    pds = np.array([0.08, 0.2])
    ours = corr.log_likelihood(stats.norm.ppf(pds), 1e-12, n, d, corr.z_grid(2001))
    binom = stats.binom.logpmf(d, n, pds).sum()
    assert ours == pytest.approx(binom, abs=1e-4)


def test_joint_default_probability_against_bivariate_normal():
    p = 0.1
    assert corr.joint_default_probability(p, 1e-12) == pytest.approx(p ** 2, rel=1e-5)
    for rho in (0.05, 0.3):
        c = stats.norm.ppf(p)
        exact = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([c, c])
        assert corr.joint_default_probability(p, rho) == pytest.approx(exact, rel=1e-4)


def test_basel_correlation_limits():
    assert corr.basel_other_retail_rho(0.0) == pytest.approx(0.16)
    assert corr.basel_other_retail_rho(1.0) == pytest.approx(0.03, abs=1e-12)
    values = corr.basel_other_retail_rho(np.array([0.01, 0.05, 0.2]))
    assert (np.diff(values) < 0).all()


def test_cohort_table_counts():
    df = pd.DataFrame({
        "issue_date": pd.to_datetime(["2013-01-01", "2013-02-01", "2013-05-01",
                                      "2013-05-01", "2016-01-01"]),
        "grade": ["A", "A", "A", "B", "A"],
        "default": [1, 0, 0, 1, 1],
    })
    table = corr.cohort_table(df, "Q", ("2013-01-01", "2013-12-31"))
    assert len(table) == 3 and table["n"].sum() == 4          # 2016 loan excluded
    row = table[(table["cohort"] == "2013Q1") & (table["grade"] == "A")].iloc[0]
    assert (row["n"], row["d"]) == (2, 1)
    n, d, cohorts, grades = corr.to_matrices(table)
    assert n.shape == (2, 2) and cohorts == ["2013Q1", "2013Q2"] and grades == ["A", "B"]
    assert n[0, 1] == 0                                        # no B loans in Q1


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        corr.z_grid(100)
    with pytest.raises(ValueError):
        corr.fit_mle(np.array([[10.0]]), np.array([[11.0]]))
    with pytest.raises(ValueError):
        corr.fit_mle(np.array([10.0]), np.array([1.0]))
