"""Checks on the loss engine. Every test uses synthetic portfolios, so the
suite runs without the Lending Club file."""
import numpy as np
import pytest
from scipy import stats

from creditrisk import portfolio as pf
from creditrisk.correlation import z_grid


def make_portfolio(n, seed=0):
    rng = np.random.default_rng(seed)
    pd_ = np.clip(rng.beta(2, 12, n), 0.005, 0.6)
    lid = rng.uniform(2_000, 20_000, n)
    return pd_, lid


def test_conditional_pd_averages_back_to_unconditional_pd():
    pd_ = np.array([0.01, 0.05, 0.15, 0.40])
    z, logw = z_grid(4001)
    for rho in (0.005, 0.05, 0.3):
        averaged = np.exp(logw) @ pf.conditional_pd(pd_, rho, z)
        np.testing.assert_allclose(averaged, pd_, rtol=1e-8)


def test_conditional_pd_rises_in_bad_states():
    p = pf.conditional_pd(np.array([0.1]), 0.1, np.array([-2.0, 0.0, 2.0]))[:, 0]
    assert p[0] > p[1] > p[2]


def test_engine_matches_direct_simulation_of_asset_values():
    """The engine never builds asset values explicitly. Compare it with a
    literal implementation of the model definition."""
    pd_, lid = make_portfolio(40, seed=1)
    rho, n_sim = 0.2, 400_000
    sim = pf.simulate(pd_, lid, rho, n_scenarios=n_sim, seed=3, stratified=False)

    rng = np.random.default_rng(99)
    z = rng.standard_normal((n_sim, 1))
    eps = rng.standard_normal((n_sim, len(pd_)))
    assets = np.sqrt(rho) * z + np.sqrt(1 - rho) * eps
    direct = (assets < stats.norm.ppf(pd_)) @ lid

    assert sim.losses.mean() == pytest.approx(direct.mean(), rel=0.01)
    assert sim.losses.std() == pytest.approx(direct.std(), rel=0.01)
    for q in (0.9, 0.99, 0.999):
        assert np.quantile(sim.losses, q) == pytest.approx(np.quantile(direct, q), rel=0.03)


def test_expected_loss_equals_sum_of_pd_times_loss():
    pd_, lid = make_portfolio(3_000, seed=2)
    sim = pf.simulate(pd_, lid, 0.05, n_scenarios=20_000, seed=5)
    assert sim.losses.mean() == pytest.approx(pd_ @ lid, rel=0.002)


def test_zero_correlation_gives_independent_defaults():
    pd_, lid = make_portfolio(500, seed=3)
    sim = pf.simulate(pd_, lid, 0.0, n_scenarios=100_000, seed=7)
    assert sim.losses.var() == pytest.approx((lid ** 2 * pd_ * (1 - pd_)).sum(), rel=0.02)


def test_large_homogeneous_portfolio_matches_vasicek_formula():
    n, p, rho = 20_000, 0.10, 0.15
    sim = pf.simulate(np.full(n, p), np.ones(n), rho, n_scenarios=20_000, seed=11)
    for q in (0.99, 0.999):
        closed_form = stats.norm.cdf((stats.norm.ppf(p) + np.sqrt(rho) * stats.norm.ppf(q))
                                     / np.sqrt(1 - rho))
        assert pf.value_at_risk(sim.losses, q) / n == pytest.approx(closed_form, rel=0.02)
        assert pf.asrf_quantile(np.full(n, p), np.ones(n), rho, q) / n == pytest.approx(closed_form, rel=1e-10)


def test_semi_analytic_agrees_with_monte_carlo():
    pd_, lid = make_portfolio(4_000, seed=4)
    for rho in (0.005, 0.05):
        sim = pf.simulate(pd_, lid, rho, n_scenarios=40_000, seed=13)
        mc = pf.risk_measures(sim.losses)
        sa = pf.semi_analytic(pd_, lid, rho).risk_measures()
        assert sa["expected_loss"] == pytest.approx(pd_ @ lid, rel=1e-6)
        for key in ("var_0.99", "var_0.999", "es_0.99", "es_0.999"):
            assert sa[key] == pytest.approx(mc[key], rel=0.015), (rho, key)


def test_semi_analytic_density_integrates_to_one_and_matches_cdf():
    pd_, lid = make_portfolio(2_000, seed=10)
    sa = pf.semi_analytic(pd_, lid, 0.03)
    x = np.linspace(0, sa.quantile(0.99999), 4001)
    density = sa.pdf(x)
    assert (density >= 0).all()
    assert np.trapezoid(density, x) == pytest.approx(1.0, abs=1e-3)
    # The integral of the density up to a point equals the distribution function there.
    mid = sa.quantile(0.5)
    upto = x <= mid
    assert np.trapezoid(density[upto], x[upto]) == pytest.approx(sa.cdf(x[upto][-1]), abs=2e-3)


def test_granular_portfolio_needs_more_capital_than_asrf_limit():
    pd_, lid = make_portfolio(2_000, seed=5)
    rho = 0.03
    assert pf.semi_analytic(pd_, lid, rho).quantile(0.999) > pf.asrf_quantile(pd_, lid, rho, 0.999)


def test_tail_contributions_add_up_to_expected_shortfall():
    pd_, lid = make_portfolio(1_500, seed=6)
    segments = np.random.default_rng(0).choice(list("ABC"), len(pd_))
    sim = pf.simulate(pd_, lid, 0.04, n_scenarios=10_000, seed=17, segments=segments)
    np.testing.assert_allclose(sim.segment_losses.sum(axis=1), sim.losses)
    for q in (0.99, 0.999):
        parts = pf.tail_contributions(sim, q)
        assert set(parts) == {"A", "B", "C"}
        assert sum(parts.values()) == pytest.approx(pf.expected_shortfall(sim.losses, q), rel=1e-12)


def test_segment_expected_losses_are_correct():
    pd_, lid = make_portfolio(3_000, seed=7)
    segments = np.where(np.arange(len(pd_)) % 3 == 0, "x", "y")
    sim = pf.simulate(pd_, lid, 0.03, n_scenarios=20_000, seed=19, segments=segments)
    for j, label in enumerate(sim.segment_labels):
        mask = segments == label
        assert sim.segment_losses[:, j].mean() == pytest.approx(pd_[mask] @ lid[mask], rel=0.01)


def test_results_do_not_depend_on_chunk_size():
    pd_, lid = make_portfolio(300, seed=8)
    a = pf.simulate(pd_, lid, 0.05, n_scenarios=2_000, seed=23, max_cells=300 * 2_000)
    b = pf.simulate(pd_, lid, 0.05, n_scenarios=2_000, seed=23, max_cells=300 * 7)
    # Same random draws; only the order of floating-point additions differs.
    np.testing.assert_allclose(a.losses, b.losses, rtol=1e-12)


def test_seed_controls_reproducibility():
    pd_, lid = make_portfolio(300, seed=9)
    a = pf.simulate(pd_, lid, 0.05, n_scenarios=1_000, seed=1)
    b = pf.simulate(pd_, lid, 0.05, n_scenarios=1_000, seed=1)
    c = pf.simulate(pd_, lid, 0.05, n_scenarios=1_000, seed=2)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert not np.array_equal(a.losses, c.losses)


def test_stratified_draws_are_standard_normal_and_cover_every_stratum():
    z = pf.stratified_normals(10_000, np.random.default_rng(0))
    u = stats.norm.cdf(z)
    assert np.array_equal(np.floor(u * 10_000).astype(int), np.arange(10_000))
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01


def test_risk_measures_are_ordered():
    losses = np.random.default_rng(0).gamma(2.0, 100.0, 50_000)
    m = pf.risk_measures(losses)
    assert m["var_0.95"] <= m["var_0.99"] <= m["var_0.999"]
    for q in (0.95, 0.99, 0.999):
        assert m[f"es_{q}"] >= m[f"var_{q}"]
        assert m[f"capital_{q}"] == pytest.approx(m[f"var_{q}"] - m["expected_loss"])


def test_var_and_es_on_a_known_sample():
    losses = np.arange(1, 1001, dtype=float)          # 1, 2, ..., 1000
    assert pf.value_at_risk(losses, 0.99) == 990.0
    assert pf.expected_shortfall(losses, 0.99) == pytest.approx(np.mean(np.arange(991, 1001)))


def test_default_correlation_matches_simulation():
    p, rho = 0.12, 0.10
    rng = np.random.default_rng(5)
    z = rng.standard_normal(2_000_000)
    a1 = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(z.size)
    a2 = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(z.size)
    c = stats.norm.ppf(p)
    empirical = np.corrcoef(a1 < c, a2 < c)[0, 1]
    assert pf.default_correlation(p, rho) == pytest.approx(empirical, abs=0.002)
    assert pf.default_correlation(p, 0.0) == 0.0


@pytest.mark.parametrize("pd_, lid, rho", [
    (np.array([0.1, 1.0]), np.array([1.0, 1.0]), 0.1),      # PD of exactly 1
    (np.array([0.1, 0.0]), np.array([1.0, 1.0]), 0.1),      # PD of exactly 0
    (np.array([0.1, 0.2]), np.array([1.0, -1.0]), 0.1),     # negative loss
    (np.array([0.1, 0.2]), np.array([1.0, np.nan]), 0.1),   # missing loss
    (np.array([0.1, 0.2]), np.array([1.0]), 0.1),           # length mismatch
    (np.array([0.1, 0.2]), np.array([1.0, 1.0]), 1.0),      # rho out of range
    (np.array([0.1, 0.2]), np.array([1.0, 1.0]), -0.1),
])
def test_bad_inputs_are_rejected(pd_, lid, rho):
    with pytest.raises(ValueError):
        pf.simulate(pd_, lid, rho, n_scenarios=10)
    with pytest.raises(ValueError):
        pf.semi_analytic(pd_, lid, rho)
