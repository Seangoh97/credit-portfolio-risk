"""Checks on the PD models, recalibration and metrics (synthetic data)."""
import numpy as np
import pandas as pd
import pytest
from scipy import special
from sklearn.metrics import roc_auc_score

from creditrisk import pd_model as pm

FEATURES = ["fico_range_low", "dti", "annual_inc", "purpose"]


def synthetic_loans(n=8_000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    fico = rng.normal(700, 30, n)
    dti = rng.uniform(0, 35, n)
    income = rng.lognormal(11, 0.5, n)
    purpose = rng.choice(["car", "credit_card", "small_business"], n, p=[0.3, 0.5, 0.2])
    eta = -1.9 + shift - 0.025 * (fico - 700) + 0.04 * (dti - 17) + 0.6 * (purpose == "small_business")
    y = rng.random(n) < special.expit(eta)
    dates = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 720, n), unit="D")
    df = pd.DataFrame({"fico_range_low": fico, "dti": dti, "annual_inc": income,
                       "purpose": purpose, "default": y.astype(int),
                       "issue_date": dates.to_period("M").to_timestamp(),
                       "sub_grade_num": np.clip(np.round((760 - fico) / 4), 1, 35)})
    df.loc[rng.random(n) < 0.05, "dti"] = np.nan
    return df


def test_one_dimensional_logistic_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 2.0, 200_000)
    y = (rng.random(x.size) < special.expit(-2.0 + 0.7 * x)).astype(float)
    a, b = pm.fit_logistic_1d(x, y)
    assert a == pytest.approx(-2.0, abs=0.05) and b == pytest.approx(0.7, abs=0.02)


def test_one_dimensional_logistic_fit_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        pm.fit_logistic_1d(np.ones(10), np.array([0, 1] * 5))
    with pytest.raises(ValueError):
        pm.fit_logistic_1d(np.arange(10.0), np.zeros(10))
    with pytest.raises(ValueError):
        pm.fit_logistic_1d(np.array([1.0, np.nan, 2.0, 3.0]), np.array([0, 1, 0, 1]))


def test_recalibration_fixes_the_level_and_keeps_the_ranking():
    rng = np.random.default_rng(2)
    true_p = rng.beta(2, 10, 100_000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    raw = special.expit(0.6 * pm.logit(true_p) - 1.0)          # distorted score
    assert abs(raw.mean() - y.mean()) > 0.03
    recal = pm.Recalibration.fit(raw, y)
    p = recal.apply(raw)
    assert p.mean() == pytest.approx(y.mean(), abs=1e-6)       # level matched exactly
    assert pm.auc(y, p) == pytest.approx(pm.auc(y, raw), abs=1e-12)
    assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(raw, kind="stable"))
    assert np.abs(p - true_p).mean() < 0.01                    # undoes the distortion


def test_recalibration_refuses_a_score_that_ranks_backwards():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.6, 20_000)
    y = (rng.random(p.size) < p).astype(int)
    with pytest.raises(RuntimeError):
        pm.Recalibration.fit(1 - p, y)


def test_auc_matches_scikit_learn_including_ties():
    rng = np.random.default_rng(4)
    y = rng.random(5_000) < 0.2
    score = np.round(rng.normal(size=5_000) + y, 1)            # many ties
    assert pm.auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)
    assert pm.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    with pytest.raises(ValueError):
        pm.auc(np.zeros(5), np.arange(5.0))


def test_ks_statistic_bounds():
    assert pm.ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(5)
    assert pm.ks_statistic(rng.random(50_000) < 0.2, rng.random(50_000)) < 0.03


def test_evaluate_and_calibration_table():
    rng = np.random.default_rng(6)
    p = rng.uniform(0.01, 0.5, 50_000)
    y = (rng.random(p.size) < p).astype(int)
    m = pm.evaluate(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert m["mean_pd"] == pytest.approx(m["default_rate"], abs=0.01)
    table = pm.calibration_table(y, p, bins=10)
    assert table["loans"].sum() == p.size and len(table) == 10
    assert table["predicted"].is_monotonic_increasing
    assert np.abs(table["predicted"] - table["observed"]).max() < 0.02


def test_models_learn_the_signal_and_are_comparable():
    train, test = synthetic_loans(8_000, seed=7), synthetic_loans(8_000, seed=8)
    y = test["default"].to_numpy()
    gbm = pm.GBMModel(features=FEATURES).fit(train)
    logit_model = pm.LogitModel(features=FEATURES).fit(train)
    grade = pm.GradeModel().fit(train)
    for model in (gbm, logit_model, grade):
        p = model.predict(test)
        assert p.shape == (len(test),) and ((p > 0) & (p < 1)).all()
        assert pm.auc(y, p) > 0.65
    # The full models see dti and purpose, which the grade proxy does not.
    assert pm.auc(y, logit_model.predict(test)) > pm.auc(y, grade.predict(test))
    assert gbm.holdout_start_ > train["issue_date"].min()
    importance = gbm.shap_importance(test)
    assert set(importance.index) == set(FEATURES) and (importance >= 0).all()


def test_gbm_is_deterministic_and_handles_unseen_categories():
    train, test = synthetic_loans(5_000, seed=9), synthetic_loans(500, seed=10)
    a = pm.GBMModel(features=FEATURES).fit(train).predict(test)
    b = pm.GBMModel(features=FEATURES).fit(train).predict(test)
    np.testing.assert_array_equal(a, b)
    odd = test.copy()
    odd["purpose"] = "never_seen_before"
    for model in (pm.GBMModel(features=FEATURES).fit(train), pm.LogitModel(features=FEATURES).fit(train)):
        p = model.predict(odd)
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_gbm_can_be_trained_on_another_target_column():
    train, test = synthetic_loans(8_000, seed=15), synthetic_loans(4_000, seed=16)
    rng = np.random.default_rng(0)
    # an "early default" style label: a random half of the defaults
    train["early_default"] = train["default"] * (rng.random(len(train)) < 0.5)
    model = pm.GBMModel(features=FEATURES, target="early_default").fit(train)
    p = model.predict(test)
    assert pm.auc(test["default"].to_numpy(), p) > 0.65          # still ranks lifetime default
    assert p.mean() < 0.75 * test["default"].mean()              # but at the early-default level


def test_recalibration_corrects_a_shift_in_default_rates():
    train = synthetic_loans(8_000, seed=11)
    calib, test = synthetic_loans(8_000, seed=12, shift=0.5), synthetic_loans(8_000, seed=13, shift=0.5)
    model = pm.LogitModel(features=FEATURES).fit(train)
    raw = model.predict(test)
    recal = pm.Recalibration.fit(model.predict(calib), calib["default"].to_numpy())
    gap_before = abs(raw.mean() - test["default"].mean())
    gap_after = abs(recal.apply(raw).mean() - test["default"].mean())
    assert gap_before > 0.03 and gap_after < 0.015


def test_bootstrap_auc_difference():
    rng = np.random.default_rng(14)
    y = rng.random(20_000) < 0.2
    strong, weak = rng.normal(size=y.size) + 1.0 * y, rng.normal(size=y.size) + 0.3 * y
    out = pm.bootstrap_auc_difference(y, strong, weak, n_boot=100)
    assert out["ci_low"] < out["difference"] < out["ci_high"] and out["ci_low"] > 0
