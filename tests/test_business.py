"""Checks on the return parameters and policy arithmetic."""
import numpy as np
import pandas as pd
import pytest

from creditrisk import business as biz
from creditrisk import config as cfg


def history():
    rows = []
    for grade in cfg.GRADES:
        # two repaid loans and one default per grade, $10,000 each
        rows.append(dict(grade=grade, default=0, funded_amnt=10_000.0, installment=320.0,
                         net_pnl=1_200.0, loss_amount=0.0))
        rows.append(dict(grade=grade, default=0, funded_amnt=10_000.0, installment=320.0,
                         net_pnl=1_000.0, loss_amount=0.0))
        rows.append(dict(grade=grade, default=1, funded_amnt=10_000.0, installment=320.0,
                         net_pnl=-4_000.0, loss_amount=5_500.0))
    return pd.DataFrame(rows)


def test_return_parameters():
    params = biz.ReturnParams.fit(history())
    assert set(params.severity) == {"A", "B", "C", "D", "E", "FG"}
    assert params.defaults_used["FG"] == 2 and params.defaults_used["A"] == 1
    scheduled = 320.0 * 36 - 10_000                            # 1,520 per loan
    for group in params.severity:
        assert params.severity[group] == pytest.approx(0.55)
        assert params.return_if_default[group] == pytest.approx(-0.40)
        assert params.interest_realisation[group] == pytest.approx(2_200 / (2 * scheduled))


def test_loss_if_default_and_expected_return():
    hist = history()
    params = biz.ReturnParams.fit(hist)
    loans = hist.iloc[:3]
    np.testing.assert_allclose(params.loss_if_default(loans), 5_500.0)
    ret_paid = (2_200 / (2 * 1_520)) * 1_520 / 10_000          # = 0.11
    np.testing.assert_allclose(params.expected_return(loans, np.zeros(3)), ret_paid)
    np.testing.assert_allclose(params.expected_return(loans, np.ones(3)), -0.40)
    np.testing.assert_allclose(params.expected_return(loans, np.full(3, 0.2)),
                               0.8 * ret_paid + 0.2 * -0.40)


def test_fit_fails_when_a_grade_has_no_defaults():
    hist = history()
    hist = hist[~((hist["grade"] == "A") & (hist["default"] == 1))]
    with pytest.raises(ValueError):
        biz.ReturnParams.fit(hist)


def test_unknown_grade_is_rejected():
    hist = history()
    params = biz.ReturnParams.fit(hist)
    with pytest.raises(ValueError):
        params.loss_if_default(hist.iloc[:2].assign(grade="Z"))


def test_decline_riskiest_picks_exactly_the_highest_pds():
    pd_ = np.array([0.05, 0.30, 0.10, 0.30, 0.20, 0.01, 0.15, 0.02, 0.03, 0.04])
    approve = biz.decline_riskiest(pd_, 0.3)
    assert (~approve).sum() == 3
    assert set(np.flatnonzero(~approve)) == {1, 3, 4}
    assert biz.decline_riskiest(pd_, 0.0).all()
    # Ties: exactly one of the two 0.30 loans is declined, and the choice is
    # reproducible for a given seed.
    one = biz.decline_riskiest(pd_, 0.1, seed=1)
    assert (~one).sum() == 1 and np.flatnonzero(~one)[0] in (1, 3)
    assert np.array_equal(one, biz.decline_riskiest(pd_, 0.1, seed=1))
    picks = {int(np.flatnonzero(~biz.decline_riskiest(pd_, 0.1, seed=s))[0]) for s in range(30)}
    assert picks == {1, 3}
    with pytest.raises(ValueError):
        biz.decline_riskiest(pd_, 1.0)


def test_policy_outcome_and_tightening_curve():
    hist = history()
    full = biz.policy_outcome(hist, np.ones(len(hist), dtype=bool))
    assert full["loans"] == 21 and full["funded"] == 210_000
    assert full["net_pnl"] == pytest.approx(7 * (1_200 + 1_000 - 4_000))
    assert full["principal_loss"] == pytest.approx(7 * 5_500)
    assert full["default_rate"] == pytest.approx(1 / 3)
    assert full["return_on_funded"] == pytest.approx(full["net_pnl"] / 210_000)

    perfect_pd = hist["default"].to_numpy() * 0.9 + 0.05       # defaults get the top PDs
    curve = biz.tightening_curve(hist, perfect_pd, shares=[0.0, 1 / 3])
    assert curve.loc[0, "net_pnl"] == pytest.approx(full["net_pnl"])
    assert curve.loc[1, "principal_loss"] == 0 and curve.loc[1, "default_rate"] == 0
    assert curve.loc[1, "net_pnl"] == pytest.approx(7 * 2_200)
    with pytest.raises(ValueError):
        biz.policy_outcome(hist, np.ones(3, dtype=bool))
