"""Checks on outcome definitions, feature engineering, leakage and splits."""
import numpy as np
import pandas as pd
import pytest

from creditrisk import config as cfg
from creditrisk import data


def raw_frame():
    """Four hand-made loans with outcomes that can be verified by hand."""
    base = {c: np.nan for c in cfg.CORE_NUMERIC + cfg.BUREAU_NUMERIC}
    rows = [
        # fully paid: $10,000 funded, $11,500 received
        dict(id="1", issue_d="Mar-2013", loan_status="Fully Paid", grade="A",
             sub_grade="A1", funded_amnt=10_000, total_pymnt=11_500,
             total_rec_prncp=10_000, total_rec_int=1_500, recoveries=0,
             collection_recovery_fee=0, emp_length="10+ years",
             last_pymnt_d="Mar-2016", earliest_cr_line="Mar-2003",
             annual_inc=50_000, loan_amnt=10_000, installment=320.0, revol_bal=5_000),
        # charged off: $4,000 principal repaid, $1,000 recovered, $200 fee
        dict(id="2", issue_d="Jun-2014", loan_status="Charged Off", grade="G",
             sub_grade="G5", funded_amnt=10_000, total_pymnt=6_500,
             total_rec_prncp=4_000, total_rec_int=1_500, recoveries=1_000,
             collection_recovery_fee=200, emp_length="< 1 year",
             last_pymnt_d="Jan-2015", earliest_cr_line="Jun-2010",
             annual_inc=0, loan_amnt=10_000, installment=400.0, revol_bal=0),
        dict(id="3", issue_d="Jan-2015", loan_status="Current", grade="C",
             sub_grade="C3", funded_amnt=5_000, total_pymnt=4_000,
             total_rec_prncp=3_500, total_rec_int=500, recoveries=0,
             collection_recovery_fee=0, emp_length=None,
             last_pymnt_d="Jan-2019", earliest_cr_line="Jan-2000",
             annual_inc=40_000, loan_amnt=5_000, installment=170.0, revol_bal=100),
        dict(id="4", issue_d="Dec-2015", loan_status="Fully Paid", grade="C",
             sub_grade="C3", funded_amnt=5_000, total_pymnt=5_600,
             total_rec_prncp=5_000, total_rec_int=600, recoveries=0,
             collection_recovery_fee=0, emp_length="3 years",
             last_pymnt_d=None, earliest_cr_line="Dec-1995",
             annual_inc=40_000, loan_amnt=5_000, installment=170.0, revol_bal=100),
    ]
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["term"] = " 36 months"
    df["application_type"] = "Individual"
    df["total_rec_late_fee"] = 0.0
    df["home_ownership"] = ["RENT", "OWN", "RENT", None]
    df["verification_status"] = "Verified"
    df["purpose"] = "car"
    return df


def test_no_outcome_field_can_reach_a_model():
    forbidden = set(cfg.OUTCOME_COLS) | set(cfg.DERIVED_OUTCOMES) | {
        "loan_status", "last_pymnt_d", "last_fico_range_low", "last_fico_range_high",
        "out_prncp", "next_pymnt_d", "debt_settlement_flag", "hardship_flag",
        "addr_state", "zip_code", "id", "issue_d", "issue_date"}
    assert forbidden.isdisjoint(cfg.FEATURES)
    assert forbidden.isdisjoint(cfg.CORE_FEATURES)
    assert set(cfg.CORE_FEATURES) <= set(cfg.FEATURES)
    assert len(set(cfg.FEATURES)) == len(cfg.FEATURES)
    assert set(cfg.LENDER_SCORE) <= set(cfg.FEATURES)


def test_engineered_features_are_built_from_origination_fields_only():
    df, _ = data.prepare(raw_frame())
    tampered = df.copy()
    for col in cfg.OUTCOME_COLS + cfg.DERIVED_OUTCOMES + ["loan_status"]:
        if col in tampered:
            tampered[col] = -999 if col not in ("loan_status", "last_pymnt_d") else "X"
    rebuilt = data.add_features(tampered)
    pd.testing.assert_frame_equal(rebuilt[cfg.FEATURES], df[cfg.FEATURES])


def test_outcome_definitions():
    df, audit = data.prepare(raw_frame())
    assert audit["rows_in_extract"] == 4
    assert audit["rows_dropped_no_final_outcome"] == 1          # the 'Current' loan
    assert list(df["id"]) == ["1", "2", "4"]

    paid, bad = df.iloc[0], df.iloc[1]
    assert (paid["default"], bad["default"]) == (0, 1)
    assert paid["loss_amount"] == 0 and np.isnan(paid["severity"])
    assert paid["net_pnl"] == 1_500 and paid["realized_return"] == pytest.approx(0.15)
    # 10,000 funded - 4,000 repaid - (1,000 recovered - 200 fee) = 5,200 lost
    assert bad["loss_amount"] == 5_200 and bad["severity"] == pytest.approx(0.52)
    # 6,500 received - 200 fee - 10,000 funded = -3,700
    assert bad["net_pnl"] == -3_700 and bad["realized_return"] == pytest.approx(-0.37)
    # issued Jun-2014, last payment Jan-2015 -> 7 months; missing date -> 0
    assert list(df["months_to_last_payment"]) == [36, 7, 0]


def test_early_default_label():
    df = pd.DataFrame({"default": [1, 1, 1, 0, 1],
                       "months_to_last_payment": [0, 8, 9, 3, 30]})
    # 12-month window, 4-month lag: last payment by month 8 counts, month 9 does not.
    assert list(data.early_default_label(df, 12, 4)) == [1, 1, 0, 0, 0]
    assert list(data.early_default_label(df, 36, 4)) == [1, 1, 1, 0, 1]
    with pytest.raises(ValueError):
        data.early_default_label(df, 4, 4)


def test_feature_engineering():
    df, _ = data.prepare(raw_frame())
    assert list(df["sub_grade_num"]) == [1.0, 35.0, 13.0]
    assert list(df["emp_length_years"]) == [10.0, 0.0, 3.0]
    assert df["credit_history_years"].iloc[0] == pytest.approx(10.0, abs=0.01)
    assert df["loan_to_income"].iloc[0] == pytest.approx(0.2)
    assert df["payment_to_income"].iloc[0] == pytest.approx(12 * 320 / 50_000)
    assert np.isnan(df["loan_to_income"].iloc[1])               # zero income
    assert df["home_ownership"].iloc[2] == "MISSING"


def test_time_split_is_ordered_and_disjoint():
    dates = pd.date_range("2011-06-01", "2016-06-01", freq="MS")
    df = pd.DataFrame({"issue_date": dates, "k": range(len(dates))})
    split = data.time_split(df)
    assert [len(split[k]) for k in ("train", "calib", "test")] == [24, 12, 12]
    ids = [set(split[k]["k"]) for k in ("train", "calib", "test")]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert split["train"]["issue_date"].max() < split["calib"]["issue_date"].min()
    assert split["calib"]["issue_date"].max() < split["test"]["issue_date"].min()


def test_training_windows_end_before_the_test_year():
    test_start = pd.Timestamp(cfg.TEST[0])
    for window in (cfg.TRAIN, cfg.CALIB, cfg.POINT_IN_TIME_TRAIN, cfg.RHO_WINDOW):
        assert pd.Timestamp(window[1]) < test_start
    assert pd.Timestamp(cfg.POINT_IN_TIME_DATE) < test_start
    # Early-performance check: the last training loan has its full
    # observation window before the point-in-time date.
    observed = pd.Timestamp(cfg.TRAIN[1]).to_period("M").to_timestamp() + pd.DateOffset(
        months=cfg.EARLY_WINDOW_MONTHS)
    assert observed <= pd.Timestamp(cfg.POINT_IN_TIME_DATE)
    # Strict check: a 36-month loan issued at the end of the point-in-time
    # window has matured before the first test loan is issued.
    matured = pd.Timestamp(cfg.POINT_IN_TIME_TRAIN[1]) + pd.DateOffset(months=cfg.TERM_MONTHS)
    assert matured <= test_start


def test_window_rejects_reversed_bounds():
    df = pd.DataFrame({"issue_date": pd.to_datetime(["2013-01-01"])})
    with pytest.raises(ValueError):
        data.window(df, ("2014-01-01", "2013-01-01"))


def test_missing_raw_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Getting the data"):
        data.build_extract(tmp_path / "nope.csv", tmp_path / "out.parquet")


def test_build_extract_filters_rows(tmp_path):
    df = raw_frame()
    df.loc[0, "term"] = " 60 months"
    df.loc[1, "application_type"] = "Joint App"
    extra = df.iloc[[3]].assign(id="5", issue_d="Jan-2016")
    summary_row = pd.DataFrame([{c: np.nan for c in df.columns}]).assign(id="Total amount funded")
    pd.concat([df, extra, summary_row]).to_csv(tmp_path / "raw.csv", index=False)
    out = data.build_extract(tmp_path / "raw.csv", tmp_path / "out.parquet", chunksize=2)
    assert list(out["id"]) == ["3", "4"]
    assert list(pd.read_parquet(tmp_path / "out.parquet")["id"]) == ["3", "4"]
