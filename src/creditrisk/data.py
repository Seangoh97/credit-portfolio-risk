"""Load the Lending Club file, define outcomes, build features, split by time.

The functions here enforce the two rules the whole project depends on:

1. Only information available when the loan was funded is used as a model
   input. Everything recorded afterwards is an outcome.
2. Samples are split by origination date, never at random.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

TERMINAL_STATUSES = {"Fully Paid": 0, "Charged Off": 1}


# ---------------------------------------------------------------------------
# Raw file -> compact extract
# ---------------------------------------------------------------------------
def build_extract(raw_csv: Path = cfg.RAW_CSV, out_path: Path = cfg.EXTRACT,
                  chunksize: int = 200_000) -> pd.DataFrame:
    """Read the 1.6 GB raw CSV in chunks and keep only the rows and columns
    the project needs: 36-month individual loans issued through 2015.

    No outcome-based filtering happens here; that is done in ``clean`` so the
    number of loans dropped at each step can be reported.
    """
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Download accepted_2007_to_2018Q4.csv "
            "(see README, 'Getting the data') and place it in the data folder."
        )
    header = pd.read_csv(raw_csv, nrows=0).columns
    missing = sorted(set(cfg.RAW_USECOLS) - set(header))
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")

    last_issue = pd.Timestamp(cfg.LAST_ISSUE_DATE)
    parts = []
    reader = pd.read_csv(raw_csv, usecols=cfg.RAW_USECOLS, chunksize=chunksize,
                         low_memory=False)
    for chunk in reader:
        issue = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = (
            (chunk["term"].astype("string").str.strip() == cfg.TERM)
            & (issue <= last_issue)
            & (chunk["application_type"] == "Individual")
        )
        parts.append(chunk.loc[keep.fillna(False)])
    df = pd.concat(parts, ignore_index=True)
    df["id"] = df["id"].astype(str)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_extract(path: Path = cfg.EXTRACT) -> pd.DataFrame:
    """Load the compact extract, building it from the raw CSV if needed."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return build_extract(out_path=path)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep loans with a final outcome and derive the outcome measures.

    Returns the cleaned frame and an audit dictionary with row counts.

    Outcome definitions (all in dollars unless stated):

    * ``default``         1 if the loan was charged off, 0 if fully paid.
    * ``loss_amount``     principal never repaid, net of recoveries after
                          collection fees. Zero for fully paid loans.
    * ``severity``        loss_amount / funded_amnt for defaulted loans. This
                          combines two effects: how much principal had been
                          repaid before default, and how much was recovered
                          afterwards.
    * ``net_pnl``         all cash received minus collection fees minus the
                          amount funded. Undiscounted, before funding and
                          servicing costs.
    * ``realized_return`` net_pnl / funded_amnt.
    * ``months_to_last_payment``  months from issue to the last payment
                          received (0 if no payment was ever made).
    """
    audit = {"rows_in_extract": int(len(df))}
    status = df["loan_status"].astype("string")
    audit["status_counts"] = {k: int(v) for k, v in status.value_counts().items()}

    out = df.loc[status.isin(list(TERMINAL_STATUSES))].copy()
    out["issue_date"] = pd.to_datetime(out["issue_d"], format="%b-%Y")
    out["default"] = out["loan_status"].map(TERMINAL_STATUSES).astype("int8")
    audit["rows_dropped_no_final_outcome"] = int(len(df) - len(out))

    funded = out["funded_amnt"].astype(float)
    ok = funded.gt(0) & out["total_pymnt"].notna()
    audit["rows_dropped_bad_amounts"] = int((~ok).sum())
    out = out.loc[ok].copy()
    funded = out["funded_amnt"].astype(float)

    net_recoveries = out["recoveries"].fillna(0.0) - out["collection_recovery_fee"].fillna(0.0)
    principal_lost = funded - out["total_rec_prncp"] - net_recoveries
    is_default = out["default"].eq(1)
    out["loss_amount"] = np.where(is_default, principal_lost, 0.0)
    out["severity"] = np.where(is_default, principal_lost / funded, np.nan)
    out["net_pnl"] = out["total_pymnt"] - out["collection_recovery_fee"].fillna(0.0) - funded
    out["realized_return"] = out["net_pnl"] / funded

    last_payment = pd.to_datetime(out["last_pymnt_d"], format="%b-%Y", errors="coerce")
    months = ((last_payment.dt.year - out["issue_date"].dt.year) * 12
              + (last_payment.dt.month - out["issue_date"].dt.month))
    out["months_to_last_payment"] = months.fillna(0).clip(lower=0).astype("int16")

    audit["rows_clean"] = int(len(out))
    audit["default_rate"] = float(out["default"].mean())
    return out.reset_index(drop=True), audit


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _sub_grade_to_num(s: pd.Series) -> pd.Series:
    """'A1' -> 1, 'A2' -> 2, ..., 'G5' -> 35."""
    s = s.astype("string")
    letter = s.str[0].map({g: i for i, g in enumerate(cfg.GRADES)})
    digit = pd.to_numeric(s.str[1], errors="coerce")
    return (letter.astype("float64") * 5 + digit).astype("float64")


def _emp_length_to_years(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '3 years' -> 3, '10+ years' -> 10, missing -> NaN."""
    s = s.astype("string").str.strip()
    years = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce").astype("float64")
    years = years.mask(s.str.startswith("<").fillna(False), 0.0)
    return years


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Uses origination-time fields only."""
    out = df.copy()
    if "issue_date" not in out:
        out["issue_date"] = pd.to_datetime(out["issue_d"], format="%b-%Y")

    out["sub_grade_num"] = _sub_grade_to_num(out["sub_grade"])
    out["emp_length_years"] = _emp_length_to_years(out["emp_length"])

    first_credit = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_history_years"] = (out["issue_date"] - first_credit).dt.days / 365.25

    income = out["annual_inc"].astype(float).where(out["annual_inc"] > 0)
    out["loan_to_income"] = out["loan_amnt"] / income
    out["payment_to_income"] = 12.0 * out["installment"] / income
    out["revol_bal_to_income"] = out["revol_bal"] / income

    for col in cfg.NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in cfg.CATEGORICAL:
        out[col] = out[col].astype("string").str.strip().fillna("MISSING").astype(object)
    return out


def early_default_label(df: pd.DataFrame,
                        window_months: int = cfg.EARLY_WINDOW_MONTHS,
                        lag_months: int = cfg.DELINQUENCY_LAG_MONTHS) -> pd.Series:
    """1 if the loan was at least `lag_months` behind on payments within
    `window_months` of being issued, and was later charged off.

    A lender can observe this `window_months` after issue, long before the
    loan matures. A charged-off loan whose last payment came m months after
    issue is `lag_months` past due at month m + lag_months.
    """
    if not 0 < lag_months < window_months:
        raise ValueError("need 0 < lag_months < window_months")
    early = df["default"].eq(1) & (df["months_to_last_payment"] + lag_months <= window_months)
    return early.astype("int8")


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """clean + add_features in one call."""
    cleaned, audit = clean(df)
    return add_features(cleaned), audit


# ---------------------------------------------------------------------------
# Time split
# ---------------------------------------------------------------------------
def window(df: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    """Loans issued within [start, end], both inclusive."""
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    if start > end:
        raise ValueError("window start is after window end")
    mask = (df["issue_date"] >= start) & (df["issue_date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train / calibration / test split by origination date."""
    split = {
        "train": window(df, cfg.TRAIN),
        "calib": window(df, cfg.CALIB),
        "test": window(df, cfg.TEST),
    }
    assert split["train"]["issue_date"].max() < split["calib"]["issue_date"].min()
    assert split["calib"]["issue_date"].max() < split["test"]["issue_date"].min()
    return split
