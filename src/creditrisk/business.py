"""Turn default probabilities into money: loss if default, expected return
per loan, and what alternative approval policies would actually have earned.

All dollar outcomes here are realised cash flows from the data. They are
undiscounted and before funding and servicing costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg

# Grade G has too few 36-month loans to estimate its own parameters reliably,
# so F and G share one set.
PARAM_GROUP = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "FG", "G": "FG"}


def _group(df: pd.DataFrame) -> pd.Series:
    group = df["grade"].map(PARAM_GROUP)
    if group.isna().any():
        raise ValueError("unknown grade in data")
    return group


def scheduled_interest(df: pd.DataFrame) -> pd.Series:
    """Interest the loan pays if every instalment is made on schedule."""
    return df["installment"] * cfg.TERM_MONTHS - df["funded_amnt"]


@dataclass
class ReturnParams:
    """Dollar-weighted averages by grade group, estimated on past vintages.

    severity              principal lost / amount funded, for defaulted loans
    return_if_default     net cash return / amount funded, for defaulted loans
                          (less negative than -severity because some interest
                          is collected before default)
    interest_realisation  net cash return of fully paid loans as a share of
                          scheduled interest (below 1 because of prepayment)
    """
    severity: dict
    return_if_default: dict
    interest_realisation: dict
    defaults_used: dict

    @classmethod
    def fit(cls, hist: pd.DataFrame) -> "ReturnParams":
        group = _group(hist)
        bad = hist["default"].eq(1)
        funded_bad = hist.loc[bad, "funded_amnt"].groupby(group[bad]).sum()
        severity = hist.loc[bad, "loss_amount"].groupby(group[bad]).sum() / funded_bad
        ret_def = hist.loc[bad, "net_pnl"].groupby(group[bad]).sum() / funded_bad
        sched = scheduled_interest(hist)
        realisation = (hist.loc[~bad, "net_pnl"].groupby(group[~bad]).sum()
                       / sched[~bad].groupby(group[~bad]).sum())
        expected = sorted(set(PARAM_GROUP.values()))
        for name, s in [("severity", severity), ("return_if_default", ret_def),
                        ("interest_realisation", realisation)]:
            if sorted(s.index) != expected or s.isna().any():
                raise ValueError(f"cannot estimate {name} for every grade group")
        return cls(severity.to_dict(), ret_def.to_dict(), realisation.to_dict(),
                   bad.groupby(group).sum().astype(int).to_dict())

    def _lookup(self, df: pd.DataFrame, table: dict) -> np.ndarray:
        return _group(df).map(table).to_numpy(dtype=float)

    def loss_if_default(self, df: pd.DataFrame) -> np.ndarray:
        """Expected dollar loss if the loan defaults."""
        return df["funded_amnt"].to_numpy(dtype=float) * self._lookup(df, self.severity)

    def expected_return(self, df: pd.DataFrame, pd_: np.ndarray) -> np.ndarray:
        """Expected net cash return per dollar funded:
        (1 - PD) * return if repaid + PD * return if default."""
        pd_ = np.asarray(pd_, dtype=float)
        funded = df["funded_amnt"].to_numpy(dtype=float)
        ret_paid = (self._lookup(df, self.interest_realisation)
                    * scheduled_interest(df).to_numpy(dtype=float) / funded)
        return (1 - pd_) * ret_paid + pd_ * self._lookup(df, self.return_if_default)


def policy_outcome(df: pd.DataFrame, approve: np.ndarray) -> dict:
    """What a book made of the approved loans actually did."""
    approve = np.asarray(approve, dtype=bool)
    if approve.shape != (len(df),):
        raise ValueError("approve must have one flag per loan")
    book = df.loc[approve]
    funded = float(book["funded_amnt"].sum())
    return {
        "loans": int(approve.sum()),
        "declined_share": float(1 - approve.mean()),
        "funded": funded,
        "default_rate": float(book["default"].mean()) if len(book) else float("nan"),
        "principal_loss": float(book["loss_amount"].sum()),
        "net_pnl": float(book["net_pnl"].sum()),
        "return_on_funded": float(book["net_pnl"].sum() / funded) if funded else float("nan"),
    }


def decline_riskiest(pd_: np.ndarray, share: float, seed: int = cfg.SEED) -> np.ndarray:
    """Approval flags when the `share` of loans with the highest PD is
    declined.

    Loans with identical PDs are ordered at random (fixed seed). This matters
    for the grade-based PD, which takes only 35 distinct values: without it,
    ties would be broken by the order of rows in the file.
    """
    pd_ = np.asarray(pd_, dtype=float)
    if not 0 <= share < 1:
        raise ValueError("share must be in [0, 1)")
    n_decline = int(round(share * len(pd_)))
    approve = np.ones(len(pd_), dtype=bool)
    if n_decline:
        tie_break = np.random.default_rng(seed).permutation(len(pd_))
        riskiest_first = np.lexsort((tie_break, -pd_))
        approve[riskiest_first[:n_decline]] = False
    return approve


def tightening_curve(df: pd.DataFrame, pd_: np.ndarray,
                     shares=np.arange(0.0, 0.3001, 0.01),
                     seed: int = cfg.SEED) -> pd.DataFrame:
    """Realised outcomes as a function of the share of loans declined, when
    loans are declined from the highest PD downward."""
    rows = []
    for share in shares:
        row = policy_outcome(df, decline_riskiest(pd_, float(share), seed))
        row["share_declined"] = float(share)
        rows.append(row)
    return pd.DataFrame(rows)
