"""Probability-of-default models, recalibration and evaluation.

Three models are compared on identical terms:

* ``GradeModel``    Lending Club's own sub-grade, turned into a PD. This is
                    the benchmark: what the lender already knew.
* ``LogitModel``    a regularised logistic regression (the scorecard-style
                    baseline most banks would recognise).
* ``GBMModel``      gradient-boosted trees (LightGBM).

Each model is fitted on the training vintages and then passed through the
same two-parameter recalibration on the following year's loans, so that
differences in results come from the models and not from the plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

EPS = 1e-6


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood fit of P(y=1) = sigmoid(a + b*x). Unpenalised.

    Written out by hand (rather than using a library class) so that it is
    exactly unpenalised and does not depend on library defaults.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-d arrays of equal length")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if y.min() == y.max():
        raise ValueError("y has a single class; cannot fit")
    mu, sd = x.mean(), x.std()
    if sd == 0:
        raise ValueError("x is constant; cannot fit a slope")
    xs = (x - mu) / sd

    def nll(theta):
        eta = theta[0] + theta[1] * xs
        # log(1 + exp(eta)) - y*eta, computed stably
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        resid = special.expit(eta) - y
        return value, np.array([resid.sum(), (resid * xs).sum()])

    start = np.array([logit(np.array([y.mean()]))[0], 0.0])
    res = optimize.minimize(nll, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 500})
    # BFGS can stop on "precision loss" when it is already at the optimum, so
    # convergence is judged by the size of the average gradient instead.
    if np.abs(res.jac).max() / len(y) > 1e-6:
        raise RuntimeError(f"1-d logistic fit did not converge: {res.message}")
    a_s, b_s = res.x
    b = b_s / sd
    a = a_s - b * mu
    return float(a), float(b)


@dataclass
class Recalibration:
    """PD = sigmoid(a + b * logit(raw score)).

    With b > 0 the map is strictly increasing, so it changes the level and
    spread of the PDs without changing how loans are ranked.
    """
    a: float
    b: float

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> "Recalibration":
        a, b = fit_logistic_1d(logit(raw), y)
        if b <= 0:
            raise RuntimeError("recalibration slope is not positive; the raw "
                               "score does not rank the calibration sample")
        return cls(a, b)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return special.expit(self.a + self.b * logit(raw))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GradeModel:
    """Lending Club's sub-grade (1 = A1 ... 35 = G5) mapped to a PD."""

    name = "LC sub-grade"

    def fit(self, df: pd.DataFrame) -> "GradeModel":
        self.a_, self.b_ = fit_logistic_1d(df["sub_grade_num"].to_numpy(),
                                           df["default"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return special.expit(self.a_ + self.b_ * df["sub_grade_num"].to_numpy(dtype=float))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to the [lower, upper] quantiles seen in training."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class LogitModel:
    """Regularised logistic regression on clipped, standardised inputs."""

    name = "Logistic regression"

    def __init__(self, features: list[str] | None = None, C: float = 0.5):
        self.features = list(features or cfg.FEATURES)
        self.C = C

    def fit(self, df: pd.DataFrame) -> "LogitModel":
        num = [f for f in self.features if f not in cfg.CATEGORICAL]
        cat = [f for f in self.features if f in cfg.CATEGORICAL]
        numeric = Pipeline([
            ("clip", QuantileClipper()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        transformers = [("num", numeric, num)]
        if cat:
            transformers.append(("cat", OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=100,
                sparse_output=False), cat))
        self.pipeline_ = Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(C=self.C, max_iter=5000)),
        ])
        self.pipeline_.fit(df[self.features], df["default"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(df[self.features])[:, 1]


@dataclass
class GBMModel:
    """Gradient-boosted trees with the number of trees chosen on the most
    recent slice of the training window (a time-ordered holdout), then
    refitted on the full training window."""

    features: list[str] = field(default_factory=lambda: list(cfg.FEATURES))
    target: str = "default"
    seed: int = cfg.SEED
    holdout_share: float = 0.2
    max_trees: int = 4000
    name: str = "Gradient boosting"

    def _params(self) -> dict:
        return dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=24, min_data_in_leaf=200, bagging_fraction=0.8,
            bagging_freq=1, feature_fraction=0.6, lambda_l2=10.0, max_bin=255,
            seed=self.seed, num_threads=1, deterministic=True,
            force_row_wise=True, verbosity=-1,
        )

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for col, cats in self.categories_.items():
            # Categories never seen in training become missing values.
            X[col] = pd.Categorical(X[col].where(X[col].isin(cats)), categories=cats)
        return X

    def fit(self, df: pd.DataFrame) -> "GBMModel":
        cat_cols = [f for f in self.features if f in cfg.CATEGORICAL]
        self.categories_ = {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}

        # Time-ordered holdout: the last `holdout_share` of loans by issue date.
        order = df["issue_date"].rank(method="first").to_numpy()
        is_holdout = order > (1 - self.holdout_share) * len(df)
        cutoff = df.loc[is_holdout, "issue_date"].min()
        is_holdout = (df["issue_date"] >= cutoff).to_numpy()   # keep months whole
        X, y = self._frame(df), df[self.target].to_numpy()

        fit_part = lgb.Dataset(X.loc[~is_holdout], label=y[~is_holdout])
        holdout = lgb.Dataset(X.loc[is_holdout], label=y[is_holdout], reference=fit_part)
        probe = lgb.train(self._params(), fit_part, num_boost_round=self.max_trees,
                          valid_sets=[holdout], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        self.n_trees_ = int(probe.best_iteration)
        self.holdout_auc_ = float(probe.best_score["holdout"]["auc"])
        self.holdout_start_ = pd.Timestamp(cutoff)

        self.model_ = lgb.train(self._params(), lgb.Dataset(X, label=y),
                                num_boost_round=self.n_trees_)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._frame(df)), dtype=float)

    def shap_importance(self, df: pd.DataFrame) -> pd.Series:
        """Mean absolute SHAP value per feature (in log-odds units)."""
        contrib = self.model_.predict(self._frame(df), pred_contrib=True)
        contrib = np.asarray(contrib)[:, :-1]          # last column is the base value
        return (pd.Series(np.abs(contrib).mean(axis=0), index=self.features)
                .sort_values(ascending=False))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity (ties averaged)."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs both classes")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Largest gap between the score distributions of defaulters and payers."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "auc": auc(y, p),
        "gini": 2 * auc(y, p) - 1,
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
        "mean_pd": float(p.mean()),
        "default_rate": float(y.mean()),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate by equal-sized PD bucket."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    frame["bucket"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False) + 1
    out = frame.groupby("bucket").agg(loans=("y", "size"), predicted=("p", "mean"),
                                      observed=("y", "mean"))
    return out.reset_index()


def bootstrap_auc_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                             n_boot: int = 200, seed: int = cfg.SEED) -> dict:
    """AUC(a) - AUC(b) with a percentile bootstrap interval.

    The same resampled loans are scored by both models in each replication,
    so the interval reflects the paired comparison.
    """
    y = np.asarray(y)
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = auc(y[idx], p_a[idx]) - auc(y[idx], p_b[idx])
    return {
        "difference": auc(y, p_a) - auc(y, p_b),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_boot": n_boot,
    }
