"""Figures. Each function takes plain numbers or frames and writes one PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

NAVY, TEAL, ORANGE, GREY, LIGHT = "#1f3a5f", "#2a9d8f", "#e76f51", "#8d99ae", "#d9dee6"

FEATURE_LABELS = {
    "int_rate": "Interest rate", "annual_inc": "Annual income",
    "acc_open_past_24mths": "Accounts opened, last 24 months",
    "sub_grade_num": "Lending Club sub-grade", "dti": "Debt-to-income ratio",
    "payment_to_income": "Loan payment / income", "fico_range_low": "FICO score",
    "mths_since_recent_bc": "Months since newest bank card",
    "purpose": "Loan purpose", "home_ownership": "Home ownership",
    "revol_bal_to_income": "Revolving balance / income",
    "total_bc_limit": "Total bank card limit",
    "mo_sin_old_rev_tl_op": "Age of oldest revolving account",
    "mths_since_recent_inq": "Months since last credit inquiry",
    "loan_to_income": "Loan amount / income", "loan_amnt": "Loan amount",
    "installment": "Monthly instalment", "emp_length_years": "Years in current job",
    "credit_history_years": "Length of credit history",
    "tot_hi_cred_lim": "Total credit limit", "avg_cur_bal": "Average account balance",
    "bc_open_to_buy": "Unused bank card credit", "mort_acc": "Mortgage accounts",
    "verification_status": "Income verification", "revol_util": "Revolving utilisation",
    "tot_cur_bal": "Total current balance", "bc_util": "Bank card utilisation",
    "num_tl_op_past_12m": "Accounts opened, last 12 months",
    "inq_last_6mths": "Credit inquiries, last 6 months",
    "total_rev_hi_lim": "Total revolving limit", "revol_bal": "Revolving balance",
    "percent_bc_gt_75": "Share of cards above 75% utilisation",
    "mo_sin_rcnt_tl": "Months since newest account",
    "mo_sin_rcnt_rev_tl_op": "Months since newest revolving account",
    "mo_sin_old_il_acct": "Age of oldest instalment account",
    "total_bal_ex_mort": "Total balance excl. mortgage",
    "total_il_high_credit_limit": "Total instalment credit limit",
    "num_actv_rev_tl": "Active revolving accounts", "open_acc": "Open accounts",
    "total_acc": "Total accounts", "mths_since_last_delinq": "Months since last delinquency",
    "num_il_tl": "Instalment accounts", "num_rev_accts": "Revolving accounts",
    "num_bc_tl": "Bank card accounts", "num_op_rev_tl": "Open revolving accounts",
    "num_actv_bc_tl": "Active bank cards", "num_bc_sats": "Bank cards in good standing",
    "num_sats": "Accounts in good standing", "num_rev_tl_bal_gt_0": "Revolving accounts with a balance",
    "pct_tl_nvr_dlq": "Share of accounts never delinquent",
    "tot_coll_amt": "Amount in collections", "delinq_2yrs": "Delinquencies, last 2 years",
    "pub_rec": "Public records", "pub_rec_bankruptcies": "Bankruptcies on record",
    "mths_since_last_record": "Months since last public record",
    "mths_since_last_major_derog": "Months since last major derogatory",
    "mths_since_recent_bc_dlq": "Months since last bank card delinquency",
    "mths_since_recent_revol_delinq": "Months since last revolving delinquency",
    "num_accts_ever_120_pd": "Accounts ever 120+ days past due",
    "num_tl_90g_dpd_24m": "Accounts 90+ days past due, last 24 months",
    "collections_12_mths_ex_med": "Collections, last 12 months",
    "acc_now_delinq": "Accounts currently delinquent", "tax_liens": "Tax liens",
    "chargeoff_within_12_mths": "Charge-offs, last 12 months",
    "delinq_amnt": "Amount past due",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _millions(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}m")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _percent(ax, axis="x", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def model_performance(y, scores: dict, calibration: pd.DataFrame, aucs: dict, path: Path):
    """ROC curves for the three models and the calibration of the main one."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = [GREY, TEAL, NAVY]
    for (name, p), colour in zip(scores.items(), colours):
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax1.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    ax1.set(xlabel="Share of repaid loans flagged", ylabel="Share of defaults caught",
            title="Ranking: models against the lender's grade")
    _percent(ax1, "x"); _percent(ax1, "y")
    ax1.legend(loc="lower right")

    top = max(calibration["predicted"].max(), calibration["observed"].max()) * 1.08
    ax2.plot([0, top], [0, top], color="#bbbbbb", lw=1, ls="--", label="Perfect calibration")
    ax2.plot(calibration["predicted"], calibration["observed"], "o-", color=NAVY, lw=2,
             label="Gradient boosting, by PD decile")
    ax2.set(xlabel="Predicted default rate", ylabel="Actual default rate (2015 loans)",
            xlim=(0, top), ylim=(0, top),
            title="Level: 2015 loans defaulted more than predicted")
    _percent(ax2, "x"); _percent(ax2, "y")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def drivers(importance: pd.Series, path: Path, top: int = 15):
    _style()
    top_features = importance.head(top)[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in top_features.index]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(labels, top_features.to_numpy(), color=NAVY)
    ax.set(xlabel="Average effect on the prediction (mean |SHAP|, log-odds)",
           title="What drives the model's default predictions")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def cohort_default_rates(cohorts: pd.DataFrame, rho: float, path: Path, min_loans: int = 100):
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    palette = {"A": "#9ecae1", "B": "#6baed6", "C": "#3182bd", "D": "#1f3a5f", "E": ORANGE}
    order = sorted(cohorts["cohort"].unique())
    position = {c: i for i, c in enumerate(order)}
    for grade, colour in palette.items():
        sub = cohorts[(cohorts["grade"] == grade) & (cohorts["n"] >= min_loans)]
        ax.plot(sub["cohort"].map(position), sub["rate"], "o-", ms=3.5, lw=1.8,
                color=colour, label=f"Grade {grade}")
    ticks = [i for i, c in enumerate(order) if c.endswith("Q1")]
    ax.set_xticks(ticks, [order[i][:4] for i in ticks])
    ax.set(xlabel="Quarter the loans were issued", ylabel="Lifetime default rate",
           title=f"Default rates move together only a little across vintages "
                 f"(implied asset correlation {rho:.1%})")
    _percent(ax, "y")
    ax.legend(ncol=5, loc="upper center")
    ax.text(0.0, -0.2, f"Cohorts with fewer than {min_loans} loans are not drawn. "
            "Grades F and G are omitted (too few loans).", transform=ax.transAxes,
            fontsize=8.5, color="#555555")
    fig.tight_layout()
    _save(fig, path)


def loss_distribution(sims: dict, expected_loss: float, realised: float,
                      capital_levels: dict, path: Path):
    """sims: label -> simulated losses ($). capital_levels: label -> VaR 99.9% ($)."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colours = [NAVY, ORANGE]
    hi = max(np.quantile(l, 0.9995) for l in sims.values()) / 1e6
    lo = min(l.min() for l in sims.values()) / 1e6
    bins = np.linspace(lo, hi, 110)
    for (label, losses), colour in zip(sims.items(), colours):
        ax.hist(losses / 1e6, bins=bins, density=True, color=colour, alpha=0.55, label=label)
    ymax = ax.get_ylim()[1] * 1.22
    ax.set_ylim(0, ymax)
    for (label, _), colour in zip(sims.items(), colours):
        var = capital_levels[label] / 1e6
        ax.axvline(var, color=colour, lw=1.6, ls=":")
        ax.text(var, ymax * 0.42, f" 1-in-1,000 loss\n ${var:,.0f}m", color=colour,
                fontsize=8.5, va="bottom")
    ax.axvline(expected_loss / 1e6, color="#222222", lw=1.6)
    ax.text(expected_loss / 1e6, ymax * 0.985, f"Expected loss ${expected_loss / 1e6:,.0f}m ",
            fontsize=9, va="top", ha="right")
    ax.axvline(realised / 1e6, color=TEAL, lw=2.2)
    ax.text(realised / 1e6, ymax * 0.985, f" Actual loss ${realised / 1e6:,.0f}m",
            color=TEAL, fontsize=9, va="top", fontweight="bold")
    ax.set(xlabel="Lifetime credit loss on the 2015 book", ylabel="Probability density",
           title="The correlation assumption decides how wide the loss distribution is")
    _millions(ax, "x")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def capital_vs_correlation(sweep: pd.DataFrame, ci: tuple, points: dict, path: Path):
    """sweep: capital by rho (semi-analytic curve). points: label -> (rho,
    Monte Carlo capital) for the two scenarios that were simulated in full."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(sweep["rho"], sweep["capital_0.999"] / 1e6, color=NAVY, lw=2.2)
    ax.axvspan(ci[0], ci[1], color=LIGHT, alpha=0.8, lw=0)
    looks = [(NAVY, (18, -12), "center"), (ORANGE, (12, -8), "top")]
    for (label, (rho, capital)), (colour, offset, va) in zip(points.items(), looks):
        cap = capital / 1e6
        ax.plot([rho], [cap], "o", color=colour, ms=8, zorder=5)
        ax.annotate(f"{label}\nrho {rho:.1%}: ${cap:,.0f}m", (rho, cap),
                    textcoords="offset points", xytext=offset, va=va, fontsize=9, color=colour)
    ax.set(xlabel="Asset correlation (rho)", ylabel="Economic capital at 99.9%",
           title="Capital is far more sensitive to correlation than to the PD model",
           xlim=(0, sweep["rho"].max()))
    ax.set_ylim(bottom=0)
    _percent(ax, "x"); _millions(ax, "y")
    ax.text(0.0, -0.2, "Shaded band: 95% profile-likelihood interval of the estimated correlation.",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    fig.tight_layout()
    _save(fig, path)


def grade_shares(table: pd.DataFrame, path: Path):
    """table: index grade, columns share_funded, share_expected_loss, share_tail_loss."""
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(table))
    width = 0.27
    series = [("share_funded", "Share of lending", LIGHT),
              ("share_expected_loss", "Share of expected loss", TEAL),
              ("share_tail_loss", "Share of tail loss (worst 1% of scenarios)", NAVY)]
    for i, (col, label, colour) in enumerate(series):
        ax.bar(x + (i - 1) * width, table[col], width, label=label, color=colour)
    ax.set_xticks(x, [f"Grade {g}" for g in table.index])
    ax.set(ylabel="Share of portfolio total",
           title="Grades C to E carry far more of the risk than of the lending")
    _percent(ax, "y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def tightening(curves: dict, path: Path, max_share: float = 0.20):
    """curves: label -> tightening_curve frame."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    looks = [(NAVY, "-"), (TEAL, "--"), (GREY, "-")]
    for (label, curve), (colour, dash) in zip(curves.items(), looks):
        c = curve[curve["share_declined"] <= max_share + 1e-9]
        base = curve.iloc[0]
        ax1.plot(c["share_declined"], c["net_pnl"] / 1e6, color=colour, ls=dash, lw=2.2, label=label)
        ax2.plot(c["share_declined"], (base["principal_loss"] - c["principal_loss"]) / 1e6,
                 color=colour, ls=dash, lw=2.2, label=label)
    ax1.axhline(next(iter(curves.values())).iloc[0]["net_pnl"] / 1e6, color="#bbbbbb", lw=1, ls="--")
    ax1.set(xlabel="Share of the riskiest loans declined", ylabel="Realised net profit, 2015 book",
            title="Profit kept when lending is tightened")
    ax2.set(xlabel="Share of the riskiest loans declined", ylabel="Principal losses avoided",
            title="Losses avoided")
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(0, max_share + 1e-9, 0.05))
        _percent(ax, "x"); _millions(ax, "y")
        ax.legend(loc="lower left" if ax is ax1 else "upper left")
    fig.tight_layout()
    _save(fig, path)
