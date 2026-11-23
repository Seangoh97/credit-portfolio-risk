"""Build the interactive dashboard (docs/index.html) from the pipeline outputs.

    python run_pipeline.py      # writes outputs/
    python build_dashboard.py   # writes docs/index.html

The page is a single self-contained HTML file: the template in
dashboard/template.html with the results inlined as JSON. GitHub Pages serves
it from the docs folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from creditrisk import config as cfg          # noqa: E402
from creditrisk.plots import FEATURE_LABELS   # noqa: E402

TEMPLATE = ROOT / "dashboard" / "template.html"
OUT = ROOT / "docs" / "index.html"
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"


def m(x: float) -> float:
    """Dollars to millions, rounded for display."""
    return round(float(x) / 1e6, 3)


def r(x: float, digits: int = 5) -> float:
    return round(float(x), digits)


def build() -> dict:
    res = json.loads((cfg.OUT_DIR / "results.json").read_text())
    tab = cfg.TAB_DIR
    metrics = pd.DataFrame(res["model_metrics"])
    rec = metrics[metrics["stage"] == "recalibrated on 2014"].set_index("model")
    rob = res["robustness"]
    port = res["portfolio"]
    corr = res["correlation"]
    samples = res["data"]["samples"]

    models = [{"name": name, "auc": r(rec.loc[name, "auc"]), "ks": r(rec.loc[name, "ks"]),
               "mean_pd": r(rec.loc[name, "mean_pd"])}
              for name in ("LC sub-grade", "Logistic regression", "Gradient boosting")]
    variants = [
        ("Main model: final outcomes of 2012-13 loans", rob["auc_main_model"], "main"),
        ("Main model without the lender's grade and rate", rob["auc_without_lender_grade_and_rate"], "check"),
        ("Core inputs only, 2012-13 loans", rob["auc_core_features_trained_2012_13"], "check"),
        ("Point-in-time: early delinquency seen by Dec 2014", rob["auc_early_performance_point_in_time"], "pit"),
        ("Lending Club sub-grade (benchmark)", rob["auc_lc_sub_grade"], "benchmark"),
        ("Point-in-time: only loans matured by Dec 2014", rob["auc_strict_point_in_time"], "pit"),
    ]

    sweep = pd.read_csv(tab / "capital_vs_correlation.csv")
    dens = pd.read_csv(tab / "loss_density_by_correlation.csv")
    rho_data, rho_ref = port["scenarios"]["data-implied"]["rho"], port["scenarios"]["Basel reference"]["rho"]
    rho_rows, density = [], []
    loss_grid = sorted(dens["loss"].unique())
    for _, row in sweep.iterrows():
        tag = "data" if abs(row["rho"] - rho_data) < 1e-12 else "reference" if abs(row["rho"] - rho_ref) < 1e-12 else ""
        rho_rows.append({"rho": r(row["rho"], 6), "tag": tag, "el": m(row["expected_loss"]),
                         "var99": m(row["var_0.99"]), "var999": m(row["var_0.999"]),
                         "es99": m(row["es_0.99"]), "cap99": m(row["capital_0.99"]),
                         "cap999": m(row["capital_0.999"]),
                         "pct": r(row["realised_loss_percentile"], 4)})
        d = dens[dens["rho"] == row["rho"]].sort_values("loss")
        assert len(d) == len(loss_grid)
        density.append([round(float(v) * 1e6, 6) for v in d["density"]])   # per $m
    assert sum(1 for x in rho_rows if x["tag"]) == 2

    rankings = []
    for key, label, tag in [("gbm", "Gradient boosting", "gbm"),
                            ("pit", "Point-in-time model", "gbm_point_in_time"),
                            ("grade", "Lending Club grade", "grade")]:
        c = pd.read_csv(tab / f"tightening_curve_{tag}.csv")
        rankings.append({
            "key": key, "label": label,
            "net_pnl": [m(v) for v in c["net_pnl"]],
            "principal_loss": [m(v) for v in c["principal_loss"]],
            "funded": [m(v) for v in c["funded"]],
            "default_rate": [r(v) for v in c["default_rate"]],
            "loans": [int(v) for v in c["loans"]],
            "expected_loss": [m(v) for v in c["expected_loss"]],
            "cap_data": [m(v) for v in c["capital_0.999_data"]],
            "cap_ref": [m(v) for v in c["capital_0.999_reference"]],
        })
        shares = [r(v, 4) for v in c["share_declined"]]

    grades = pd.read_csv(tab / "risk_by_grade.csv")
    calib = pd.read_csv(tab / "calibration_by_decile.csv")
    shap = pd.read_csv(tab / "shap_importance.csv").head(12)
    rule = pd.DataFrame(res["expected_return_rule"]).set_index("pd_model").loc["Gradient boosting"]
    mc = {k: port["scenarios"][s]["monte_carlo"] for k, s in
          [("data", "data-implied"), ("reference", "Basel reference")]}

    return {
        "meta": {
            "loans_test": port["loans"], "funded_test": m(port["funded"]),
            "loans_clean": res["data"]["audit"]["rows_clean"],
            "default_rate": {"train": r(samples["train"]["default_rate"]),
                             "calib": r(samples["calib"]["default_rate"]),
                             "test": r(samples["test"]["default_rate"])},
            "train_loans": samples["train"]["loans"], "strict_loans": rob["strict_training_loans"],
            "scenarios": res["settings"]["scenarios"], "checks": len(res["checks"]),
            "checks_passed": sum(c["passed"] for c in res["checks"].values()),
        },
        "models": models,
        "variants": [{"label": a, "auc": r(b), "kind": c} for a, b, c in variants],
        "gain": {k: r(res["auc_gain"]["gbm_vs_grade"][k]) for k in ("difference", "ci_low", "ci_high")},
        "calibration": [{"decile": int(x.bucket), "predicted": r(x.predicted), "observed": r(x.observed)}
                        for x in calib.itertuples()],
        "drivers": [{"label": FEATURE_LABELS.get(x.feature, x.feature), "value": r(x.mean_abs_shap, 4)}
                    for x in shap.itertuples()],
        "portfolio": {"expected_loss": m(port["expected_loss"]), "realised_loss": m(port["realised_loss"]),
                      "gap": m(port["forecast_gap"]), "gap_default": m(port["gap_from_default_rate"]),
                      "gap_severity": m(port["gap_from_severity"]), "funded": m(port["funded"]),
                      "mean_pd": r(port["mean_pd"]), "default_rate": r(port["default_rate"])},
        "correlation": {"rho_data": r(rho_data, 6), "ci_low": r(corr["mle_quarterly"]["ci_low"], 6),
                        "ci_high": r(corr["mle_quarterly"]["ci_high"], 6), "rho_ref": r(rho_ref, 6),
                        "annual": r(corr["mle_annual"]["rho"], 6), "moments": r(corr["moments_quarterly"], 6)},
        "monte_carlo": {k: {"var999": m(v["var_0.999"]), "cap999": m(v["capital_0.999"])} for k, v in mc.items()},
        "rho": rho_rows,
        "loss_grid": [m(v) for v in loss_grid],
        "density": density,
        "policy": {"shares": shares, "rankings": rankings,
                   "book": {"net_pnl": m(res["book_actual"]["net_pnl"]),
                            "return": r(res["book_actual"]["return_on_funded"])}},
        "rule": {"declined_share": r(rule["declined_share"]), "funded_declined": m(rule["funded_declined"]),
                 "default_rate": r(rule["declined_default_rate"]), "net_pnl": m(rule["declined_net_pnl"])},
        "grades": [{"grade": g.grade, "loans": int(g.loans), "funded": m(g.funded), "mean_pd": r(g.mean_pd),
                    "default_rate": r(g.default_rate), "realised_return": r(g.realised_return),
                    "share_funded": r(g.share_funded), "share_el": r(g.share_expected_loss),
                    "share_tail": r(g.share_tail_loss)} for g in grades.itertuples()],
    }


def main():
    payload = json.dumps(build(), separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text()
    if html.count(PLACEHOLDER) != 1:
        raise ValueError("template must contain the data placeholder exactly once")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # "</" inside inline JSON could close the script tag early; escape it.
    OUT.write_text(html.replace(PLACEHOLDER, payload.replace("</", "<\\/")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
