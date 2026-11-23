"""Run the whole analysis: data -> PD models -> correlation -> loss
distribution -> business impact. Writes tables, figures and results.json to
the outputs folder.

    python run_pipeline.py                 # full run, about 10 minutes
    python run_pipeline.py --scenarios 2000    # faster, noisier tail estimates
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from creditrisk import business as biz          # noqa: E402
from creditrisk import config as cfg            # noqa: E402
from creditrisk import correlation as corr      # noqa: E402
from creditrisk import data, plots              # noqa: E402
from creditrisk import pd_model as pm           # noqa: E402
from creditrisk import portfolio as pf          # noqa: E402

LEVELS = (0.95, 0.99, 0.999)
T0 = time.time()


def log(message: str):
    print(f"[{time.time() - T0:6.0f}s] {message}", flush=True)


def check(results: dict, name: str, value: float, limit: float):
    """Record a numerical self-check and stop the run if it fails."""
    passed = bool(abs(value) <= limit)
    results["checks"][name] = {"value": float(value), "limit": limit, "passed": passed}
    if not passed:
        raise AssertionError(f"self-check failed: {name} = {value:.3g} (limit {limit})")


def main(n_scenarios: int):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"checks": {}, "settings": {
        "seed": cfg.SEED, "scenarios": n_scenarios, "train": cfg.TRAIN,
        "calibration": cfg.CALIB, "test": cfg.TEST,
        "point_in_time_train": cfg.POINT_IN_TIME_TRAIN, "rho_window": cfg.RHO_WINDOW}}

    # ------------------------------------------------------------------ data
    log("Loading data")
    loans, audit = data.prepare(data.load_extract())
    split = data.time_split(loans)
    train, calib, test = split["train"], split["calib"], split["test"]
    y_test = test["default"].to_numpy()
    results["data"] = {"audit": audit, "samples": {
        name: {"loans": int(len(df)), "default_rate": float(df["default"].mean()),
               "funded": float(df["funded_amnt"].sum()),
               "first_issue": str(df["issue_date"].min().date()),
               "last_issue": str(df["issue_date"].max().date())}
        for name, df in split.items()}}

    history = pd.concat([train, calib], ignore_index=True)
    params = biz.ReturnParams.fit(history)
    results["return_params"] = {
        "severity": params.severity, "return_if_default": params.return_if_default,
        "interest_realisation": params.interest_realisation,
        "defaults_used": params.defaults_used}

    # ------------------------------------------------------------- PD models
    log("Fitting PD models")
    models = {"LC sub-grade": pm.GradeModel().fit(train),
              "Logistic regression": pm.LogitModel().fit(train),
              "Gradient boosting": pm.GBMModel().fit(train)}
    pds, raw_scores, metrics = {}, {}, []
    results["recalibration"] = {}
    for name, model in models.items():
        raw_calib, raw_test = model.predict(calib), model.predict(test)
        recal = pm.Recalibration.fit(raw_calib, calib["default"].to_numpy())
        pds[name], raw_scores[name] = recal.apply(raw_test), raw_test
        results["recalibration"][name] = {"intercept": recal.a, "slope": recal.b}
        check(results, f"recalibrated mean PD equals calibration default rate [{name}]",
              recal.apply(raw_calib).mean() - calib["default"].mean(), 1e-6)
        metrics.append({"model": name, "stage": "as fitted on 2012-13", **pm.evaluate(y_test, raw_test)})
        metrics.append({"model": name, "stage": "recalibrated on 2014", **pm.evaluate(y_test, pds[name])})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(cfg.TAB_DIR / "model_metrics.csv", index=False)
    results["model_metrics"] = metrics.to_dict(orient="records")
    gbm = models["Gradient boosting"]
    results["gbm"] = {"trees": gbm.n_trees_, "holdout_auc": gbm.holdout_auc_,
                      "holdout_start": str(gbm.holdout_start_.date()),
                      "features": len(gbm.features)}
    p_gbm, p_grade = pds["Gradient boosting"], pds["LC sub-grade"]

    log("Bootstrapping AUC differences")
    results["auc_gain"] = {
        "gbm_vs_grade": pm.bootstrap_auc_difference(y_test, p_gbm, p_grade),
        "gbm_vs_logit": pm.bootstrap_auc_difference(y_test, p_gbm, pds["Logistic regression"])}

    log("Robustness models")
    no_lender = [f for f in cfg.FEATURES if f not in cfg.LENDER_SCORE]
    core_recent = pm.GBMModel(features=list(cfg.CORE_FEATURES)).fit(train)
    strict_train = data.window(loans, cfg.POINT_IN_TIME_TRAIN)
    strict = pm.GBMModel(features=list(cfg.CORE_FEATURES)).fit(strict_train)
    # Same loans and features as the main model, labelled only with what had
    # been observed by the point-in-time date.
    observed_by = train["issue_date"].max() + pd.DateOffset(months=cfg.EARLY_WINDOW_MONTHS)
    assert observed_by <= pd.Timestamp(cfg.POINT_IN_TIME_DATE)
    early_train = train.assign(early_default=data.early_default_label(train))
    early = pm.GBMModel(target="early_default").fit(early_train)
    score_early = early.predict(test)      # used for ranking only, not as a PD
    results["robustness"] = {
        "auc_early_performance_point_in_time": pm.auc(y_test, score_early),
        "early_default_rate_in_training": float(early_train["early_default"].mean()),
        "early_defaults_in_training": int(early_train["early_default"].sum()),
        "auc_lc_sub_grade": pm.auc(y_test, p_grade),
        "auc_main_model": pm.auc(y_test, p_gbm),
        "auc_without_lender_grade_and_rate": pm.auc(
            y_test, pm.GBMModel(features=no_lender).fit(train).predict(test)),
        "auc_core_features_trained_2012_13": pm.auc(y_test, core_recent.predict(test)),
        "auc_strict_point_in_time": pm.auc(y_test, strict.predict(test)),
        "strict_training_loans": int(len(strict_train)),
        "strict_training_last_issue": str(strict_train["issue_date"].max().date()),
    }

    calibration = pm.calibration_table(y_test, p_gbm)
    calibration.to_csv(cfg.TAB_DIR / "calibration_by_decile.csv", index=False)
    importance = gbm.shap_importance(test.sample(20_000, random_state=cfg.SEED))
    importance.rename("mean_abs_shap").to_csv(cfg.TAB_DIR / "shap_importance.csv",
                                              index_label="feature")
    results["top_drivers"] = importance.head(10).to_dict()

    # ----------------------------------------------------------- correlation
    log("Estimating asset correlation")
    cohorts = corr.cohort_table(loans, "Q")
    cohorts.to_csv(cfg.TAB_DIR / "cohort_default_rates.csv", index=False)
    n_q, d_q, _, _ = corr.to_matrices(cohorts)
    mle = corr.fit_mle(n_q, d_q)
    mle_fine = corr.fit_mle(n_q, d_q, points=8001)
    check(results, "rho unchanged when integration grid is doubled",
          mle["rho"] - mle_fine["rho"], 1e-5)
    n_y, d_y, _, _ = corr.to_matrices(corr.cohort_table(loans, "Y"))
    annual = corr.fit_mle(n_y, d_y)
    rho_data = mle["rho"]
    annual_pd = 1 - (1 - p_gbm) ** (12 / cfg.TERM_MONTHS)
    rho_ref = float(np.average(corr.basel_other_retail_rho(annual_pd),
                               weights=test["funded_amnt"].to_numpy()))
    results["correlation"] = {
        "mle_quarterly": mle, "mle_annual": annual,
        "moments_quarterly": corr.fit_mom(n_q, d_q)["rho"],
        "moments_annual": corr.fit_mom(n_y, d_y)["rho"],
        "cohort_loans": float(n_q.sum()), "basel_reference": rho_ref,
        "default_correlation_at_data_rho": pf.default_correlation(float(p_gbm.mean()), rho_data),
        "default_correlation_at_reference_rho": pf.default_correlation(float(p_gbm.mean()), rho_ref)}

    # ------------------------------------------------------ loss distribution
    log("Simulating the 2015 book")
    lid = params.loss_if_default(test)
    grades = test["grade"].to_numpy()
    funded = test["funded_amnt"].to_numpy(dtype=float)
    realised_loss = float(test["loss_amount"].sum())
    expected_loss = float(p_gbm @ lid)
    scenarios = {"data-implied": rho_data, "Basel reference": rho_ref}
    sims, portfolio = {}, {}
    for label, rho in scenarios.items():
        sim = pf.simulate(p_gbm, lid, rho, n_scenarios=n_scenarios, seed=cfg.SEED, segments=grades)
        mc = pf.risk_measures(sim.losses, LEVELS)
        sa_model = pf.semi_analytic(p_gbm, lid, rho)
        sa = sa_model.risk_measures(LEVELS)
        tail = pf.tail_contributions(sim, 0.99)
        sims[label] = sim
        portfolio[label] = {
            "rho": rho, "monte_carlo": mc, "semi_analytic": sa,
            "asrf_var_0.999": pf.asrf_quantile(p_gbm, lid, rho, 0.999),
            "realised_loss_percentile": float((sim.losses < realised_loss).mean()),
            "realised_loss_percentile_semi_analytic": sa_model.cdf(realised_loss),
            "tail_contribution_0.99": tail}
        if n_scenarios >= 20_000:      # tolerances assume the default scenario count
            check(results, f"Monte Carlo vs analytic expected loss [{label}]",
                  mc["expected_loss"] / expected_loss - 1, 0.002)
            check(results, f"Monte Carlo vs semi-analytic 99.9% loss [{label}]",
                  mc["var_0.999"] / sa["var_0.999"] - 1, 0.015)
            check(results, f"Monte Carlo vs semi-analytic 99% expected shortfall [{label}]",
                  mc["es_0.99"] / sa["es_0.99"] - 1, 0.01)
        check(results, f"tail contributions sum to expected shortfall [{label}]",
              sum(tail.values()) / mc["es_0.99"] - 1, 1e-9)
        log(f"  rho {rho:.4f}: capital 99.9% ${mc['capital_0.999'] / 1e6:,.1f}m")

    # Why the forecast missed: more defaults than predicted, or bigger losses per default?
    loss_actual_defaults_assumed_severity = float(lid @ y_test)
    results["portfolio"] = {
        "loans": int(len(test)), "funded": float(funded.sum()),
        "loss_if_default_total": float(lid.sum()),
        "mean_pd": float(p_gbm.mean()), "default_rate": float(y_test.mean()),
        "expected_loss": expected_loss, "realised_loss": realised_loss,
        "forecast_gap": realised_loss - expected_loss,
        "gap_from_default_rate": loss_actual_defaults_assumed_severity - expected_loss,
        "gap_from_severity": realised_loss - loss_actual_defaults_assumed_severity,
        "scenarios": portfolio}

    log("Correlation sweep and PD-model comparison")
    # The sweep includes the two scenario values so the dashboard can show them exactly.
    sweep_rhos = sorted({0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
                         0.06, 0.07, 0.08, 0.09, 0.10, 0.12, rho_data, rho_ref})
    loss_grid = np.arange(0.0, 800e6 + 1, 4e6)          # $0m to $800m in $4m steps
    sweep, densities = [], []
    for rho in sweep_rhos:
        sa_model = pf.semi_analytic(p_gbm, lid, rho)
        sweep.append({"rho": rho, **sa_model.risk_measures(LEVELS),
                      "realised_loss_percentile": sa_model.cdf(realised_loss)})
        densities.append(pd.DataFrame({"rho": rho, "loss": loss_grid,
                                       "density": sa_model.pdf(loss_grid)}))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(cfg.TAB_DIR / "capital_vs_correlation.csv", index=False)
    pd.concat(densities).to_csv(cfg.TAB_DIR / "loss_density_by_correlation.csv", index=False)
    results["capital_by_pd_model"] = {
        name: {label: pf.semi_analytic(p, lid, rho).risk_measures(LEVELS)
               for label, rho in scenarios.items()}
        for name, p in pds.items()}

    grade_table = pd.DataFrame({
        "loans": test.groupby("grade").size(),
        "funded": test.groupby("grade")["funded_amnt"].sum(),
        "mean_pd": pd.Series(p_gbm).groupby(grades).mean(),
        "default_rate": test.groupby("grade")["default"].mean(),
        "expected_loss": pd.Series(p_gbm * lid).groupby(grades).sum(),
        "realised_loss": test.groupby("grade")["loss_amount"].sum(),
        "realised_return": test.groupby("grade")["net_pnl"].sum()
                           / test.groupby("grade")["funded_amnt"].sum(),
    })
    for label in scenarios:
        key = "data" if label == "data-implied" else "reference"
        grade_table[f"tail_loss_{key}"] = pd.Series(portfolio[label]["tail_contribution_0.99"])
    grade_table["share_funded"] = grade_table["funded"] / grade_table["funded"].sum()
    grade_table["share_expected_loss"] = grade_table["expected_loss"] / grade_table["expected_loss"].sum()
    grade_table["share_tail_loss"] = grade_table["tail_loss_reference"] / grade_table["tail_loss_reference"].sum()
    grade_table.to_csv(cfg.TAB_DIR / "risk_by_grade.csv", index_label="grade")
    results["risk_by_grade"] = grade_table.reset_index(names="grade").to_dict(orient="records")

    # --------------------------------------------------------------- business
    log("Business impact")
    rankings = {"Gradient boosting": ("gbm", p_gbm),
                "Gradient boosting, point-in-time version": ("gbm_point_in_time", score_early),
                "Lending Club sub-grade": ("grade", p_grade)}
    curves = {}
    for label, (tag, score) in rankings.items():
        curve = biz.tightening_curve(test, score)
        # Risk of the approved book at every cut level, measured with the main model's PDs.
        extra = []
        for share in curve["share_declined"]:
            approve = biz.decline_riskiest(score, float(share))
            row = {"expected_loss": float(p_gbm[approve] @ lid[approve])}
            for scen, rho in scenarios.items():
                key = "data" if scen == "data-implied" else "reference"
                row[f"capital_0.999_{key}"] = pf.semi_analytic(
                    p_gbm[approve], lid[approve], rho).risk_measures((0.999,))["capital_0.999"]
            extra.append(row)
        curves[label] = pd.concat([curve, pd.DataFrame(extra)], axis=1)
        curves[label].to_csv(cfg.TAB_DIR / f"tightening_curve_{tag}.csv", index=False)
        log(f"  tightening curve done: {label}")

    def policy(name: str, pd_: np.ndarray, share: float) -> dict:
        approve = biz.decline_riskiest(pd_, share)
        out = {"ranking": name, **biz.policy_outcome(test, approve)}
        out["funded_declined"] = float(funded[~approve].sum())
        # Risk of the approved book, always measured with the best PDs available.
        out["expected_loss"] = float(p_gbm[approve] @ lid[approve])
        for label, rho in scenarios.items():
            key = "data" if label == "data-implied" else "reference"
            out[f"capital_0.999_{key}"] = pf.semi_analytic(
                p_gbm[approve], lid[approve], rho).risk_measures((0.999,))["capital_0.999"]
        return out

    tightening = pd.DataFrame(
        [policy(name, score, share) for share in (0.0, 0.05, 0.10, 0.20)
         for name, (_, score) in rankings.items()])
    tightening.to_csv(cfg.TAB_DIR / "tightening_policies.csv", index=False)
    results["tightening"] = tightening.to_dict(orient="records")
    # The dashboard reads the curves; the README quotes the policy table. They must agree.
    worst = 0.0
    for _, pol in tightening.iterrows():
        curve = curves[pol["ranking"]]
        # The policy table stores the realised share (e.g. 14,139 / 282,787 loans), the curve
        # the nominal one (5%); they differ by less than one loan in the share.
        match = curve.loc[(curve["share_declined"] - pol["declined_share"]).abs() < 1.0 / len(test)]
        assert len(match) == 1, f"no unique curve point for share {pol['declined_share']}"
        point = match.iloc[0]
        for col in ("net_pnl", "principal_loss", "expected_loss",
                    "capital_0.999_data", "capital_0.999_reference"):
            worst = max(worst, abs(point[col] / pol[col] - 1) if pol[col] else abs(point[col]))
    check(results, "tightening curves match the policy table", worst, 1e-9)

    # Ex-ante rule: decline loans whose expected return is negative.
    rule = []
    for name, p in pds.items():
        expected_return = params.expected_return(test, p)
        approve = expected_return >= 0
        declined = test.loc[~approve]
        rule.append({"pd_model": name, **biz.policy_outcome(test, approve),
                     "funded_declined": float(declined["funded_amnt"].sum()),
                     "declined_default_rate": float(declined["default"].mean()),
                     "declined_net_pnl": float(declined["net_pnl"].sum()),
                     "declined_principal_loss": float(declined["loss_amount"].sum()),
                     "mean_expected_return": float(np.average(expected_return, weights=funded))})
    rule = pd.DataFrame(rule)
    rule.to_csv(cfg.TAB_DIR / "expected_return_rule.csv", index=False)
    results["expected_return_rule"] = rule.to_dict(orient="records")
    results["book_actual"] = biz.policy_outcome(test, np.ones(len(test), dtype=bool))

    # ---------------------------------------------------------------- figures
    log("Drawing figures")
    aucs = {name: pm.auc(y_test, p) for name, p in pds.items()}
    plots.model_performance(y_test, pds, calibration, aucs, cfg.FIG_DIR / "fig1_model_performance.png")
    plots.drivers(importance, cfg.FIG_DIR / "fig2_drivers.png")
    plots.cohort_default_rates(cohorts, rho_data, cfg.FIG_DIR / "fig3_cohort_default_rates.png")
    plots.loss_distribution(
        {f"Correlation {rho:.1%} ({label})": sims[label].losses for label, rho in scenarios.items()},
        expected_loss, realised_loss,
        {f"Correlation {rho:.1%} ({label})": portfolio[label]["monte_carlo"]["var_0.999"]
         for label, rho in scenarios.items()},
        cfg.FIG_DIR / "fig4_loss_distribution.png")
    plots.capital_vs_correlation(
        sweep, (mle["ci_low"], mle["ci_high"]),
        {"Estimated from\nLending Club vintages":
             (rho_data, portfolio["data-implied"]["monte_carlo"]["capital_0.999"]),
         "Basel reference\n(other retail)":
             (rho_ref, portfolio["Basel reference"]["monte_carlo"]["capital_0.999"])},
        cfg.FIG_DIR / "fig5_capital_vs_correlation.png")
    plots.grade_shares(grade_table, cfg.FIG_DIR / "fig6_risk_by_grade.png")
    plots.tightening(curves, cfg.FIG_DIR / "fig7_tightening.png")

    results["runtime_seconds"] = round(time.time() - T0)
    with open(cfg.OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    passed = sum(c["passed"] for c in results["checks"].values())
    log(f"Done. {passed}/{len(results['checks'])} self-checks passed. "
        f"Results in {cfg.OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", type=int, default=20_000,
                        help="Monte Carlo scenarios (default 20,000)")
    main(parser.parse_args().scenarios)
