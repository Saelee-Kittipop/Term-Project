#!/usr/bin/env python
"""Post-hoc analysis of a FROZEN run (no retraining, no threshold changes, nothing fed back into selection).

Adds, from the run's own saved artifacts:
  F11 ROC curves (test)                       <- test/test_forecasts.csv
  F12 log-RMSE by calendar year (test)        <- test/test_metrics_by_year.csv
  F13 forecast metrics by model (test)        <- test/test_metrics.csv
  F14 permutation importance (XAI, test)      <- frozen checkpoints + scalers, post_hoc/permutation_importance.csv
  F15 net CAGR / Sharpe vs cost (test)        <- backtest/backtest_summary.csv
  post_hoc/seed_stability.csv                 <- training/grid_summary.csv + test/test_metrics_per_seed.csv

Usage: python scripts/post_hoc_analysis.py --root <ROOT> --run-name formal_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volvix import pipeline as P  # noqa: E402
from volvix.config import load_config  # noqa: E402
from volvix.io_utils import git_commit, read_json, utc_now, write_csv, write_json  # noqa: E402
from volvix.plotting import COLORS, MODELS, FigureWriter  # noqa: E402
from volvix.training import ensemble_predict_log, windows_for_positions  # noqa: E402

DL = ("L_R", "L_RV", "T_R", "T_RV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--perm-repeats", type=int, default=10)
    a = ap.parse_args()
    paths = P.Paths(a.root, a.run_name)
    import yaml
    cfg = yaml.safe_load((paths.run / "config_frozen.yaml").read_text())
    fz = P.verify_freeze(paths)                        # refuse if any frozen artifact changed
    out = paths.run / "post_hoc"
    out.mkdir(exist_ok=True)
    w = FigureWriter(paths.figures, paths.run.name)
    kind = "formal:final_test:post_hoc"
    tf = pd.read_csv(paths.test / "test_forecasts.csv", parse_dates=["origin_date"])
    sc = tf[tf["score_eligible"]].copy()
    tau = fz["tau"]

    # F11 ROC ------------------------------------------------------------------------------
    from sklearn.metrics import roc_auc_score, roc_curve
    h = (sc["y"] > tau).to_numpy()
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in MODELS:
        fpr, tpr, thr = roc_curve(h, sc[m].to_numpy())
        auc = roc_auc_score(h, sc[m].to_numpy())
        ax.plot(fpr, tpr, color=COLORS[m], lw=1, label=f"{m} (AUC {auc:.3f})")
        rows.append(pd.DataFrame({"model": m, "fpr": fpr, "tpr": tpr, "threshold": thr, "auc": auc}))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Test ROC (score = forecast vol, positives = y > tau, n = {len(sc)})")
    ax.legend(fontsize=7)
    w.save(fig, "F11_test_roc", pd.concat(rows), "post_hoc_analysis.roc", "ROC curve บน test (เสริม ไม่ใช้แทน AP)", kind)

    # F12 by year --------------------------------------------------------------------------
    by = pd.read_csv(paths.test / "test_metrics_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    years = sorted(by["year"].unique())
    width = 0.11
    for k, m in enumerate(MODELS):
        g = by[by["model"] == m].set_index("year").reindex(years)
        ax.bar(np.arange(len(years)) + (k - 3) * width, g["log_RMSE"], width, color=COLORS[m], label=m)
    ax.set_xticks(range(len(years)), years)
    ax.set_ylabel("log-RMSE (lower is better)")
    ax.set_title("Test log-RMSE by calendar year of forecast origin")
    ax.legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    w.save(fig, "F12_test_logrmse_by_year", by[["model", "year", "n", "log_RMSE", "RMSE_pp", "QLIKE", "alert_F1", "alert_prevalence"]],
           "post_hoc_analysis.by_year", "log-RMSE รายปีบน test", kind)

    # F13 metrics by model -----------------------------------------------------------------
    tm = pd.read_csv(paths.test / "test_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, lab in zip(axes, ["log_RMSE", "RMSE_pp", "QLIKE"], ["log-RMSE", "RMSE (pp)", "QLIKE"]):
        ax.bar(tm["model"], tm[col], color=[COLORS[m] for m in tm["model"]])
        lo = tm[col].min()
        hi = tm[col].max()
        pad = (hi - lo) * 0.6 if hi > lo else 1
        ax.set_ylim(lo - pad, hi + pad * 0.5)
        ax.set_title(f"{lab} (lower is better)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.suptitle(f"Test forecast metrics, n = {int(tm['n'].iloc[0])} origins (axis does not start at zero)")
    fig.tight_layout()
    w.save(fig, "F13_test_metrics_by_model", tm[["model", "n", "MAE_pp", "RMSE_pp", "log_RMSE", "QLIKE"]],
           "post_hoc_analysis.metrics", "ตัวชี้วัดการพยากรณ์บน test ตามโมเดล", kind)

    # F14 permutation importance (XAI) -----------------------------------------------------
    panel = pd.read_parquet(paths.processed / "panel.parquet")
    samples = pd.read_parquet(paths.dataset / "samples.parquet")
    test = samples[samples["split"] == "test"]
    pos = test["pos"].to_numpy()
    z_true = test["z"].to_numpy()
    rng = np.random.default_rng(2026)
    rows = []
    for code in DL:
        arch, fs = P.CODE_TO_ARCH_FS[code]
        X = windows_for_positions(panel, cfg, fs, paths.scalers, pos)
        base_z, _ = ensemble_predict_log(cfg, fz["selection"], arch, fs, X, paths.training)
        base = float(np.mean((base_z - z_true) ** 2))
        names = cfg["data"]["tickers"] + (["VIX_lag1"] if fs == "RV" else [])
        for j, name in enumerate(names):
            inc = []
            for _ in range(a.perm_repeats):
                Xp = X.copy()
                Xp[:, :, j] = X[rng.permutation(len(X)), :, j]   # swap whole 60-day series of one input across origins
                zp, _ = ensemble_predict_log(cfg, fz["selection"], arch, fs, Xp, paths.training)
                inc.append(float(np.mean((zp - z_true) ** 2)) - base)
            rows.append({"model": code, "feature": name, "base_logmse": base, "mean_increase_logmse": float(np.mean(inc)),
                         "sd_increase_logmse": float(np.std(inc, ddof=1)), "repeats": a.perm_repeats})
        print(f"[xai] {code} base log-MSE {base:.4f}")
    pi = pd.DataFrame(rows)
    write_csv(out / "permutation_importance.csv", pi)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, code in zip(axes, DL):
        g = pi[pi["model"] == code].sort_values("mean_increase_logmse")
        ax.barh(g["feature"], g["mean_increase_logmse"], xerr=g["sd_increase_logmse"], color=COLORS[code])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(code, fontsize=9)
        ax.set_xlabel("increase in test log-MSE")
    fig.suptitle(f"Permutation importance on test (ensemble, {a.perm_repeats} shuffles; post-hoc, no retraining)")
    fig.tight_layout()
    w.save(fig, "F14_permutation_importance", pi, "post_hoc_analysis.permutation_importance",
           "ความสำคัญของ input แบบ permutation (XAI หลังการทดลอง)", kind)

    # F15 cost sensitivity -----------------------------------------------------------------
    bs = pd.read_csv(paths.backtest / "backtest_summary.csv")
    keep = ["Reference_fixed", "NoTilt_ACWI75_SHY20_GLD5"] + [f"{m}_overlay" for m in MODELS]
    d = bs[bs["strategy"].isin(keep)][["strategy", "cost_bps", "net_CAGR", "sharpe", "ann_vol", "max_drawdown", "turnover_per_year"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, g in d.groupby("strategy"):
        key = s.replace("_overlay", "")
        style = dict(color=COLORS.get(key, "#333333"), lw=2.2 if s == "Reference_fixed" else 1, marker="o", ms=3)
        axes[0].plot(g["cost_bps"], g["net_CAGR"] * 100, label=s, **style)
        axes[1].plot(g["cost_bps"], g["sharpe"], **{**style})
    axes[0].set_ylabel("net CAGR (%)")
    axes[1].set_ylabel("Sharpe (excess over daily RF)")
    for ax in axes:
        ax.set_xlabel("cost per side (bps)")
        ax.set_xticks([5, 10, 20])
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Test backtest 2020-01-03 to 2025-12-31: sensitivity to transaction costs")
    fig.tight_layout()
    w.save(fig, "F15_cost_sensitivity", d, "post_hoc_analysis.cost_sensitivity", "ความไวต่อต้นทุนซื้อขาย", kind)

    # seed stability -----------------------------------------------------------------------
    g = pd.read_csv(paths.training / "grid_summary.csv")
    ps = pd.read_csv(paths.test / "test_metrics_per_seed.csv")
    ps["model"] = ps["model_seed"].str.rsplit("_s", n=1).str[0]
    stab = []
    for code in DL:
        arch, fs = P.CODE_TO_ARCH_FS[code]
        s = fz["selection"][arch]
        sel = g[(g["arch"] == arch) & (g["feature_set"] == fs) & np.isclose(g["lr"], s["lr"]) & np.isclose(g["dropout"], s["dropout"])]
        p = ps[ps["model"] == code]
        stab.append({"model": code, "lr": s["lr"], "dropout": s["dropout"], "n_seeds": len(sel),
                     "val_logmse_mean": sel["best_val_logmse"].mean(), "val_logmse_sd": sel["best_val_logmse"].std(ddof=1),
                     "best_epoch_mean": sel["best_epoch"].mean(), "best_epoch_min": sel["best_epoch"].min(),
                     "best_epoch_max": sel["best_epoch"].max(), "n_params": int(sel["n_params"].iloc[0]),
                     "train_time_sec_mean": sel["train_time_sec"].mean(),
                     "test_logRMSE_seed_mean": p["log_RMSE"].mean(), "test_logRMSE_seed_sd": p["log_RMSE"].std(ddof=1),
                     "test_logRMSE_seed_min": p["log_RMSE"].min(), "test_logRMSE_seed_max": p["log_RMSE"].max()})
    write_csv(out / "seed_stability.csv", pd.DataFrame(stab))

    # append figure manifest rows
    old = pd.read_csv(paths.figures / "figure_manifest.csv")
    new = pd.DataFrame(w.rows)
    new["script"] = "scripts/post_hoc_analysis.py"
    allm = pd.concat([old[~old["figure_id"].isin(new["figure_id"])], new], ignore_index=True)
    write_csv(paths.figures / "figure_manifest.csv", allm)
    write_json(out / "post_hoc_manifest.json", {"created_utc": utc_now(), "git_commit": git_commit(), "run": paths.run.name,
                                                "note": "post-hoc only: no retraining, thresholds/cutoffs/rules unchanged",
                                                "freeze_verified": True, "perm_repeats": a.perm_repeats,
                                                "figures": new["figure_id"].tolist()})
    print("done", new["figure_id"].tolist())


if __name__ == "__main__":
    main()
