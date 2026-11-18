"""Figures: every figure is saved as PNG + SVG together with the exact CSV used to draw it, and is listed in
figures/figure_manifest.csv (figure -> data CSV -> script/function -> run id -> git commit)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import git_commit, read_json, utc_now, write_csv

COLORS = {"HV20": "#8c8c8c", "EWMA": "#b07aa1", "GARCH": "#9c755f", "L_R": "#4e79a7", "L_RV": "#a0cbe8",
          "T_R": "#e15759", "T_RV": "#ff9d9a", "Reference_fixed": "#000000", "NoTilt_ACWI75_SHY20_GLD5": "#59a14f"}
MODELS = ("HV20", "EWMA", "GARCH", "L_R", "L_RV", "T_R", "T_RV")


class FigureWriter:
    def __init__(self, fig_dir: Path, run_id: str):
        self.dir, self.run_id, self.rows = fig_dir, run_id, []
        (fig_dir / "data").mkdir(parents=True, exist_ok=True)
        self.commit = git_commit()
        self.synthetic = False

    def save(self, fig, fig_id: str, data: pd.DataFrame, func: str, title_th: str, result_kind: str):
        if self.synthetic:
            fig.text(0.5, 0.5, "SYNTHETIC DATA - NOT A RESULT", ha="center", va="center", fontsize=28,
                     color="red", alpha=0.25, rotation=20, transform=fig.transFigure)
            result_kind = "SYNTHETIC:" + result_kind
        png, svg, csv = self.dir / f"{fig_id}.png", self.dir / f"{fig_id}.svg", self.dir / "data" / f"{fig_id}.csv"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        write_csv(csv, data)
        self.rows.append({"figure_id": fig_id, "png": png.name, "svg": svg.name, "data_csv": f"data/{csv.name}",
                          "script": "src/volvix/plotting.py", "function": func, "run_id": self.run_id,
                          "git_commit": self.commit, "result_kind": result_kind, "title_th": title_th,
                          "created_utc": utc_now(), "status": "created"})

    def skip(self, fig_id: str, reason: str):
        self.rows.append({"figure_id": fig_id, "status": f"skipped: {reason}", "run_id": self.run_id})

    def manifest(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        write_csv(self.dir / "figure_manifest.csv", df)
        return df


def _split_lines(ax, cfg):
    for s in ("validation", "test"):
        ax.axvline(pd.Timestamp(cfg["splits"][s]["start"]), color="k", ls="--", lw=0.8)


def fig_data_overview(w, cfg, panel, samples, tau, kind):
    d = pd.DataFrame({"date": panel.index, "r_ref": panel["r_ref"].to_numpy(), "vix": panel["vix"].to_numpy()})
    d = d.merge(samples[["origin_date", "y", "split"]].rename(columns={"origin_date": "date"}), on="date", how="left")
    d["tau_train_p90"] = tau
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(d["date"], d["r_ref"] * 100, lw=0.5, color="#4e79a7")
    ax[0].set_ylabel("Reference portfolio\ndaily return (%)")
    ax[1].plot(d["date"], d["y"] * 100, lw=0.6, color="#e15759")
    ax[1].axhline(tau * 100, color="k", lw=0.8, label=f"tau = train P90 = {tau*100:.2f}%")
    ax[1].set_ylabel("Target y: annualized\nfwd 5-day SD (%)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(d["date"], d["vix"], lw=0.6, color="#59a14f")
    ax[2].set_ylabel("VIX (index pts)")
    for a in ax:
        _split_lines(a, cfg)
    ax[0].set_title(f"Data overview, train | validation | test (n sessions = {len(d)})")
    w.save(fig, "F01_data_overview", d, "fig_data_overview", "ผลตอบแทนพอร์ต target และ VIX พร้อมเส้นแบ่งช่วงและ tau", kind)


def fig_train_corr(w, cfg, panel, samples, kind):
    tick = cfg["data"]["tickers"]
    end = pd.Timestamp(cfg["splits"]["train"]["end"])
    r = panel.loc[(panel.index <= end) & panel["ret_valid"], [f"ret_{t}" for t in tick]]
    r.columns = tick
    c = r.corr()
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(c.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(tick)), tick)
    ax.set_yticks(range(len(tick)), tick)
    for i in range(len(tick)):
        for j in range(len(tick)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Training-period return correlation (n = {len(r)} days)")
    w.save(fig, "F02_train_return_correlation", c.reset_index().rename(columns={"index": "ticker"}),
           "fig_train_corr", "สหสัมพันธ์ผลตอบแทนช่วง training", kind)


def fig_learning_curves(w, cfg, paths, selection, kind):
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for i, arch in enumerate(("LSTM", "Transformer")):
        s = selection[arch]
        for j, fs in enumerate(("R", "RV")):
            ax = axes[i, j]
            for seed in cfg["training"]["seeds"]:
                rid = f"{arch}_{fs}_lr{s['lr']:g}_do{s['dropout']:g}_s{seed}"
                h = pd.read_csv(paths.training / rid / "history.csv")
                h["run_id"] = rid
                rows.append(h)
                ax.plot(h["epoch"], h["train_logmse_eval"], lw=0.8, ls="--")
                ax.plot(h["epoch"], h["val_logmse"], lw=1.2, label=f"seed {seed} val")
            ax.set_title(f"{arch} {fs} (lr={s['lr']:g}, dropout={s['dropout']:g})")
            ax.set_xlabel("epoch")
            ax.set_ylabel("log-MSE")
            ax.legend(fontsize=7)
    fig.suptitle("Learning curves (dashed = training, solid = validation)")
    w.save(fig, "F03_learning_curves", pd.concat(rows), "fig_learning_curves", "Learning curves ของ config ที่เลือก", kind)


def fig_grid(w, cfg, paths, kind):
    g = pd.read_csv(paths.training / "grid_summary.csv")
    agg = g.groupby(["arch", "lr", "dropout"])["best_val_logmse"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{a}\nlr={l:g} do={d:g}" for a, l, d in agg[["arch", "lr", "dropout"]].itertuples(index=False)]
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], color=["#4e79a7" if a == "LSTM" else "#e15759" for a in agg["arch"]])
    ax.set_xticks(range(len(agg)), labels, fontsize=7)
    ax.set_ylabel("mean validation log-MSE\n(2 feature sets x seeds)")
    ax.set_title("Hyperparameter selection on validation")
    w.save(fig, "F10_validation_grid", agg, "fig_grid", "ผลการเลือก hyperparameter บน validation", kind)


def fig_forecasts(w, tf, kind):
    sc = tf[tf["score_eligible"]].copy()
    cols = ["origin_date", "y"] + [m for m in MODELS if m in sc]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 14), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.plot(sc["origin_date"], sc["y"] * 100, color="k", lw=0.5, label="actual")
        ax.plot(sc["origin_date"], sc[m] * 100, color=COLORS[m], lw=0.9, label=m)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_ylabel("%")
    axes[0].set_title(f"Test: actual vs forecast annualized fwd 5-day SD (n = {len(sc)} origins)")
    w.save(fig, "F04_test_forecast_vs_actual", sc[cols], "fig_forecasts", "ค่าจริงเทียบค่าพยากรณ์บน test", kind)


def fig_vix_diff(w, tf, kind):
    sc = tf[tf["score_eligible"]].copy()
    d = pd.DataFrame({"origin_date": sc["origin_date"], "LSTM_RV_minus_R_pp": (sc["L_RV"] - sc["L_R"]) * 100,
                      "Transformer_RV_minus_R_pp": (sc["T_RV"] - sc["T_R"]) * 100})
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(d["origin_date"], d["LSTM_RV_minus_R_pp"], lw=0.7, color="#4e79a7")
    ax[1].plot(d["origin_date"], d["Transformer_RV_minus_R_pp"], lw=0.7, color="#e15759")
    for a, t in zip(ax, ("LSTM: forecast(RV) - forecast(R)", "Transformer: forecast(RV) - forecast(R)")):
        a.axhline(0, color="k", lw=0.6)
        a.set_title(t, fontsize=9)
        a.set_ylabel("pp")
    w.save(fig, "F05_vix_forecast_difference", d, "fig_vix_diff", "ผลต่าง forecast จากการเพิ่ม VIX ภายใน architecture", kind)


def fig_confusion(w, tf, tau, cut, kind):
    sc = tf[tf["score_eligible"]]
    rows = []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 2.8))
    for ax, m in zip(axes, MODELS):
        c = cut[m]["cutoff"]
        h, a = sc["y"] > tau, sc[m] > c
        mat = np.array([[int((h & a).sum()), int((h & ~a).sum())], [int((~h & a).sum()), int((~h & ~a).sum())]])
        rows.append({"model": m, "cutoff": c, "TP": mat[0, 0], "FN": mat[0, 1], "FP": mat[1, 0], "TN": mat[1, 1]})
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{['TP','FN','FP','TN'][2*i+j]}\n{mat[i, j]}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["alert", "no alert"], fontsize=7)
        ax.set_yticks([0, 1], ["high", "normal"], fontsize=7)
        ax.set_title(m, fontsize=9)
    fig.suptitle(f"Test confusion matrices (tau = {tau*100:.2f}%, n = {len(sc)})")
    w.save(fig, "F06_test_confusion_matrices", pd.DataFrame(rows), "fig_confusion", "Confusion matrix บน test", kind)


def fig_pr(w, tf, tau, kind):
    from sklearn.metrics import precision_recall_curve
    sc = tf[tf["score_eligible"]]
    h = (sc["y"] > tau).to_numpy()
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))
    if h.any():
        for m in MODELS:
            p, r, thr = precision_recall_curve(h, sc[m].to_numpy())
            ax.plot(r, p, color=COLORS[m], lw=1, label=m)
            rows.append(pd.DataFrame({"model": m, "precision": p, "recall": r, "threshold": np.append(thr, np.nan)}))
        ax.axhline(h.mean(), color="k", ls=":", label=f"prevalence = {h.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7)
    ax.set_title(f"Test precision-recall (score = forecast vol, n = {len(sc)})")
    w.save(fig, "F07_test_precision_recall", pd.concat(rows) if rows else pd.DataFrame(), "fig_pr", "Precision-Recall curve บน test", kind)


def fig_nav(w, daily, base_cost, kind):
    d = daily[daily["cost_bps"] == base_cost]
    keep = [s for s in d["strategy"].unique() if s.endswith("_overlay") or s in ("Reference_fixed", "NoTilt_ACWI75_SHY20_GLD5")]
    d = d[d["strategy"].isin(keep)].copy()
    d["drawdown"] = d.groupby("strategy")["nav"].transform(lambda x: x / x.cummax() - 1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for s, g in d.groupby("strategy"):
        key = s.replace("_overlay", "")
        ax[0].plot(g["date"], g["nav"], lw=1.4 if key in ("Reference_fixed",) else 0.9, color=COLORS.get(key), label=s)
        ax[1].plot(g["date"], g["drawdown"] * 100, lw=0.8, color=COLORS.get(key))
    ax[0].set_ylabel("NAV after costs (start 100)")
    ax[1].set_ylabel("Drawdown (%)")
    ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_title(f"Test backtest, cost = {base_cost} bps per side, signal t -> trade t+1")
    w.save(fig, "F08_nav_and_drawdown", d[["strategy", "date", "nav", "drawdown"]], "fig_nav", "NAV หลังต้นทุนและ drawdown", kind)


def fig_weights(w, cfg, daily, base_cost, kind):
    d = daily[(daily["cost_bps"] == base_cost) & daily["strategy"].str.endswith("_overlay")].copy()
    d["w_SHY"] = d["w_post_SHY"]
    d["w_equity_ACWI_XLK_XLV"] = d["w_post_ACWI"] + d["w_post_XLK"] + d["w_post_XLV"]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 12), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        g = d[d["strategy"] == f"{m}_overlay"]
        ax.plot(g["date"], g["w_SHY"], lw=0.7, color="#4e79a7", label="SHY")
        ax.plot(g["date"], g["w_equity_ACWI_XLK_XLV"], lw=0.7, color="#e15759", label="equity")
        ax.set_title(m, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    axes[0].set_ylim(0, 1)
    w.save(fig, "F09_overlay_weights", d[["strategy", "date", "w_SHY", "w_equity_ACWI_XLK_XLV"]], "fig_weights",
           "น้ำหนัก SHY และหุ้นตามเวลาของ overlay", kind)


def make_all_figures(cfg, paths, run_id: str) -> pd.DataFrame:
    w = FigureWriter(paths.figures, run_id)
    w.synthetic = bool(read_json(paths.run / "run_manifest.json").get("data", {}).get("snapshot_synthetic", False))
    kind_val = f"{cfg.get('run_kind')}:validation"
    kind_test = f"{cfg.get('run_kind')}:final_test"
    panel = pd.read_parquet(paths.processed / "panel.parquet")
    samples = pd.read_parquet(paths.dataset / "samples.parquet")
    tr = samples.loc[samples["split"] == "train", "y"]
    tau = float(np.quantile(tr, cfg["alerts"]["tau_quantile"]))
    fig_data_overview(w, cfg, panel, samples, tau, f"{cfg.get('run_kind')}:data")
    fig_train_corr(w, cfg, panel, samples, f"{cfg.get('run_kind')}:data")
    if (paths.training / "grid_summary.csv").exists():
        fig_grid(w, cfg, paths, kind_val)
    if (paths.frozen / "freeze.json").exists():
        fz = read_json(paths.frozen / "freeze.json")
        fig_learning_curves(w, cfg, paths, fz["selection"], kind_val)
        if (paths.test / "test_forecasts.csv").exists():
            tf = pd.read_csv(paths.test / "test_forecasts.csv", parse_dates=["origin_date"])
            fig_forecasts(w, tf, kind_test)
            fig_vix_diff(w, tf, kind_test)
            fig_confusion(w, tf, fz["tau"], fz["cutoffs"], kind_test)
            fig_pr(w, tf, fz["tau"], kind_test)
        else:
            w.skip("F04-F07", "test stage not run")
        if (paths.backtest / "backtest_daily.csv").exists():
            daily = pd.read_csv(paths.backtest / "backtest_daily.csv", parse_dates=["date"])
            fig_nav(w, daily, cfg["backtest"]["cost_bps_base"], kind_test)
            fig_weights(w, cfg, daily, cfg["backtest"]["cost_bps_base"], kind_test)
        else:
            w.skip("F08-F09", "backtest stage not run")
    else:
        w.skip("F03-F09", "freeze stage not run")
    return w.manifest()
