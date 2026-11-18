"""Stage orchestration. Every stage reads/writes files under one run directory, so the pipeline is resumable.

Stages (in order): data -> baselines -> train -> freeze -> test -> backtest -> figures -> report
`smoke` runs (configs/smoke.yaml) may run data/baselines/train/freeze(validation only) but never test.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import backtest as bt
from .baselines import all_baselines
from .config import config_hash, weights_vector
from .data.clean import clean_snapshot
from .data.dataset import build_samples, save_dataset_artifacts
from .data.download import create_snapshot, latest_snapshot, verify_snapshot
from .data.synthetic import make_synthetic_snapshot
from .evaluation import (alert_metrics, diebold_mariano, forecast_metrics, loss_series, paired_block_bootstrap,
                         select_cutoff, tau_from_train, var_backtest)
from .io_utils import (RunManifest, environment_info, pip_freeze, read_json, sha256_file, utc_now, write_csv,
                       write_json)
from .training import (ARCHS, FEATURE_SETS, MODEL_CODES, ensemble_predict_log, ensemble_runs, fit_scalers,
                       grid_summary, load_split_arrays, run_grid, select_configs, to_vol, windows_for_positions)

BASELINES = ("HV20", "EWMA", "GARCH")
DL_MODELS = ("L_R", "L_RV", "T_R", "T_RV")
ALL_MODELS = BASELINES + DL_MODELS
CODE_TO_ARCH_FS = {v: k for k, v in MODEL_CODES.items()}


class Paths:
    def __init__(self, root: str | Path, run_name: str):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.run = self.root / "runs" / run_name
        for name in ["processed", "dataset", "scalers", "baselines", "training", "frozen", "validation", "test",
                     "backtest", "figures", "reports", "environment"]:
            setattr(self, name, self.run / name)

    def mkdirs(self):
        for v in vars(self).values():
            if isinstance(v, Path) and v != self.root and v != self.data_dir:
                v.mkdir(parents=True, exist_ok=True)


def _load(paths: Paths):
    panel = pd.read_parquet(paths.processed / "panel.parquet")
    samples = pd.read_parquet(paths.dataset / "samples.parquet")
    return panel, samples


def init_run(cfg: dict, paths: Paths) -> RunManifest:
    paths.mkdirs()
    frozen_cfg = paths.run / "config_frozen.yaml"
    if frozen_cfg.exists():
        old = yaml.safe_load(frozen_cfg.read_text())
        if config_hash(old) != config_hash(cfg):
            raise RuntimeError("config differs from the one frozen for this run directory; use a new run name")
    else:
        frozen_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    m = RunManifest(paths.run / "run_manifest.json")
    m.set(run_dir=str(paths.run), run_kind=cfg.get("run_kind", "formal"), config_hash=config_hash(cfg),
          config_sha256=sha256_file(frozen_cfg), seeds=cfg["training"]["seeds"], environment=environment_info())
    pip_freeze(paths.environment / "pip_freeze.txt")
    write_json(paths.environment / "environment.json", environment_info())
    return m


# ------------------------------------------------------------------------------------------ data
def stage_data(cfg, paths: Paths, snapshot: str = "latest", manual_dir=None, log=print) -> dict:
    m = init_run(cfg, paths)
    if m.completed("data") and (paths.dataset / "samples.parquet").exists():
        # never rebuild data/scalers under existing checkpoints; use a new run name for new data
        log("[data] already completed for this run - skipped (use a new --run-name to rebuild)")
        return {"skipped": True}
    with m.stage("data") as st:
        if snapshot == "new":
            snap = create_snapshot(cfg, paths.data_dir, manual_dir=manual_dir)
        elif snapshot.startswith("files:"):
            from .data.from_files import create_snapshot_from_files
            snap = create_snapshot_from_files(cfg, paths.data_dir, snapshot.split(":", 1)[1])
        elif snapshot == "synthetic":
            snap = paths.data_dir / "raw" / "SYNTHETIC"
            if not (snap / "snapshot_metadata.json").exists():
                snap = make_synthetic_snapshot(cfg, paths.data_dir)
        elif snapshot == "latest":
            snap = latest_snapshot(paths.data_dir)
            if snap is None:
                snap = create_snapshot(cfg, paths.data_dir, manual_dir=manual_dir)
        else:
            snap = paths.data_dir / "raw" / snapshot
        meta = verify_snapshot(snap)
        if cfg.get("run_kind") == "formal" and meta.get("SYNTHETIC"):
            raise RuntimeError("formal runs must use a real data snapshot, not SYNTHETIC")
        log(f"[data] snapshot {snap.name} verified ({len(meta['files'])} files)")
        summary = clean_snapshot(cfg, snap, paths.processed)
        panel = pd.read_parquet(paths.processed / "panel.parquet")
        samples = build_samples(panel, cfg)
        res = save_dataset_artifacts(samples, panel, cfg, paths.dataset)
        scalers = fit_scalers(panel, samples, cfg, paths.scalers)
        from .reporting import data_quality_report_th
        data_quality_report_th(paths, cfg)
        hashes = {"snapshot_id": snap.name, "snapshot_synthetic": bool(meta.get("SYNTHETIC", False)),
                  "snapshot_files_sha256": {k: v["sha256"] for k, v in meta["files"].items()},
                  "panel_sha256": sha256_file(paths.processed / "panel.parquet"),
                  "samples_sha256": sha256_file(paths.dataset / "samples.parquet")}
        m.set(data=hashes)
        st.extra.update({"alignment_checks_all_passed": res["alignment_checks_all_passed"], **hashes})
        if not res["alignment_checks_all_passed"]:
            raise RuntimeError(f"alignment checks failed: {res['failed_checks']} - do not train")
    return {"quality": summary, "split": res, "scalers": scalers}


# ------------------------------------------------------------------------------------------ baselines
def stage_baselines(cfg, paths: Paths, log=print) -> dict:
    m = RunManifest(paths.run / "run_manifest.json")
    if m.completed("baselines") and (paths.baselines / "baseline_forecasts.parquet").exists():
        log("[baselines] already completed for this run - skipped")
        return read_json(paths.baselines / "baseline_info.json")
    with m.stage("baselines") as st:
        panel, _ = _load(paths)
        fc, info = all_baselines(panel, cfg)
        fc.to_parquet(paths.baselines / "baseline_forecasts.parquet")
        write_csv(paths.baselines / "baseline_forecasts.csv", fc.reset_index())
        write_json(paths.baselines / "baseline_info.json", info)
        st.extra["garch_converged"] = info["garch"]["converged"]
        log(f"[baselines] GARCH converged={info['garch']['converged']} params={info['garch'].get('params')}")
    return info


# ------------------------------------------------------------------------------------------ train
def stage_train(cfg, paths: Paths, max_runs=None, only=None, log=print) -> pd.DataFrame:
    m = RunManifest(paths.run / "run_manifest.json")
    with m.stage("train") as st:
        panel, samples = _load(paths)
        grid = run_grid(panel, samples, cfg, paths.training, paths.scalers, only=only, max_runs=max_runs, log=log)
        st.extra.update({"runs_completed": int((grid["status"] == "completed").sum()), "runs_total": int(len(grid))})
    return grid


# ------------------------------------------------------------------------------------------ freeze
def _dl_forecasts(cfg, selection, panel, positions, paths) -> tuple[dict, dict]:
    eps = cfg["target"]["log_epsilon"]
    ens, seeds = {}, {}
    for code in DL_MODELS:
        arch, fs = CODE_TO_ARCH_FS[code]
        X = windows_for_positions(panel, cfg, fs, paths.scalers, positions)
        z, per = ensemble_predict_log(cfg, selection, arch, fs, X, paths.training)
        ens[code] = to_vol(z, eps)
        for spec, zz in zip(ensemble_runs(cfg, selection, arch, fs), per):
            seeds[f"{code}_s{spec.seed}"] = to_vol(zz, eps)
    return ens, seeds


def stage_freeze(cfg, paths: Paths, log=print) -> dict:
    """Select configs, compute tau / validation forecasts / cutoffs / static-comparator weights, then freeze."""
    m = RunManifest(paths.run / "run_manifest.json")
    with m.stage("freeze") as st:
        panel, samples = _load(paths)
        grid = grid_summary(cfg, paths.training)
        selection = select_configs(cfg, grid)
        write_json(paths.frozen / "selection.json", selection)
        if any(v["status"] != "selected" for v in selection.values()):
            raise RuntimeError("grid incomplete - finish training before freezing")
        tr = samples[samples["split"] == "train"]
        va = samples[samples["split"] == "validation"]
        tau = tau_from_train(tr["y"].to_numpy(), cfg["alerts"]["tau_quantile"])
        base = pd.read_parquet(paths.baselines / "baseline_forecasts.parquet")
        ens, seeds = _dl_forecasts(cfg, selection, panel, va["pos"].to_numpy(), paths)
        vf = pd.DataFrame({"origin_date": va["origin_date"].to_numpy(), "target_end_date": va["target_end_date"].to_numpy(),
                           "y": va["y"].to_numpy()})
        for b in BASELINES:
            vf[b] = base.loc[va["origin_date"], b].to_numpy()
        for k, v in {**ens, **seeds}.items():
            vf[k] = v
        write_csv(paths.validation / "validation_forecasts.csv", vf)
        y, h = vf["y"].to_numpy(), vf["y"].to_numpy() > tau
        cut, rows = {}, []
        for mname in ALL_MODELS:
            f = vf[mname].to_numpy()
            if not np.isfinite(f).all():
                cut[mname] = {"cutoff": None, "rule": "unavailable (non-finite forecasts)", "q_val_alert_freq": None}
                continue
            c = select_cutoff(f, h, tau)
            c["q_val_alert_freq"] = float((f > c["cutoff"]).mean())
            cut[mname] = c
            rows.append({"model": mname, "split": "validation", **forecast_metrics(y, f, cfg["target"]["log_epsilon"]),
                         **{f"alert_{k}": v for k, v in alert_metrics(y, f, tau, c["cutoff"]).items()}})
        write_csv(paths.validation / "validation_metrics.csv", pd.DataFrame(rows))
        # per-seed validation metrics (no seed is chosen; all are reported)
        seed_rows = [{"model_seed": k, **forecast_metrics(y, vf[k].to_numpy(), cfg["target"]["log_epsilon"])} for k in seeds]
        write_csv(paths.validation / "validation_metrics_per_seed.csv", pd.DataFrame(seed_rows))
        w_n, w_a = weights_vector(cfg, "reference_weights"), weights_vector(cfg, "alert_weights")
        static = {mname: list(((1 - c["q_val_alert_freq"]) * w_n + c["q_val_alert_freq"] * w_a))
                  for mname, c in cut.items() if c["q_val_alert_freq"] is not None}
        sel_specs = [s for a in ARCHS for fs in FEATURE_SETS for s in ensemble_runs(cfg, selection, a, fs)]
        hashes = {"config_frozen.yaml": sha256_file(paths.run / "config_frozen.yaml"),
                  "panel.parquet": sha256_file(paths.processed / "panel.parquet"),
                  "samples.parquet": sha256_file(paths.dataset / "samples.parquet"),
                  "scaler_R.json": sha256_file(paths.scalers / "scaler_R.json"),
                  "scaler_RV.json": sha256_file(paths.scalers / "scaler_RV.json"),
                  "baseline_forecasts.parquet": sha256_file(paths.baselines / "baseline_forecasts.parquet"),
                  **{f"checkpoint:{s.run_id}": sha256_file(paths.training / s.run_id / "best.pt") for s in sel_specs}}
        freeze = {"frozen_utc": utc_now(), "run_kind": cfg.get("run_kind"), "tau": tau, "tau_quantile": cfg["alerts"]["tau_quantile"],
                  "n_train_targets": int(len(tr)), "validation_high_risk_share": float(h.mean()),
                  "selection": {a: {k: v for k, v in s.items() if k != "table"} for a, s in selection.items()},
                  "cutoffs": cut, "static_comparator_weights": static, "tickers": cfg["data"]["tickers"],
                  "hashes": hashes, "git_commit": m.data["stages"][-1]["git_commit"]}
        write_json(paths.frozen / "freeze.json", freeze)
        st.extra.update({"tau": tau, "selection": freeze["selection"]})
        log(f"[freeze] tau={tau:.4f} selection={freeze['selection']}")
    return freeze


def verify_freeze(paths: Paths) -> dict:
    fz = read_json(paths.frozen / "freeze.json")
    for name, h in fz["hashes"].items():
        if name.startswith("checkpoint:"):
            p = paths.training / name.split(":", 1)[1] / "best.pt"
        else:
            p = {"config_frozen.yaml": paths.run / "config_frozen.yaml", "panel.parquet": paths.processed / "panel.parquet",
                 "samples.parquet": paths.dataset / "samples.parquet", "scaler_R.json": paths.scalers / "scaler_R.json",
                 "scaler_RV.json": paths.scalers / "scaler_RV.json",
                 "baseline_forecasts.parquet": paths.baselines / "baseline_forecasts.parquet"}[name]
        if sha256_file(p) != h:
            raise RuntimeError(f"frozen artifact changed after freeze: {name}")
    return fz


# ------------------------------------------------------------------------------------------ test
def stage_test(cfg, paths: Paths, allow_rerun_reason: str | None = None, log=print) -> dict:
    synthetic = RunManifest(paths.run / "run_manifest.json").data.get("data", {}).get("snapshot_synthetic", True)
    kind = cfg.get("run_kind")
    if not ((kind == "formal" and not synthetic) or (kind == "synthetic_e2e" and synthetic)):
        raise RuntimeError("test evaluation is allowed only for formal runs on real data, or for code-path "
                           "checks on SYNTHETIC data (run_kind=synthetic_e2e). Smoke runs stop at validation.")
    fz = verify_freeze(paths)
    marker = paths.test / "TEST_OPENED.json"
    if marker.exists():
        if not allow_rerun_reason:
            raise RuntimeError("test set already opened for this run. Re-running requires an explicit reason "
                               "(implementation fix only, never tuning) via allow_rerun_reason")
        prev = read_json(marker)
        prev.setdefault("reruns", []).append({"utc": utc_now(), "reason": allow_rerun_reason})
        write_json(marker, prev)
    else:
        write_json(marker, {"opened_utc": utc_now(), "freeze_utc": fz["frozen_utc"], "note": "single use of test set"})
    m = RunManifest(paths.run / "run_manifest.json")
    eps = cfg["target"]["log_epsilon"]
    with m.stage("test") as st:
        panel, samples = _load(paths)
        trade = samples[samples["trade_eligible"]].copy()
        base = pd.read_parquet(paths.baselines / "baseline_forecasts.parquet")
        ens, seeds = _dl_forecasts(cfg, fz["selection"], panel, trade["pos"].to_numpy(), paths)
        tf = pd.DataFrame({"origin_date": trade["origin_date"].to_numpy(), "target_end_date": trade["target_end_date"].to_numpy(),
                           "y": trade["y"].where(trade["split"] == "test").to_numpy(),
                           "score_eligible": (trade["split"] == "test").to_numpy()})
        nxt = {d: panel.index[i + 1] for i, d in enumerate(panel.index[:-1])}
        tf["trade_date"] = tf["origin_date"].map(nxt)
        for b in BASELINES:
            tf[b] = base.loc[trade["origin_date"], b].to_numpy()
        for k, v in {**ens, **seeds}.items():
            tf[k] = v
        tau = fz["tau"]
        for mname in ALL_MODELS:
            c = fz["cutoffs"][mname]["cutoff"]
            tf[f"alert_{mname}"] = (tf[mname] > c) if c is not None else np.nan
        write_csv(paths.test / "test_forecasts.csv", tf)
        sc = tf[tf["score_eligible"]].copy()
        y = sc["y"].to_numpy()
        rows, yrows, srows = [], [], []
        hv_low = sc["HV20"].to_numpy() <= tau
        for mname in ALL_MODELS:
            c = fz["cutoffs"][mname]["cutoff"]
            f = sc[mname].to_numpy()
            if c is None or not np.isfinite(f).all():
                rows.append({"model": mname, "status": "unavailable"})
                continue
            rows.append({"model": mname, "status": "ok", "tau": tau, "cutoff": c, **forecast_metrics(y, f, eps),
                         **{f"alert_{k}": v for k, v in alert_metrics(y, f, tau, c).items()}})
            for yr in cfg["evaluation"]["per_year"]:
                msk = pd.DatetimeIndex(sc["origin_date"]).year == yr
                if msk.any():
                    yrows.append({"model": mname, "year": yr, **forecast_metrics(y[msk], f[msk], eps),
                                  **{f"alert_{k}": v for k, v in alert_metrics(y[msk], f[msk], tau, c).items()}})
            srows.append({"model": mname, "subgroup": "trailing_HV20_at_origin_<=_tau", "n": int(hv_low.sum()),
                          **{f"alert_{k}": v for k, v in alert_metrics(y[hv_low], f[hv_low], tau, c).items()}})
        write_csv(paths.test / "test_metrics.csv", pd.DataFrame(rows))
        write_csv(paths.test / "test_metrics_by_year.csv", pd.DataFrame(yrows))
        write_csv(paths.test / "test_alert_subgroup_low_trailing_vol.csv", pd.DataFrame(srows))
        write_csv(paths.test / "test_metrics_per_seed.csv",
                  pd.DataFrame([{"model_seed": k, **forecast_metrics(y, sc[k].to_numpy(), eps)} for k in seeds]))
        # paired comparisons (moving-block bootstrap) + supplementary DM
        b = cfg["evaluation"]["bootstrap"]
        pairs = [("L_RV", "L_R"), ("T_RV", "T_R"), ("T_R", "L_R"), ("T_RV", "L_RV")] + \
                [(d, bb) for d in DL_MODELS for bb in BASELINES]
        prow, drow = [], []
        for a, bb in pairs:
            if not (np.isfinite(sc[a]).all() and np.isfinite(sc[bb]).all()):
                continue
            for kind in ("log_se", "se", "qlike"):
                d = loss_series(y, sc[a], kind, eps) - loss_series(y, sc[bb], kind, eps)
                prow.append({"model_a": a, "model_b": bb, "loss": kind, "primary": (a, bb) in pairs[:4],
                             **paired_block_bootstrap(d, b["block_length"], b["reps"], b["seed"])})
                drow.append({"model_a": a, "model_b": bb, "loss": kind,
                             **diebold_mariano(loss_series(y, sc[a], kind, eps), loss_series(y, sc[bb], kind, eps),
                                               h=cfg["target"]["horizon"])})
        write_csv(paths.test / "test_paired_bootstrap.csv", pd.DataFrame(prow))
        write_csv(paths.test / "supplementary_diebold_mariano.csv", pd.DataFrame(drow))
        # supplementary VaR backtests (Project instructions; not in the report's locked metrics)
        r = panel["r_ref"]
        pos = samples.set_index("origin_date").loc[sc["origin_date"], "pos"].to_numpy()
        r1 = r.to_numpy()[pos + 1]
        r5 = np.array([r.to_numpy()[p + 1:p + 6].sum() for p in pos])
        vrows = []
        for mname in ALL_MODELS:
            if np.isfinite(sc[mname]).all():
                vb = var_backtest(sc[mname].to_numpy(), r1, r5, tuple(cfg["evaluation"]["supplementary"]["var"]["levels"]))
                vb.insert(0, "model", mname)
                vrows.append(vb)
        if vrows:
            write_csv(paths.test / "supplementary_var_backtest.csv", pd.concat(vrows))
        st.extra.update({"n_test_scored": int(len(sc)), "n_test_trade_origins": int(len(tf))})
        log(f"[test] scored={len(sc)} trade_origins={len(tf)}")
    return {"n_scored": int(len(sc))}


# ------------------------------------------------------------------------------------------ backtest
def stage_backtest(cfg, paths: Paths, log=print) -> pd.DataFrame:
    fz = verify_freeze(paths)
    if not (paths.test / "test_forecasts.csv").exists():
        raise RuntimeError("run the test stage first")
    m = RunManifest(paths.run / "run_manifest.json")
    with m.stage("backtest") as st:
        panel, samples = _load(paths)
        tick = cfg["data"]["tickers"]
        tf = pd.read_csv(paths.test / "test_forecasts.csv", parse_dates=["origin_date", "trade_date", "target_end_date"])
        t0 = pd.Timestamp(cfg["splits"]["test"]["start"])
        dates = panel.index[(panel.index >= t0) & (panel.index <= pd.Timestamp(cfg["splits"]["test"]["end"]))]
        rets = panel.loc[dates, [f"ret_{t}" for t in tick]].copy()
        rets.columns = tick
        first_signal = tf["origin_date"].min()
        start = dates[dates > first_signal][0]                   # common first tradable session after first signal
        w_n, w_a = weights_vector(cfg, "reference_weights"), weights_vector(cfg, "alert_weights")
        w_nt = weights_vector(cfg, "no_tilt_weights")
        delay = cfg["backtest"]["signal_delay_days"]
        strategies, siglog = {}, []
        strategies["Reference_fixed"] = (bt.build_targets(dates, None, w_n, w_a, tick, static_w=w_n), "benchmark")
        strategies["NoTilt_ACWI75_SHY20_GLD5"] = (bt.build_targets(dates, None, w_n, w_a, tick, static_w=w_nt), "benchmark")
        for mname in ALL_MODELS:
            c = fz["cutoffs"][mname]["cutoff"]
            if c is None:
                continue
            sig = pd.Series((tf[mname] > c).to_numpy(), index=tf["origin_date"])
            T, state, lg = bt.build_targets(dates, sig, w_n, w_a, tick, delay=delay)
            siglog += [{"strategy": f"{mname}_overlay", **x} for x in lg]
            strategies[f"{mname}_overlay"] = ((T, state, lg), "overlay")
            sw = np.array(fz["static_comparator_weights"][mname])
            strategies[f"{mname}_static"] = (bt.build_targets(dates, None, w_n, w_a, tick, static_w=sw), "static_comparator")
        rf = panel["rf"]
        navs, summ = [], []
        for cost in cfg["backtest"]["cost_bps_sensitivity"]:
            for name, ((T, state, _), kind) in strategies.items():
                res = bt.simulate(rets, T, cost, start, cfg["backtest"]["initial_nav"], cfg["backtest"]["bisection_tol"])
                res["alert_state_for_trade"] = state.reindex(res.index).astype(float)
                s = bt.summarize(res, rf, cfg["backtest"]["initial_nav"], cfg["target"]["annualization"])
                s.update({"strategy": name, "kind": kind, "cost_bps": cost,
                          "alert_frequency": float(state.reindex(res.index[:-1]).mean()) if kind == "overlay" else None,
                          "is_base_cost": cost == cfg["backtest"]["cost_bps_base"]})
                summ.append(s)
                r2 = res.reset_index()
                r2.insert(0, "cost_bps", cost)
                r2.insert(0, "strategy", name)
                navs.append(r2)
        summary = pd.DataFrame(summ)
        write_csv(paths.backtest / "backtest_summary.csv", summary)
        write_csv(paths.backtest / "backtest_daily.csv", pd.concat(navs, ignore_index=True))
        write_csv(paths.backtest / "signal_log.csv", pd.DataFrame(siglog, columns=["strategy", "trade_date", "origin", "issue"]))
        st.extra.update({"start": str(start.date()), "strategies": len(strategies)})
        log(f"[backtest] start={start.date()} strategies={len(strategies)}")
    return summary


# ------------------------------------------------------------------------------------------ figures/report
def stage_figures(cfg, paths: Paths, log=print):
    from .plotting import make_all_figures
    m = RunManifest(paths.run / "run_manifest.json")
    with m.stage("figures") as st:
        man = make_all_figures(cfg, paths, run_id=paths.run.name)
        st.extra["figures"] = int(len(man))
    return man


def stage_report(cfg, paths: Paths, log=print):
    from .reporting import handover_th
    m = RunManifest(paths.run / "run_manifest.json")
    with m.stage("report"):
        p = handover_th(cfg, paths)
    return p
