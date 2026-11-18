"""Thai-language data-quality report and handover document, generated only from saved artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import read_json, utc_now, write_csv


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(ไม่มีข้อมูล)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _synthetic_banner(paths) -> str:
    m = read_json(paths.run / "run_manifest.json")
    if m.get("data", {}).get("snapshot_synthetic"):
        return ("> **คำเตือน: รันนี้ใช้ข้อมูลสังเคราะห์ (SYNTHETIC) เพื่อทดสอบโค้ดเท่านั้น "
                "ตัวเลขทั้งหมดไม่ใช่ผลการวิจัยและห้ามนำไปใช้ในรายงาน**\n\n")
    return ""


def descriptive_stats(panel, samples, cfg) -> pd.DataFrame:
    rows = []
    tick = cfg["data"]["tickers"]
    for s in ("train", "validation", "test"):
        part = samples[samples["split"] == s]
        dates = part["origin_date"]
        for name, ser in [("y_target", part["y"])] + [
                (f"ret_{t}", panel.loc[panel.index.isin(dates), f"ret_{t}"]) for t in tick] + [
                ("r_ref", panel.loc[panel.index.isin(dates), "r_ref"]), ("vix_lag1", panel.loc[panel.index.isin(dates), "vix_lag1"])]:
            x = ser.dropna().to_numpy(float)
            rows.append({"split": s, "variable": name, "n": len(x), "missing": int(ser.isna().sum()),
                         "mean": x.mean() if len(x) else np.nan, "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                         "q10": np.quantile(x, .1) if len(x) else np.nan, "q50": np.quantile(x, .5) if len(x) else np.nan,
                         "q90": np.quantile(x, .9) if len(x) else np.nan, "min": x.min() if len(x) else np.nan,
                         "max": x.max() if len(x) else np.nan})
    return pd.DataFrame(rows)


def data_quality_report_th(paths, cfg) -> Path:
    q = read_json(paths.processed / "data_quality.json")
    steps = pd.read_csv(paths.processed / "step_log.csv")
    excl = pd.read_csv(paths.processed / "exclusion_log.csv")
    man = pd.read_csv(paths.dataset / "split_manifest.csv")
    chk = pd.read_csv(paths.dataset / "alignment_checks.csv")
    ex = pd.read_csv(paths.dataset / "alignment_examples.csv")
    panel = pd.read_parquet(paths.processed / "panel.parquet")
    samples = pd.read_parquet(paths.dataset / "samples.parquet")
    desc = descriptive_stats(panel, samples, cfg)
    write_csv(paths.reports / "descriptive_stats_by_split.csv", desc)
    tr_y = samples.loc[samples["split"] == "train", "y"]
    tau = float(np.quantile(tr_y, cfg["alerts"]["tau_quantile"]))
    hr = pd.DataFrame([{"split": s, "n": int((samples["split"] == s).sum()),
                        "high_risk_share_y_gt_tau": float((samples.loc[samples["split"] == s, "y"] > tau).mean())}
                       for s in ("train", "validation", "test")])
    write_csv(paths.reports / "high_risk_share_by_split.csv", hr)
    ex_counts = excl.groupby(["stage", "item"]).size().reset_index(name="count") if len(excl) else pd.DataFrame()
    ca = pd.read_csv(paths.processed / "corporate_actions_audit.csv") if (paths.processed / "corporate_actions_audit.csv").stat().st_size > 1 else pd.DataFrame()
    t = []
    t.append(f"# รายงานคุณภาพข้อมูล (Data Quality Report)\n\nสร้างอัตโนมัติเมื่อ {utc_now()} จาก run `{paths.run.name}`\n\n")
    t.append(_synthetic_banner(paths))
    t.append("## 1. แหล่งข้อมูลและ snapshot\n\n")
    t.append(f"- Snapshot: `{q['snapshot_id']}` (ไฟล์ดิบไม่ถูกแก้ไข ตรวจ SHA-256 ก่อนใช้ทุกครั้ง)\n"
             f"- ที่มาของไฟล์: {q.get('provenance')}\n"
             f"- แหล่ง VIX ที่ใช้จริง: {q.get('vix_source_used')} ; เทียบ Cboe กับ Yahoo ^VIX: {q.get('vix_crosscheck_cboe_vs_yahoo')}\n"
             f"- ปฏิทินซื้อขาย: {q['calendar_source']}\n"
             f"- วันแรกที่มีราคาแต่ละ ETF: {q['first_price_date_by_ticker']}\n"
             f"- วันเริ่มข้อมูลร่วมจริง: **{q['common_start']}** ถึง {q['last_common_date']} ({q['sessions']} วันทำการ)\n"
             f"- วันที่มีผลตอบแทนใช้ได้ครบ 5 ETF: {q['valid_return_days']}\n"
             f"- RF (Fama/French) มีถึง: {q['rf_last_available']} ; วันทำการที่ไม่มี RF: {q['rf_missing_sessions']}\n\n")
    t.append("## 2. จำนวนแถวก่อนและหลังแต่ละขั้นตอน\n\n" + md_table(steps, "{:.0f}") + "\n")
    t.append("## 3. ความครบถ้วนของปฏิทินและรายการที่ตัด\n\n")
    t.append(f"- Missing sessions ราย ETF: {q['missing_sessions_by_ticker']}\n"
             f"- แถวราคาในวันที่ไม่ใช่วันทำการ: {q['non_session_price_rows']}\n"
             f"- วันทำการที่ไม่มี VIX: {q['vix_missing_sessions']}\n\n"
             "ไม่มีการเติมค่า (forward/backward fill) ผลตอบแทนที่คร่อม missing session ถูกทำเป็นค่าว่าง "
             "และหน้าต่าง Input/target ที่มีค่าว่างถูกตัดออกจากตัวอย่าง รายละเอียดรายวันอยู่ใน `processed/exclusion_log.csv`\n\n")
    t.append(md_table(ex_counts, "{:.0f}") + "\n")
    t.append("## 4. Adjusted prices และ corporate actions\n\n")
    t.append(f"- ใช้ Close จาก yfinance `auto_adjust=True` อย่างเดียว ไม่บวก Dividends ซ้ำ\n"
             f"- จำนวน event (dividend/split): {q['corporate_action_events']} ; split events: {q['split_events']}\n"
             f"- ค่าความต่างสูงสุดระหว่าง adjusted return กับ (Close+Div)/Close_prev−1 ในวัน event: {q['corporate_action_max_abs_diff']}\n"
             f"- จำนวน event ที่ดูเหมือนนับเงินปันผลซ้ำ: {q['dividend_double_count_suspects']} (ควรเป็น 0)\n"
             f"- ผลตอบแทนรายวันที่ |r| > {cfg['data']['extreme_return_flag_abs']:.0%}: {q['extreme_return_flags']} รายการ "
             "(flag เท่านั้น ไม่ตัดทิ้ง เพื่อรักษาข้อมูลวิกฤตจริง) ดู `processed/flagged_extreme_returns.csv`\n\n")
    t.append("## 5. การแบ่งข้อมูลและการ purge\n\n" + md_table(man, "{:.0f}") + "\n")
    t.append(f"τ (P90 ของ training targets ที่ label ครบ) = **{tau:.6f}** ({tau*100:.2f}% annualized)\n\n" + md_table(hr) + "\n")
    t.append("## 6. ตรวจ alignment ก่อนฝึก (ผลจากการคำนวณจริงบนข้อมูลชุดนี้)\n\n" + md_table(chk) + "\n")
    t.append("ตัวอย่างวันที่ (Input สิ้นสุดที่ t, target เริ่ม t+1 และสิ้นสุด t+5):\n\n" + md_table(ex, "{:.6f}") + "\n")
    t.append("## 7. สถิติเชิงพรรณนาแยกตาม split\n\n" + md_table(desc, "{:.5f}") + "\n")
    p = paths.reports / "data_quality_report_th.md"
    p.write_text("".join(t), encoding="utf-8")
    return p


def handover_th(cfg, paths) -> Path:
    m = read_json(paths.run / "run_manifest.json")
    stages = pd.DataFrame([{k: s.get(k) for k in ("stage", "status", "started_utc", "runtime_sec", "git_commit")}
                           for s in m["stages"] if not (s["stage"] == "report" and s["status"] == "running")])
    t = [f"# เอกสารส่งมอบผลการทดลอง (Handover สำหรับรายงานและสไลด์)\n\nRun: `{paths.run.name}` | ประเภท: **{m.get('run_kind')}** | "
         f"สร้างเมื่อ {utc_now()}\n\n", _synthetic_banner(paths)]
    t.append("## 1. สถานะแต่ละขั้นตอน\n\n" + md_table(stages, "{:.1f}") + "\n")
    t.append("ความหมายของประเภทผล: **smoke** = ตรวจว่าโค้ดรันได้ (ไม่ใช่ผล), **validation** = ใช้เลือกโมเดล/cutoff "
             "(รายงานได้ในฐานะผลการพัฒนา), **final test** = ผลขั้นสุดท้ายหลังตรึง protocol (ใช้ตอบ RQ1–RQ3)\n\n")
    if (paths.training / "grid_summary.csv").exists():
        g = pd.read_csv(paths.training / "grid_summary.csv")
        t.append(f"## 2. Training grid\n\nเสร็จ {int((g['status']=='completed').sum())} / {len(g)} runs\n\n")
        t.append(md_table(g[["run_id", "status", "best_val_logmse", "best_epoch", "n_params", "train_time_sec", "peak_gpu_mem_mb"]]) + "\n")
    if (paths.frozen / "freeze.json").exists():
        fz = read_json(paths.frozen / "freeze.json")
        t.append(f"## 3. ค่าที่ตรึงก่อนเปิด test (validation)\n\n- τ = {fz['tau']:.6f}\n- Config ที่เลือก: {fz['selection']}\n\n")
        cut = pd.DataFrame([{"model": k, **v} for k, v in fz["cutoffs"].items()])
        t.append(md_table(cut) + "\n")
        vm = pd.read_csv(paths.validation / "validation_metrics.csv")
        t.append("### Validation metrics (ผลช่วงพัฒนา ไม่ใช่ผลสุดท้าย)\n\n" +
                 md_table(vm[["model", "n", "MAE_pp", "RMSE_pp", "log_RMSE", "QLIKE", "alert_precision", "alert_recall", "alert_F1", "alert_AP"]]) + "\n")
    if (paths.test / "test_metrics.csv").exists():
        tm = pd.read_csv(paths.test / "test_metrics.csv")
        t.append("## 4. ผล final test (ตาราง 'ผลการพยากรณ์' และ 'ผลการแจ้งเตือน' ในบทที่ 4)\n\n")
        t.append(md_table(tm[[c for c in ["model", "n", "MAE_pp", "RMSE_pp", "log_RMSE", "QLIKE"] if c in tm]]) + "\n")
        t.append(md_table(tm[[c for c in ["model", "alert_precision", "alert_recall", "alert_F1", "alert_AP",
                                          "alert_false_alerts_per_100", "alert_missed_high_risk", "alert_prevalence"] if c in tm]]) + "\n")
        pb = pd.read_csv(paths.test / "test_paired_bootstrap.csv")
        t.append("### Paired differences (model_a − model_b, ค่าลบ = model_a ดีกว่า; moving-block bootstrap 20 วัน × 1000)\n\n" +
                 md_table(pb[pb["primary"]][["model_a", "model_b", "loss", "mean_diff", "ci_low_2.5", "ci_high_97.5"]], "{:.6f}") + "\n")
    if (paths.backtest / "backtest_summary.csv").exists():
        bs = pd.read_csv(paths.backtest / "backtest_summary.csv")
        base = bs[bs["is_base_cost"]]
        t.append(f"## 5. ผลทางเศรษฐศาสตร์ (ต้นทุน {cfg['backtest']['cost_bps_base']} bps)\n\n" +
                 md_table(base[["strategy", "net_CAGR", "ann_vol", "sharpe", "max_drawdown", "turnover_per_year",
                                "cumulative_cost", "alert_frequency", "dd_recovery"]]) + "\n")
    if (paths.figures / "figure_manifest.csv").exists():
        fm = pd.read_csv(paths.figures / "figure_manifest.csv")
        t.append("## 6. รายการภาพ (แต่ละภาพมี CSV ข้อมูลต้นทาง)\n\n" +
                 md_table(fm[[c for c in ["figure_id", "title_th", "data_csv", "result_kind", "status"] if c in fm]]) + "\n")
    t.append("## 7. ข้อควรระวังในการเขียนรายงาน\n\n"
             "- ห้ามเรียกโมเดลว่าดีที่สุดจากตัวชี้วัดเดียว ใช้ log-RMSE เป็นหลักและอ่านร่วมกับ RMSE/MAE/QLIKE, alerts และผลหลังต้นทุน\n"
             "- target 5 วันซ้อนกัน CI จาก bootstrap เป็นหลักฐานเสริม ไม่ใช่การทดลองแบบสุ่ม\n"
             "- ไฟล์ `supplementary_*` (Diebold–Mariano, VaR/Kupiec/Christoffersen) มาจากคำสั่งใน Project ไม่อยู่ใน protocol หลักของร่างรายงาน ให้รายงานแยกเป็นผลเสริม\n"
             "- หากจะเปลี่ยนสิ่งใดหลังเปิด test ต้องแยกเป็น exploratory analysis ใน run ใหม่\n")
    p = paths.reports / "handover_th.md"
    p.write_text("".join(t), encoding="utf-8")
    return p
