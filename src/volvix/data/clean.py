"""Clean a raw snapshot into an aligned daily panel, logging every row change and exclusion.

Rules (report section 3 'แหล่งข้อมูลและการตรวจคุณภาพ'):
* use adjusted Close (auto_adjust=True) consistently; never add the Dividends column again;
* keep the exchange calendar; a return is valid only when the price exists on the session AND on the
  immediately preceding session (no multi-day return labelled as a daily return);
* never forward/backward fill; missing data invalidates the affected rows and they are logged;
* extreme returns are flagged, never removed;
* VIX enters as level lagged one trading session (row s uses VIX of session s-1).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import read_json, write_csv, write_json


class StepLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, step: str, item: str, before: int, after: int, note_th: str):
        self.rows.append({"step": step, "item": item, "rows_before": int(before), "rows_after": int(after),
                          "removed": int(before - after), "note_th": note_th})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class ExclusionLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, date, item: str, stage: str, reason: str):
        self.rows.append({"date": pd.Timestamp(date).date().isoformat() if date is not None else None,
                          "item": item, "stage": stage, "reason": reason})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["date", "item", "stage", "reason"])


def exchange_sessions(start, end, calendar: str = "XNYS") -> pd.DatetimeIndex | None:
    try:
        import pandas_market_calendars as mcal
    except Exception:
        return None
    cal = mcal.get_calendar("NYSE" if calendar == "XNYS" else calendar)
    sched = cal.schedule(start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))
    return pd.DatetimeIndex(sched.index).normalize().tz_localize(None)


def _load_etf(path: Path, ticker: str, steps: StepLog, excl: ExclusionLog, start, end) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    n0 = len(df)
    steps.add("load", ticker, n0, n0, "อ่านไฟล์ adjusted จาก raw snapshot")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if not df["date"].is_monotonic_increasing:
        excl.add(None, ticker, "order", "ไฟล์ไม่เรียงวันที่ จึงเรียงใหม่ (ไม่ตัดแถว)")
        df = df.sort_values("date", kind="mergesort")
    dup = df["date"].duplicated(keep=False)
    if dup.any():
        before = len(df)
        keep = []
        for d, g in df[dup].groupby("date"):
            if (g.drop(columns=["date"]).nunique(dropna=False) <= 1).all():
                keep.append(g.index[0])
                excl.add(d, ticker, "duplicates", "แถวซ้ำที่ค่าเหมือนกัน เก็บไว้หนึ่งแถว")
            else:
                excl.add(d, ticker, "duplicates", "แถวซ้ำที่ค่าต่างกัน ตัดวันนั้นของ ticker นี้")
        df = pd.concat([df[~dup], df.loc[keep]]).sort_values("date", kind="mergesort")
        steps.add("duplicates", ticker, before, len(df), "จัดการวันที่ซ้ำ")
    before = len(df)
    in_range = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    df = df[in_range]
    steps.add("date_range", ticker, before, len(df), f"จำกัดช่วง {start} ถึง {end}")
    before = len(df)
    bad = ~np.isfinite(df["Close"].astype(float)) | (df["Close"].astype(float) <= 0)
    for d in df.loc[bad, "date"]:
        excl.add(d, ticker, "price_validity", "ราคา Close ว่างหรือไม่เป็นบวก")
    df = df[~bad]
    steps.add("price_validity", ticker, before, len(df), "ตัดราคาว่างหรือไม่เป็นบวก")
    return df.set_index("date")


def corporate_action_audit(adj: pd.DataFrame, unadj: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compare adjusted returns with unadjusted total returns on event days (dividends/splits)."""
    j = adj[["Close", "Dividends", "Stock Splits"]].join(unadj[["Close"]].rename(columns={"Close": "Close_raw"}), how="inner")
    j["r_adj"] = j["Close"].pct_change(fill_method=None)
    # Yahoo's unadjusted 'Close' and 'Dividends' are already split-adjusted, so no split factor here.
    j["r_raw_total"] = (j["Close_raw"] + j["Dividends"]) / j["Close_raw"].shift(1) - 1
    j["r_raw_total_double_div"] = (j["Close_raw"] + 2 * j["Dividends"]) / j["Close_raw"].shift(1) - 1
    ev = j[(j["Dividends"] != 0) | (j["Stock Splits"] != 0)].copy()
    ev["ticker"] = ticker
    ev["abs_diff_total"] = (ev["r_adj"] - ev["r_raw_total"]).abs()
    ev["abs_diff_double"] = (ev["r_adj"] - ev["r_raw_total_double_div"]).abs()
    ev["dividend_double_counted_suspect"] = ev["abs_diff_double"] < ev["abs_diff_total"]
    return ev.reset_index()


def parse_cboe_vix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce").dt.normalize()
    return df[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})


def parse_french_daily(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append((parts[0], parts[4]))
    df = pd.DataFrame(rows, columns=["date", "rf_pct"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["rf"] = df["rf_pct"].astype(float) / 100.0      # percent -> decimal
    return df[["date", "rf"]]


def clean_snapshot(cfg: dict, snap: str | Path, out_dir: str | Path) -> dict:
    snap, out_dir = Path(snap), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(snap / "snapshot_metadata.json")
    tickers = cfg["data"]["tickers"]
    start = cfg["data"]["download_start"]
    end = (pd.Timestamp(cfg["data"]["download_end_exclusive"]) - pd.Timedelta(days=1)).date().isoformat()
    steps, excl = StepLog(), ExclusionLog()

    # 1. ETF prices -----------------------------------------------------------------------
    etf, ca = {}, []
    for t in tickers:
        etf[t] = _load_etf(snap / f"etf_{t}_adjusted.csv", t, steps, excl, start, end)
        up = snap / f"etf_{t}_unadjusted.csv"
        if up.exists():
            un = pd.read_csv(up, parse_dates=["date"]).drop_duplicates("date").set_index("date")
            ca.append(corporate_action_audit(etf[t], un, t))
    first_dates = {t: etf[t].index.min() for t in tickers}
    common_start = max(first_dates.values())
    last_common = min(etf[t].index.max() for t in tickers)

    # 2. Trading calendar ------------------------------------------------------------------
    sessions = exchange_sessions(common_start, last_common, cfg["data"].get("trading_calendar", "XNYS"))
    calendar_source = "pandas_market_calendars:NYSE"
    union = sorted(set().union(*[set(etf[t].index[etf[t].index >= common_start]) for t in tickers]))
    if sessions is None:
        sessions = pd.DatetimeIndex(union)
        calendar_source = "union_of_etf_dates (pandas_market_calendars unavailable)"
    non_session = sorted(set(union) - set(sessions))
    for d in non_session:
        excl.add(d, "ALL", "calendar", "มีราคาในวันที่ไม่ใช่วันทำการ NYSE จึงไม่ใช้")
    panel = pd.DataFrame(index=pd.DatetimeIndex(sessions, name="date"))
    steps.add("calendar", "panel", len(union), len(panel),
              f"ใช้ปฏิทิน {calendar_source} ตั้งแต่วันข้อมูลร่วมแรก {common_start.date()} ถึง {last_common.date()}")

    for t in tickers:
        p = etf[t]["Close"].astype(float).reindex(panel.index)       # NO fill
        panel[f"price_{t}"] = p
        for d in panel.index[p.isna()]:
            excl.add(d, t, "missing_session", "ไม่มีราคาในวันทำการ NYSE (ไม่เติมค่า)")
        prev_ok = p.shift(1).notna()
        r = p / p.shift(1) - 1.0
        r[~(p.notna() & prev_ok)] = np.nan                         # no return across a missing session
        panel[f"ret_{t}"] = r
    ret_cols = [f"ret_{t}" for t in tickers]
    panel["ret_valid"] = panel[ret_cols].notna().all(axis=1)
    panel.loc[panel.index[0], "ret_valid"] = False                 # first common day has no return
    n_valid = int(panel["ret_valid"].sum())
    steps.add("returns", "panel", len(panel), n_valid,
              "คำนวณ simple return เฉพาะวันที่มีราคาทั้งวันนั้นและวันทำการก่อนหน้า ครบทั้ง 5 ETF")
    for d in panel.index[~panel["ret_valid"]][1:]:
        excl.add(d, "ALL", "return_validity", "ผลตอบแทนอย่างน้อยหนึ่ง ETF ใช้ไม่ได้ (missing session)")

    w = np.array([cfg["portfolio"]["reference_weights"][t] for t in tickers])
    panel["r_ref"] = np.where(panel["ret_valid"], panel[ret_cols].fillna(0.0).to_numpy() @ w, np.nan)

    thr = cfg["data"].get("extreme_return_flag_abs", 0.15)
    flags = []
    for t in tickers:
        big = panel[f"ret_{t}"].abs() > thr
        for d, v in panel.loc[big, f"ret_{t}"].items():
            flags.append({"date": d.date().isoformat(), "ticker": t, "return": v,
                          "action": "flag_only_not_removed"})

    # 3. VIX ------------------------------------------------------------------------------
    vix = parse_cboe_vix(snap / "vix_cboe.csv")
    n0 = len(vix)
    vix = vix.dropna(subset=["date"])
    bad = ~np.isfinite(vix["vix"].astype(float)) | (vix["vix"].astype(float) <= 0)
    for d in vix.loc[bad, "date"]:
        excl.add(d, "VIX", "price_validity", "ค่า VIX ว่างหรือไม่เป็นบวก")
    vix = vix[~bad]
    d1 = vix["date"].duplicated(keep=False)
    for d in vix.loc[d1, "date"].unique():
        excl.add(d, "VIX", "duplicates", "วันที่ VIX ซ้ำ ตัดทิ้งทั้งวัน")
    vix = vix[~d1].set_index("date").sort_index()
    steps.add("vix_load", "VIX", n0, len(vix), "อ่าน Cboe VIX CLOSE ตัดค่าที่ใช้ไม่ได้และวันที่ซ้ำ")
    panel["vix"] = vix["vix"].astype(float).reindex(panel.index)   # NO fill
    for d in panel.index[panel["vix"].isna()]:
        excl.add(d, "VIX", "missing_session", "ไม่มีค่า VIX ในวันทำการ ETF (ไม่เติมค่า)")
    panel["vix_lag1"] = panel["vix"].shift(1)                      # VIX of the previous trading session
    steps.add("vix_align", "VIX", len(panel), int(panel["vix_lag1"].notna().sum()),
              "จัด VIX ให้ตรงปฏิทินและ lag 1 วันทำการ")

    # 4. Risk-free (evaluation only) -------------------------------------------------------
    rf = parse_french_daily(snap / "ff_factors_daily.zip").drop_duplicates("date").set_index("date")
    panel["rf"] = rf["rf"].reindex(panel.index)                    # NO fill
    n_rf = int(panel["rf"].notna().sum())
    steps.add("rf_align", "RF", len(panel), n_rf, "จัด RF รายวัน (percent/100) ให้ตรงวัน ใช้สำหรับ Sharpe เท่านั้น")
    rf_last = rf.index.max()

    # 5. Save -------------------------------------------------------------------------------
    panel.to_parquet(out_dir / "panel.parquet")
    write_csv(out_dir / "step_log.csv", steps.frame())
    write_csv(out_dir / "exclusion_log.csv", excl.frame())
    write_csv(out_dir / "flagged_extreme_returns.csv", pd.DataFrame(flags, columns=["date", "ticker", "return", "action"]))
    ca_df = pd.concat(ca, ignore_index=True) if ca else pd.DataFrame()
    write_csv(out_dir / "corporate_actions_audit.csv", ca_df)
    summary = {
        "snapshot_id": meta["snapshot_id"], "calendar_source": calendar_source,
        "provenance": meta.get("provenance", "yfinance"), "vix_source_used": meta.get("vix_source_used", "cboe"),
        "vix_crosscheck_cboe_vs_yahoo": meta.get("vix_crosscheck_cboe_vs_yahoo"),
        "first_price_date_by_ticker": {t: str(first_dates[t].date()) for t in tickers},
        "common_start": str(common_start.date()), "last_common_date": str(last_common.date()),
        "sessions": int(len(panel)), "valid_return_days": n_valid,
        "missing_sessions_by_ticker": {t: int(panel[f"price_{t}"].isna().sum()) for t in tickers},
        "non_session_price_rows": len(non_session),
        "vix_missing_sessions": int(panel["vix"].isna().sum()),
        "rf_missing_sessions": int(panel["rf"].isna().sum()), "rf_last_available": str(rf_last.date()),
        "extreme_return_flags": len(flags),
        "corporate_action_events": int(len(ca_df)),
        "corporate_action_max_abs_diff": float(ca_df["abs_diff_total"].max()) if len(ca_df) else None,
        "dividend_double_count_suspects": int(ca_df["dividend_double_counted_suspect"].sum()) if len(ca_df) else 0,
        "split_events": int((ca_df["Stock Splits"] != 0).sum()) if len(ca_df) else 0,
    }
    write_json(out_dir / "data_quality.json", summary)
    return summary
