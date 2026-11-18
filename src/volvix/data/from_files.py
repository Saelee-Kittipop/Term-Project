"""Build an immutable raw snapshot from files downloaded outside Python (e.g. through a browser).

Used when the execution environment cannot reach Yahoo/Cboe/French directly. Inputs (exact names):
    volvix_yahoo_chart_<TICKER>.json          raw response of Yahoo chart API v8
        https://query1.finance.yahoo.com/v8/finance/chart/<T>?period1=..&period2=..&interval=1d
        &events=div,split&includeAdjustedClose=true   (the endpoint yfinance itself uses)
    VIX_History.csv                           Cboe
    F-F_Research_Data_Factors_daily_CSV.zip   Kenneth French Data Library

The original bytes are copied unchanged to <snapshot>/original/. The CSVs that the cleaner reads are
derived deterministically by this module, with the same convention as yfinance auto_adjust=True:
adjusted Close = adjclose and Open/High/Low are scaled by adjclose/close.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import environment_info, sha256_file, utc_now, write_json

YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}"
             "&interval=1d&events=div%2Csplit&includeAdjustedClose=true")


def _ny_dates(ts) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()


def parse_yahoo_chart(raw: bytes | str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (adjusted, unadjusted, meta) frames in the yfinance history() column layout."""
    j = json.loads(raw)
    if j["chart"].get("error"):
        raise ValueError(f"Yahoo error: {j['chart']['error']}")
    r = j["chart"]["result"][0]
    dates = _ny_dates(r["timestamp"])
    q = r["indicators"]["quote"][0]
    close = np.array(q["close"], dtype=float)
    adj = np.array(r["indicators"]["adjclose"][0]["adjclose"], dtype=float) if r["indicators"].get("adjclose") else close.copy()
    o, h, lo = (np.array(q[k], dtype=float) for k in ("open", "high", "low"))
    vol = np.array(q["volume"], dtype=float)
    div = pd.Series(0.0, index=dates)
    spl = pd.Series(0.0, index=dates)
    ev = r.get("events") or {}
    for e in (ev.get("dividends") or {}).values():
        d = _ny_dates([e["date"]])[0]
        if d in div.index:
            div.loc[d] += float(e["amount"])
    for e in (ev.get("splits") or {}).values():
        d = _ny_dates([e["date"]])[0]
        if d in spl.index:
            spl.loc[d] = float(e["numerator"]) / float(e["denominator"])
    ratio = adj / close
    adjusted = pd.DataFrame({"Open": o * ratio, "High": h * ratio, "Low": lo * ratio, "Close": adj, "Volume": vol,
                             "Dividends": div.to_numpy(), "Stock Splits": spl.to_numpy()}, index=dates)
    unadjusted = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": close, "Adj Close": adj, "Volume": vol,
                               "Dividends": div.to_numpy(), "Stock Splits": spl.to_numpy()}, index=dates)
    adjusted.index.name = unadjusted.index.name = "date"
    meta = {k: r["meta"].get(k) for k in ("symbol", "currency", "exchangeName", "instrumentType", "firstTradeDate",
                                          "exchangeTimezoneName", "gmtoffset", "dataGranularity")}
    meta["rows"] = int(len(dates))
    meta["null_close_rows"] = int(np.isnan(close).sum())
    meta["dividend_events"] = int((div != 0).sum())
    meta["split_events"] = int((spl != 0).sum())
    return adjusted, unadjusted, meta


def create_snapshot_from_files(cfg: dict, data_dir: str | Path, files_dir: str | Path, snapshot_id: str | None = None,
                               provenance: str = "downloaded in the user's desktop browser (Claude built-in browser)") -> Path:
    files_dir = Path(files_dir)
    snapshot_id = snapshot_id or "FILES_" + utc_now().replace(":", "").replace("-", "")
    snap = Path(data_dir) / "raw" / snapshot_id
    if (snap / "snapshot_metadata.json").exists():
        raise FileExistsError(f"snapshot {snap} already exists and is immutable")
    (snap / "original").mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    p1 = int(pd.Timestamp(d["download_start"], tz="UTC").timestamp())
    p2 = int(pd.Timestamp(d["download_end_exclusive"], tz="UTC").timestamp())
    meta = {"snapshot_id": snapshot_id, "created_utc": utc_now(), "environment": environment_info(),
            "provenance": provenance, "request": {"tickers": d["tickers"], "start": d["download_start"],
                                                  "end_exclusive": d["download_end_exclusive"]},
            "files": {}, "derived_by": "src/volvix/data/from_files.py (yfinance auto_adjust convention)"}

    def rec(rel: str, **extra):
        p = snap / rel
        meta["files"][rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, **extra}

    for t in d["tickers"]:
        name = f"volvix_yahoo_chart_{t}.json"
        src = files_dir / name
        shutil.copyfile(src, snap / "original" / name)
        rec(f"original/{name}", source="yahoo_chart_api_v8", url=YAHOO_URL.format(t=t, p1=p1, p2=p2),
            file_mtime_utc=pd.Timestamp(os.path.getmtime(src), unit="s", tz="UTC").isoformat())
        adj, un, m = parse_yahoo_chart((snap / "original" / name).read_bytes())
        if m["symbol"] != t:
            raise ValueError(f"{name} holds {m['symbol']}, expected {t}")
        adj = adj[(adj.index >= pd.Timestamp(d["download_start"])) & (adj.index < pd.Timestamp(d["download_end_exclusive"]))]
        un = un.loc[adj.index]
        adj.to_csv(snap / f"etf_{t}_adjusted.csv")
        un.to_csv(snap / f"etf_{t}_unadjusted.csv")
        rec(f"etf_{t}_adjusted.csv", source="derived", derived_from=f"original/{name}", yahoo_meta=m)
        rec(f"etf_{t}_unadjusted.csv", source="derived", derived_from=f"original/{name}")
    for name, url in (("VIX_History.csv", d["urls"]["cboe_vix"]), ("F-F_Research_Data_Factors_daily_CSV.zip", d["urls"]["french_daily_zip"]),
                      ("volvix_yahoo_chart_VIX.json", None)):
        src = files_dir / name
        if not src.exists() and name == "F-F_Research_Data_Factors_daily_CSV.zip":
            csv = files_dir / "F-F_Research_Data_Factors_daily.csv"
            if not csv.exists():
                raise FileNotFoundError(src)
            # macOS/Safari can auto-extract the downloaded zip; keep the extracted CSV as the original and
            # re-wrap it (byte-identical member) so the cleaner reads the same format
            shutil.copyfile(csv, snap / "original" / csv.name)
            rec(f"original/{csv.name}", source="http_via_browser_auto_extracted_by_os", url=url,
                note="server sent a zip; the OS extracted it on download, so the CSV member is the original kept")
            with zipfile.ZipFile(snap / "ff_factors_daily.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(snap / "original" / csv.name, csv.name)
            rec("ff_factors_daily.zip", source="rezipped", derived_from=f"original/{csv.name}")
            continue
        if not src.exists():
            continue
        shutil.copyfile(src, snap / "original" / name)
        rec(f"original/{name}", source="yahoo_chart_api_v8" if name.endswith(".json") else "http", url=url,
            file_mtime_utc=pd.Timestamp(os.path.getmtime(src), unit="s", tz="UTC").isoformat())
    yv = None
    if (snap / "original" / "volvix_yahoo_chart_VIX.json").exists():
        yv, _, _ = parse_yahoo_chart((snap / "original" / "volvix_yahoo_chart_VIX.json").read_bytes())
    if (snap / "original" / "VIX_History.csv").exists():
        shutil.copyfile(snap / "original" / "VIX_History.csv", snap / "vix_cboe.csv")
        rec("vix_cboe.csv", source="copy", derived_from="original/VIX_History.csv")
        meta["vix_source_used"] = "cboe"
        if yv is not None:                         # cross-check Cboe close against Yahoo ^VIX close
            cb = pd.read_csv(snap / "vix_cboe.csv")
            cb.columns = [c.strip().upper() for c in cb.columns]
            cb = pd.Series(cb["CLOSE"].to_numpy(float), index=pd.to_datetime(cb["DATE"], format="%m/%d/%Y"))
            cb = cb[~cb.index.duplicated()]
            j = pd.concat([cb.rename("cboe"), yv["Close"].rename("yahoo")], axis=1, join="inner").dropna()
            diff = (j["cboe"] - j["yahoo"]).abs()
            meta["vix_crosscheck_cboe_vs_yahoo"] = {"common_days": int(len(j)), "max_abs_diff": float(diff.max()) if len(j) else None,
                                                    "days_abs_diff_gt_0.05": int((diff > 0.05).sum())}
    elif yv is not None:                           # documented fallback when Cboe is unreachable
        fb = pd.DataFrame({"DATE": yv.index.strftime("%m/%d/%Y"), "OPEN": yv["Open"], "HIGH": yv["High"],
                           "LOW": yv["Low"], "CLOSE": yv["Close"]})
        fb.to_csv(snap / "vix_cboe.csv", index=False)
        rec("vix_cboe.csv", source="derived_fallback", derived_from="original/volvix_yahoo_chart_VIX.json")
        meta["vix_source_used"] = "yahoo_^VIX_fallback (Cboe unavailable)"
    else:
        raise FileNotFoundError("no VIX source: neither VIX_History.csv nor volvix_yahoo_chart_VIX.json")
    if (snap / "original" / "F-F_Research_Data_Factors_daily_CSV.zip").exists():
        shutil.copyfile(snap / "original" / "F-F_Research_Data_Factors_daily_CSV.zip", snap / "ff_factors_daily.zip")
        rec("ff_factors_daily.zip", source="copy", derived_from="original/F-F_Research_Data_Factors_daily_CSV.zip")
    write_json(snap / "snapshot_metadata.json", meta)
    for p in snap.rglob("*"):
        if p.is_file():
            try:
                os.chmod(p, 0o444)
            except OSError:
                pass
    return snap
