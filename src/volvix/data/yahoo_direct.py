"""Download raw files directly over HTTPS (no yfinance): Yahoo chart API v8 for ETF prices and ^VIX,
Cboe VIX_History.csv and the Kenneth French daily factors zip. Bytes are saved exactly as received.

The Yahoo endpoint is the one yfinance itself calls; adjusted prices are derived from it in
from_files.parse_yahoo_chart with the yfinance auto_adjust=True convention.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ..io_utils import utc_now, write_json

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")
HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
VIX_YAHOO_FILE = "volvix_yahoo_chart_VIX.json"


def yahoo_chart_url(symbol: str, start: str, end_exclusive: str, host: str = HOSTS[0]) -> str:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end_exclusive, tz="UTC").timestamp())
    sym = urllib.request.quote(symbol, safe="")
    return (f"https://{host}/v8/finance/chart/{sym}?period1={p1}&period2={p2}"
            "&interval=1d&events=div%2Csplit&includeAdjustedClose=true")


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_yahoo_chart(symbol: str, start: str, end_exclusive: str, attempts: int = 5, wait: float = 8.0,
                      getter=None, log=print) -> tuple[bytes, str]:
    """Return (raw bytes, url). Retries on HTTP 429/5xx and alternates query1/query2. Never fabricates data."""
    getter = getter or http_get
    last = None
    for k in range(attempts):
        url = yahoo_chart_url(symbol, start, end_exclusive, HOSTS[k % len(HOSTS)])
        try:
            raw = getter(url)
            j = json.loads(raw)
            if j.get("chart", {}).get("error"):
                raise ValueError(j["chart"]["error"])
            if not j["chart"]["result"][0].get("timestamp"):
                raise ValueError("no timestamps in response")
            return raw, url
        except (OSError, ValueError, KeyError, TypeError) as e:     # URLError/HTTPError are OSError
            last = e
            log(f"[yahoo] {symbol} attempt {k + 1}/{attempts} failed: {e}")
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"Yahoo chart download failed for {symbol}: {last}")


def download_direct_files(cfg: dict, staging: str | Path, manual_dir: str | Path | None = None,
                          getter=None, log=print, pause: float = 2.0) -> dict:
    """Write the 5 ETF chart JSONs, Yahoo ^VIX JSON, Cboe VIX CSV and French zip into `staging`."""
    getter = getter or http_get
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    status = {"started_utc": utc_now(), "files": {}}

    def manual(name: str) -> bool:
        if manual_dir and (Path(manual_dir) / name).exists():
            shutil.copyfile(Path(manual_dir) / name, staging / name)
            status["files"][name] = {"source": "manual", "path": str(Path(manual_dir) / name)}
            return True
        return False

    for t in list(d["tickers"]) + ["^VIX"]:
        name = VIX_YAHOO_FILE if t == "^VIX" else f"volvix_yahoo_chart_{t}.json"
        if manual(name):
            continue
        try:
            raw, url = fetch_yahoo_chart(t, d["download_start"], d["download_end_exclusive"], getter=getter, log=log)
        except RuntimeError as e:
            if t == "^VIX":                        # ^VIX is only a cross-check / fallback; ETFs are required
                status["files"][name] = {"source": "yahoo_chart_api_v8", "error": str(e)}
                continue
            raise
        (staging / name).write_bytes(raw)
        status["files"][name] = {"source": "yahoo_chart_api_v8", "url": url, "downloaded_utc": utc_now(), "bytes": len(raw)}
        log(f"[yahoo] {t}: {len(raw):,} bytes")
        time.sleep(pause)
    for name, url in (("VIX_History.csv", d["urls"]["cboe_vix"]),
                      ("F-F_Research_Data_Factors_daily_CSV.zip", d["urls"]["french_daily_zip"])):
        if manual(name):
            continue
        try:
            raw = getter(url)
            (staging / name).write_bytes(raw)
            status["files"][name] = {"source": "http", "url": url, "downloaded_utc": utc_now(), "bytes": len(raw)}
            log(f"[http] {name}: {len(raw):,} bytes")
        except Exception as e:                     # noqa: BLE001
            status["files"][name] = {"source": "http", "url": url, "error": str(e)}
            if name.startswith("F-F"):
                raise RuntimeError(f"French RF download failed: {e}")
            log(f"[http] Cboe VIX failed ({e}); Yahoo ^VIX will be used as a documented fallback")
    write_json(staging / "download_status.json", status)
    return status
