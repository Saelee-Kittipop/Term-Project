"""Download immutable raw snapshots (ETF prices, VIX, Fama/French RF) with metadata and hashes.

A snapshot directory is written once and never modified afterwards:

    <data_dir>/raw/<snapshot_id>/
        etf_<TICKER>_adjusted.csv      yfinance history(auto_adjust=True, actions=True)
        etf_<TICKER>_unadjusted.csv    yfinance history(auto_adjust=False, actions=True)  (audit only)
        vix_cboe.csv                   original bytes from Cboe
        ff_factors_daily.zip           original bytes from the Kenneth French Data Library
        snapshot_metadata.json         download timestamps, URLs, package versions, sha256 per file
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

from ..io_utils import environment_info, sha256_file, utc_now, write_json, read_json

UA = {"User-Agent": "Mozilla/5.0 (research; volvix)"}


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download_etf(ticker: str, start: str, end: str, auto_adjust: bool, repair: bool) -> pd.DataFrame:
    import time
    import yfinance as yf
    last_err = None
    for attempt in range(4):                       # Yahoo occasionally rate-limits; retry, never fabricate
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=auto_adjust,
                                           actions=True, repair=repair, raise_errors=True)
            if df is not None and not df.empty:
                break
            last_err = RuntimeError("empty frame")
        except Exception as e:                     # noqa: BLE001
            last_err = e
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")
    df = df.copy()
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "date"
    return df


def create_snapshot(cfg: dict, data_dir: str | Path, snapshot_id: str | None = None,
                    manual_dir: str | Path | None = None) -> Path:
    """Download all raw inputs into a new immutable snapshot. Returns the snapshot path.

    manual_dir: optional folder holding files with the same names (used when a source is
    unreachable); such files are copied byte-for-byte and marked source=manual.
    """
    snapshot_id = snapshot_id or utc_now().replace(":", "").replace("-", "")
    snap = Path(data_dir) / "raw" / snapshot_id
    if (snap / "snapshot_metadata.json").exists():
        raise FileExistsError(f"snapshot {snap} already exists and is immutable")
    snap.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    meta = {"snapshot_id": snapshot_id, "created_utc": utc_now(), "environment": environment_info(),
            "request": {"tickers": d["tickers"], "start": d["download_start"],
                        "end_exclusive": d["download_end_exclusive"], "yfinance": d["yfinance"]},
            "files": {}}

    def _record(name: str, source: str, url: str | None, extra: dict | None = None):
        p = snap / name
        meta["files"][name] = {"source": source, "url": url, "downloaded_utc": utc_now(),
                               "bytes": p.stat().st_size, "sha256": sha256_file(p), **(extra or {})}

    def _manual(name: str) -> bool:
        if manual_dir and (Path(manual_dir) / name).exists():
            (snap / name).write_bytes((Path(manual_dir) / name).read_bytes())
            _record(name, "manual", None, {"manual_path": str(Path(manual_dir) / name)})
            return True
        return False

    for t in d["tickers"]:
        for adj in (True, False):
            name = f"etf_{t}_{'adjusted' if adj else 'unadjusted'}.csv"
            if _manual(name):
                continue
            df = _download_etf(t, d["download_start"], d["download_end_exclusive"], adj, d["yfinance"]["repair"])
            df.to_csv(snap / name)
            _record(name, "yfinance", f"yfinance.Ticker('{t}').history(auto_adjust={adj})",
                    {"rows": int(len(df)), "first": str(df.index.min().date()), "last": str(df.index.max().date())})

    for name, url in (("vix_cboe.csv", d["urls"]["cboe_vix"]), ("ff_factors_daily.zip", d["urls"]["french_daily_zip"])):
        if _manual(name):
            continue
        (snap / name).write_bytes(_http_get(url))
        _record(name, "http", url)

    write_json(snap / "snapshot_metadata.json", meta)
    # make raw files read-only (immutability guard; ignored on filesystems that do not support it)
    for p in snap.iterdir():
        try:
            os.chmod(p, 0o444)
        except OSError:
            pass
    return snap


def verify_snapshot(snap: str | Path) -> dict:
    """Re-hash every file and compare with snapshot_metadata.json. Raises on mismatch."""
    snap = Path(snap)
    meta = read_json(snap / "snapshot_metadata.json")
    for name, rec in meta["files"].items():
        h = sha256_file(snap / name)
        if h != rec["sha256"]:
            raise RuntimeError(f"hash mismatch for {name}: {h} != {rec['sha256']}")
    return meta


def latest_snapshot(data_dir: str | Path) -> Path | None:
    root = Path(data_dir) / "raw"
    snaps = sorted(p for p in root.glob("*") if (p / "snapshot_metadata.json").exists()) if root.exists() else []
    return snaps[-1] if snaps else None
