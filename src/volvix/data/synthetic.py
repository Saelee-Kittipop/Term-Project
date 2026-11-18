"""SYNTHETIC raw snapshot for tests and code-path smoke runs. NEVER a research result.

Writes files in exactly the raw-snapshot format (so cleaning is exercised end to end) and
injects known defects: one duplicated row, one missing session for one ETF, one dividend.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import sha256_file, utc_now, write_json
from .clean import exchange_sessions


def make_synthetic_snapshot(cfg: dict, data_dir: str | Path, snapshot_id: str = "SYNTHETIC", seed: int = 7,
                            start: str = "2008-03-26", end: str = "2025-12-31") -> Path:
    rng = np.random.default_rng(seed)
    snap = Path(data_dir) / "raw" / snapshot_id
    snap.mkdir(parents=True, exist_ok=True)
    sess = exchange_sessions(start, end)
    if sess is None:
        sess = pd.bdate_range(start, end)
    n = len(sess)
    # GARCH-like common volatility factor with a few stress regimes
    h = np.empty(n)
    h[0] = 1e-4
    eps = rng.standard_normal(n)
    shock = np.zeros(n)
    for s in rng.choice(np.arange(200, n - 50), 8, replace=False):
        shock[s:s + 30] += 4e-4
    for t in range(1, n):
        h[t] = 2e-6 + 0.08 * (np.sqrt(h[t - 1]) * eps[t - 1]) ** 2 + 0.9 * h[t - 1] + shock[t]
    common = np.sqrt(h) * eps
    vix = np.clip(100 * np.sqrt(252 * h) * (1 + 0.1 * rng.standard_normal(n)) + 3, 9, 90)
    betas = {"ACWI": 1.0, "SHY": 0.02, "XLK": 1.2, "XLV": 0.8, "GLD": 0.1}
    idio = {"ACWI": 0.003, "SHY": 0.0007, "XLK": 0.006, "XLV": 0.005, "GLD": 0.009}
    meta = {"snapshot_id": snapshot_id, "created_utc": utc_now(), "SYNTHETIC": True,
            "warning": "synthetic data for tests/smoke only - not research data", "files": {}}
    for k, t in enumerate(cfg["data"]["tickers"]):
        r = betas[t] * common + idio[t] * rng.standard_normal(n) + 0.0002
        px = 50 * np.cumprod(1 + r)
        df = pd.DataFrame({"date": sess, "Open": px, "High": px, "Low": px, "Close": px, "Volume": 1e6,
                           "Dividends": 0.0, "Stock Splits": 0.0})
        un = df.copy()
        if t == "SHY":                       # one dividend, Yahoo-style back-adjustment before the ex-date
            ex, D = 1000, 0.2
            adj = px.copy()
            adj[:ex] *= 1 - D / px[ex - 1]
            for c in ("Open", "High", "Low", "Close"):
                df[c] = adj
            df.loc[ex, "Dividends"] = D
            un.loc[ex, "Dividends"] = D
            un["Adj Close"] = adj
        if t == "XLV":                       # missing session (row removed) -> two invalid returns
            df = df.drop(index=1500)
            un = un.drop(index=1500)
        if t == "GLD":                       # duplicated identical row
            df = pd.concat([df.iloc[:10], df.iloc[[9]], df.iloc[10:]])
        df.to_csv(snap / f"etf_{t}_adjusted.csv", index=False)
        un.to_csv(snap / f"etf_{t}_unadjusted.csv", index=False)
    v = pd.DataFrame({"DATE": sess.strftime("%m/%d/%Y"), "OPEN": vix, "HIGH": vix, "LOW": vix, "CLOSE": vix})
    v.to_csv(snap / "vix_cboe.csv", index=False)
    rf = pd.DataFrame({"d": sess.strftime("%Y%m%d"), "m": 0.0, "s": 0.0, "h": 0.0, "rf": 0.008})
    buf = io.StringIO()
    buf.write("Synthetic F-F daily file\n\n,Mkt-RF,SMB,HML,RF\n")
    for row in rf.itertuples(index=False):
        buf.write(f"{row.d},{row.m:.2f},{row.s:.2f},{row.h:.2f},{row.rf:.3f}\n")
    with zipfile.ZipFile(snap / "ff_factors_daily.zip", "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.csv", buf.getvalue())
    for p in sorted(snap.iterdir()):
        if p.name != "snapshot_metadata.json":
            meta["files"][p.name] = {"source": "synthetic", "sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(snap / "snapshot_metadata.json", meta)
    return snap
