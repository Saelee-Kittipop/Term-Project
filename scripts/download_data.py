#!/usr/bin/env python
"""Download raw data directly from Yahoo (chart API v8), Cboe and Kenneth French into an immutable snapshot,
then print a per-ticker summary.   python scripts/download_data.py --root <DRIVE_ROOT> [--override ...]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from volvix.config import load_config  # noqa: E402
from volvix.data.download import create_snapshot, verify_snapshot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--snapshot-id", default=None)
    ap.add_argument("--manual-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(overrides=a.override)
    snap = create_snapshot(cfg, Path(a.root) / "data", a.snapshot_id, a.manual_dir)
    meta = verify_snapshot(snap)
    rows = []
    for t in cfg["data"]["tickers"]:
        df = pd.read_csv(snap / f"etf_{t}_adjusted.csv", parse_dates=["date"])
        rows.append({"ticker": t, "rows": len(df), "first": df["date"].min().date(), "last": df["date"].max().date(),
                     "null_close": int(df["Close"].isna().sum()), "dividends": int((df["Dividends"] != 0).sum()),
                     "splits": int((df["Stock Splits"] != 0).sum()), "last_adj_close": round(float(df["Close"].iloc[-1]), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("VIX source:", meta.get("vix_source_used", "cboe"), "| cross-check:", meta.get("vix_crosscheck_cboe_vs_yahoo"))
    print("SNAPSHOT_ID=" + snap.name)


if __name__ == "__main__":
    main()
