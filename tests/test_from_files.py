"""Snapshot built from browser-downloaded files (Yahoo chart JSON, Cboe CSV, French zip)."""
import json

import numpy as np
import pandas as pd

from volvix.data.from_files import create_snapshot_from_files, parse_yahoo_chart
from volvix.data.download import verify_snapshot


def _chart(symbol, n=5, split_at=None):
    ts = [int(pd.Timestamp(f"2024-01-{d:02d} 14:30", tz="UTC").timestamp()) for d in range(2, 2 + n)]
    close = [100.0 + i for i in range(n)]
    adj = [c * 0.9 for c in close]
    ev = {"dividends": {str(ts[2]): {"amount": 0.5, "date": ts[2]}}}
    if split_at is not None:
        ev["splits"] = {str(ts[split_at]): {"date": ts[split_at], "numerator": 2, "denominator": 1, "splitRatio": "2:1"}}
    return {"chart": {"result": [{"meta": {"symbol": symbol, "currency": "USD", "exchangeTimezoneName": "America/New_York",
                                           "gmtoffset": -18000, "firstTradeDate": ts[0]},
                                  "timestamp": ts, "events": ev,
                                  "indicators": {"quote": [{"open": close, "high": close, "low": close, "close": close,
                                                            "volume": [1000] * n}],
                                                 "adjclose": [{"adjclose": adj}]}}], "error": None}}


def test_parse_yahoo_chart_matches_yfinance_auto_adjust_convention():
    adj, un, meta = parse_yahoo_chart(json.dumps(_chart("XLK", split_at=3)))
    assert list(adj.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
    assert np.allclose(adj["Close"], un["Adj Close"]) and np.allclose(adj["Open"] / un["Open"], 0.9)
    assert adj["Dividends"].iloc[2] == 0.5 and adj["Stock Splits"].iloc[3] == 2.0
    assert meta["dividend_events"] == 1 and meta["split_events"] == 1


def test_snapshot_from_files_keeps_original_bytes_and_hashes(tmp_path, cfg):
    src = tmp_path / "dl"
    src.mkdir()
    for t in cfg["data"]["tickers"]:
        (src / f"volvix_yahoo_chart_{t}.json").write_text(json.dumps(_chart(t)))
    (src / "VIX_History.csv").write_text("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,13,13,13,13.2\n")
    (src / "F-F_Research_Data_Factors_daily_CSV.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    snap = create_snapshot_from_files(cfg, tmp_path / "data", src, snapshot_id="T1")
    meta = verify_snapshot(snap)
    assert (snap / "original" / "volvix_yahoo_chart_ACWI.json").read_bytes() == (src / "volvix_yahoo_chart_ACWI.json").read_bytes()
    assert meta["files"]["etf_ACWI_adjusted.csv"]["derived_from"] == "original/volvix_yahoo_chart_ACWI.json"
    df = pd.read_csv(snap / "etf_SHY_adjusted.csv")
    assert list(df.columns[:5]) == ["date", "Open", "High", "Low", "Close"]
