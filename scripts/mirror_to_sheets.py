#!/usr/bin/env python
"""Mirror canonical CSV summaries of one run into a Google Sheet (one tab per CSV).

CSV/JSON files on Drive remain the canonical results; the Sheet is a convenience copy.
Run inside Colab after `from google.colab import auth; auth.authenticate_user()`.

    python scripts/mirror_to_sheets.py --run-dir /content/drive/MyDrive/vix_term_project/runs/formal_v1 \
        [--spreadsheet-id <ID>]     # omit to create a new spreadsheet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABS = [
    ("dataset/split_manifest.csv", "split_manifest"),
    ("dataset/alignment_checks.csv", "alignment_checks"),
    ("processed/step_log.csv", "data_step_log"),
    ("training/grid_summary.csv", "grid_summary"),
    ("validation/validation_metrics.csv", "validation_metrics"),
    ("test/test_metrics.csv", "test_metrics"),
    ("test/test_metrics_by_year.csv", "test_metrics_by_year"),
    ("test/test_paired_bootstrap.csv", "test_paired_bootstrap"),
    ("test/test_metrics_per_seed.csv", "test_metrics_per_seed"),
    ("backtest/backtest_summary.csv", "backtest_summary"),
    ("figures/figure_manifest.csv", "figure_manifest"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--spreadsheet-id", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    import gspread
    from google.auth import default
    creds, _ = default(scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    run = Path(a.run_dir)
    sh = gc.open_by_key(a.spreadsheet_id) if a.spreadsheet_id else gc.create(a.title or f"VIX term project - {run.name}")
    written = []
    for rel, tab in TABS:
        p = run / rel
        if not p.exists():
            continue
        df = pd.read_csv(p).astype(object).where(lambda d: d.notna(), "")
        try:
            ws = sh.worksheet(tab)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(tab, rows=max(len(df) + 5, 20), cols=max(len(df.columns) + 2, 5))
        ws.update([list(map(str, df.columns))] + df.values.tolist(), value_input_option="RAW")
        written.append(tab)
    readme = [["source_run_dir", str(run)], ["note", "Canonical results are the CSV/JSON files in the run directory; this sheet is a mirror."],
              ["tabs", ", ".join(written)]]
    try:
        ws = sh.worksheet("README")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("README", rows=10, cols=3)
    ws.update(readme)
    print("spreadsheet:", sh.url)


if __name__ == "__main__":
    main()
