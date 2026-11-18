#!/usr/bin/env python
"""Command-line entry point. Every stage is resumable; re-running a completed stage is cheap or skipped.

Examples
--------
# smoke (train/validation only, never test)
python scripts/run_pipeline.py --root /content/drive/MyDrive/vix_term_project --run-name smoke \
    --override configs/smoke.yaml --stages data,baselines,train,freeze,figures,report

# formal experiment, in pieces (safe to interrupt)
python scripts/run_pipeline.py --root ... --run-name formal_v1 --stages data,baselines
python scripts/run_pipeline.py --root ... --run-name formal_v1 --stages train --max-runs 12
python scripts/run_pipeline.py --root ... --run-name formal_v1 --stages freeze
python scripts/run_pipeline.py --root ... --run-name formal_v1 --stages test,backtest,figures,report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volvix import pipeline as P  # noqa: E402
from volvix.config import load_config  # noqa: E402

ORDER = ["data", "baselines", "train", "freeze", "test", "backtest", "figures", "report"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root (use a Google Drive folder in Colab)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--stages", default="data,baselines,train,freeze")
    ap.add_argument("--snapshot", default="latest", help="latest | new | synthetic | <snapshot_id>")
    ap.add_argument("--manual-dir", default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--allow-test-rerun-reason", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    paths = P.Paths(a.root, a.run_name)
    stages = ORDER if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    for s in stages:
        if s not in ORDER:
            raise SystemExit(f"unknown stage {s}")
    if "data" not in stages:
        P.init_run(cfg, paths)            # records environment, checks the config is the frozen one
    for s in [x for x in ORDER if x in stages]:
        print(f"===== stage: {s} =====", flush=True)
        if s == "data":
            P.stage_data(cfg, paths, a.snapshot, a.manual_dir)
        elif s == "baselines":
            P.stage_baselines(cfg, paths)
        elif s == "train":
            g = P.stage_train(cfg, paths, max_runs=a.max_runs)
            print(g["status"].value_counts().to_string())
        elif s == "freeze":
            P.stage_freeze(cfg, paths)
        elif s == "test":
            P.stage_test(cfg, paths, a.allow_test_rerun_reason)
        elif s == "backtest":
            P.stage_backtest(cfg, paths)
        elif s == "figures":
            P.stage_figures(cfg, paths)
        elif s == "report":
            print(P.stage_report(cfg, paths))


if __name__ == "__main__":
    main()
