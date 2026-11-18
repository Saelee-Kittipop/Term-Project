#!/usr/bin/env python
"""Pack the small, reviewable artifacts of a run into one zip (checkpoints and parquet stay on Drive).

    python scripts/make_bundle.py --root <DRIVE_ROOT> --run-name formal_v1 [--extra-run smoke_v1] [--out bundle.zip]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SMALL = {".csv", ".json", ".md", ".yaml", ".txt", ".png", ".svg", ".log"}
MAX_BYTES = 25 * 2**20


def add_run(z: zipfile.ZipFile, run: Path, arc_prefix: str) -> int:
    n = 0
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.suffix in SMALL and p.stat().st_size <= MAX_BYTES:
            z.write(p, f"{arc_prefix}/{p.relative_to(run)}")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--extra-run", action="append", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out or root / f"bundle_{a.run_name}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        n = add_run(z, root / "runs" / a.run_name, f"runs/{a.run_name}")
        for e in a.extra_run:
            n += add_run(z, root / "runs" / e, f"runs/{e}")
        for meta in (root / "data" / "raw").glob("*/snapshot_metadata.json"):
            z.write(meta, f"data/raw/{meta.parent.name}/snapshot_metadata.json")
            n += 1
        for log in root.glob("*.log"):
            z.write(log, log.name)
            n += 1
    print(f"{out} ({n} files, {out.stat().st_size/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
