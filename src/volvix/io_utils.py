"""Hashing, JSON/CSV IO, environment capture and the run manifest."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, df: pd.DataFrame, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)
    return path


def git_commit(repo_dir: str | Path | None = None) -> str:
    repo_dir = Path(repo_dir or Path(__file__).resolve().parents[2])
    env_commit = os.environ.get("VOLVIX_GIT_COMMIT")
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            commit = out.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return commit + ("-dirty" if dirty else "")
    except Exception:
        pass
    return env_commit or "unknown"


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ["torch", "arch", "sklearn", "yfinance", "scipy", "matplotlib", "pandas_market_calendars"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def pip_freeze(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    path.write_text(out.stdout)
    return path


class RunManifest:
    """Append-only record of every stage: commit, config hash, data hashes, seeds, runtime, status."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = read_json(self.path) if self.path.exists() else {"created_utc": utc_now(), "stages": []}

    def save(self):
        write_json(self.path, self.data)

    def set(self, **kw):
        self.data.update(kw)
        self.save()

    def stage(self, name: str):
        return _StageCtx(self, name)

    def completed(self, name: str) -> bool:
        return any(s["stage"] == name and s["status"] == "completed" for s in self.data["stages"])


class _StageCtx:
    def __init__(self, manifest: RunManifest, name: str):
        self.m, self.name, self.extra = manifest, name, {}

    def __enter__(self):
        self.t0 = time.time()
        self.rec = {"stage": self.name, "started_utc": utc_now(), "git_commit": git_commit(), "status": "running"}
        self.m.data["stages"].append(self.rec)
        self.m.save()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.rec["runtime_sec"] = round(time.time() - self.t0, 2)
        self.rec["finished_utc"] = utc_now()
        self.rec["status"] = "completed" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        self.rec.update(self.extra)
        self.m.save()
        return False
