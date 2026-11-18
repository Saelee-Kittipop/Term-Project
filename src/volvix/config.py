"""Load the locked protocol (+ optional overrides) and compute a stable hash."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .io_utils import sha256_obj

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "protocol.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: list[str | Path] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for o in overrides or []:
        with open(o, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg.setdefault("run_kind", "formal")
    validate_config(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    return sha256_obj(cfg)[:16]


def validate_config(cfg: dict) -> None:
    tick = cfg["data"]["tickers"]
    for key in ("reference_weights", "alert_weights", "no_tilt_weights"):
        w = cfg["portfolio"][key]
        assert list(w.keys()) == list(tick), f"{key} must follow ticker order {tick}"
        assert abs(sum(w.values()) - 1.0) < 1e-12, f"{key} must sum to 1"
        assert all(v >= 0 for v in w.values()), f"{key} must be long-only"
    # Alert portfolio = 50% reference + 50% SHY (report section 'Backtest')
    ref, alt = cfg["portfolio"]["reference_weights"], cfg["portfolio"]["alert_weights"]
    for t in tick:
        expected = 0.5 * ref[t] + (0.5 if t == "SHY" else 0.0)
        assert abs(alt[t] - expected) < 1e-12, f"alert weight {t} != 0.5*ref + 0.5*SHY"
    assert cfg["target"]["horizon"] == 5 and cfg["target"]["ddof"] == 1
    assert cfg["features"]["lookback"] == 60


def weights_vector(cfg: dict, key: str):
    import numpy as np
    return np.array([cfg["portfolio"][key][t] for t in cfg["data"]["tickers"]], dtype=float)
