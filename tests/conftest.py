import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volvix.config import load_config  # noqa: E402
from volvix.data.clean import clean_snapshot  # noqa: E402
from volvix.data.dataset import build_samples  # noqa: E402
from volvix.data.synthetic import make_synthetic_snapshot  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("synth")
    snap = make_synthetic_snapshot(cfg, root / "data")
    out = root / "processed"
    q = clean_snapshot(cfg, snap, out)
    panel = pd.read_parquet(out / "panel.parquet")
    samples = build_samples(panel, cfg)
    return {"root": root, "snap": snap, "panel": panel, "samples": samples, "quality": q, "processed": out}
