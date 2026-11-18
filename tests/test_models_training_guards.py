"""Model shapes, training resume determinism, selection rules, and protocol guards."""
import numpy as np
import pandas as pd
import pytest
import torch

from volvix import pipeline as P
from volvix.config import load_config
from volvix import training as T
from volvix.models import LSTMRegressor, TransformerRegressor, count_parameters
from volvix.training import RunSpec, grid_specs, select_configs, train_one


def test_model_shapes_and_sizes():
    x = torch.randn(4, 60, 6)
    lstm, tr = LSTMRegressor(6, dropout=0.1), TransformerRegressor(6, dropout=0.1)
    assert lstm(x).shape == (4,) and tr(x).shape == (4,)
    assert lstm.lstm.hidden_size == 32 and tr.encoder.layers[0].self_attn.num_heads == 2
    assert tr.encoder.layers[0].linear1.out_features == 64 and len(tr.encoder.layers) == 1
    assert 0 < count_parameters(lstm) < 10000 and 0 < count_parameters(tr) < 20000


def test_grid_is_48_runs(cfg):
    specs = grid_specs(cfg)
    assert len(specs) == 48 and len({s.run_id for s in specs}) == 48


def _arrays(n=96, f=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 60, f)).astype(np.float32)
    z = (X[:, -5:, 0].std(axis=1) + rng.normal(0, 0.05, n)).astype(np.float32)
    d = pd.bdate_range("2010-01-01", periods=n).to_numpy()
    return {"train": {"X": X[:64], "z": z[:64], "dates": d[:64]}, "validation": {"X": X[64:], "z": z[64:], "dates": d[64:]}}


def test_training_learns_and_resume_is_identical(tmp_path, cfg, monkeypatch):
    """Crash during epoch 4, resume from last.pt: the history must equal an uninterrupted run."""
    c = {**cfg, "training": {**cfg["training"], "max_epochs": 6, "patience": 10}}
    arr = _arrays()
    spec = RunSpec("LSTM", "R", 0.001, 0.1, 11)
    cpu = torch.device("cpu")
    full = train_one(spec, arr, c, tmp_path / "full", device=cpu, log=lambda *a: None)

    class Crash(Exception):
        pass

    orig, calls = T._mse, {"n": 0}

    def crashing(*a, **k):
        calls["n"] += 1
        if calls["n"] > 6:                              # 2 evaluations per epoch -> crash inside epoch 4
            raise Crash()
        return orig(*a, **k)

    monkeypatch.setattr(T, "_mse", crashing)
    with pytest.raises(Crash):
        train_one(spec, arr, c, tmp_path / "part", device=cpu, log=lambda *a: None)
    monkeypatch.setattr(T, "_mse", orig)
    assert (tmp_path / "part" / "last.pt").exists() and not (tmp_path / "part" / "status.json").exists()
    resumed = train_one(spec, arr, c, tmp_path / "part", device=cpu, log=lambda *a: None)
    h1 = pd.read_csv(tmp_path / "full" / "history.csv")
    h2 = pd.read_csv(tmp_path / "part" / "history.csv")
    assert len(h1) == len(h2) == 6
    assert np.allclose(h1["val_logmse"], h2["val_logmse"], rtol=1e-5, atol=1e-7)
    assert resumed["best_epoch"] == full["best_epoch"]
    assert h1["train_logmse_eval"].iloc[-1] < h1["train_logmse_eval"].iloc[0]   # it learns
    # best checkpoint restored: validation prediction loss equals the best epoch's val loss
    vp = pd.read_parquet(tmp_path / "full" / "val_pred.parquet")
    assert np.isclose(np.mean((vp["z_pred"] - vp["z_true"]) ** 2), full["best_val_logmse"], rtol=1e-5)


def test_selection_averages_feature_sets_and_seeds_with_tie_break(cfg):
    rows = []
    for s in grid_specs(cfg):
        v = 1.0 if (s.lr, s.dropout) in [(0.001, 0.1), (0.001, 0.2)] else 2.0   # tie between two dropouts
        rows.append({"arch": s.arch, "lr": s.lr, "dropout": s.dropout, "best_val_logmse": v, "status": "completed"})
    sel = select_configs(cfg, pd.DataFrame(rows))
    assert sel["LSTM"]["dropout"] == 0.2 and sel["LSTM"]["lr"] == 0.001          # higher dropout wins ties
    rows2 = [{**r, "best_val_logmse": 1.0} for r in rows]
    sel2 = select_configs(cfg, pd.DataFrame(rows2))
    assert sel2["Transformer"]["dropout"] == 0.2 and sel2["Transformer"]["lr"] == 0.0003  # then lower lr
    rows[0]["status"] = "pending"
    assert select_configs(cfg, pd.DataFrame(rows))[rows[0]["arch"]]["status"] == "pending"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    cfg = load_config(overrides=[P.Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"])
    paths = P.Paths(root, "smoke")
    P.stage_data(cfg, paths, "synthetic", log=lambda *a: None)
    P.stage_baselines(cfg, paths, log=lambda *a: None)
    P.stage_train(cfg, paths, log=lambda *a: None)
    P.stage_freeze(cfg, paths, log=lambda *a: None)
    return cfg, paths


def test_smoke_run_cannot_open_test(smoke_run):
    cfg, paths = smoke_run
    with pytest.raises(RuntimeError, match="test evaluation is allowed only"):
        P.stage_test(cfg, paths)
    assert not (paths.test / "TEST_OPENED.json").exists()


def test_formal_run_refuses_synthetic_data(tmp_path, cfg):
    with pytest.raises(RuntimeError, match="formal runs must use a real data snapshot"):
        P.stage_data(cfg, P.Paths(tmp_path, "formal"), "synthetic", log=lambda *a: None)


def test_freeze_detects_changed_artifacts(smoke_run):
    cfg, paths = smoke_run
    fz = P.verify_freeze(paths)
    assert fz["tau"] > 0 and set(fz["cutoffs"]) == set(P.ALL_MODELS)
    sc = paths.scalers / "scaler_R.json"
    orig = sc.read_text()
    sc.write_text(orig.replace('"n_fit_rows"', '"n_fit_rows_tampered"'))
    with pytest.raises(RuntimeError, match="changed after freeze"):
        P.verify_freeze(paths)
    sc.write_text(orig)


def test_tau_and_cutoffs_use_train_and_validation_only(smoke_run):
    cfg, paths = smoke_run
    fz = P.verify_freeze(paths)
    s = pd.read_parquet(paths.dataset / "samples.parquet")
    assert np.isclose(fz["tau"], np.quantile(s.loc[s["split"] == "train", "y"], 0.9))
    vf = pd.read_csv(paths.validation / "validation_forecasts.csv")
    assert len(vf) == int((s["split"] == "validation").sum())
    q = fz["cutoffs"]["EWMA"]["q_val_alert_freq"]
    assert np.isclose(q, (vf["EWMA"] > fz["cutoffs"]["EWMA"]["cutoff"]).mean())
    w = np.array(fz["static_comparator_weights"]["EWMA"])
    assert np.isclose(w.sum(), 1.0)
