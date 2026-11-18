"""48-run grid (2 architectures x 2 feature sets x 4 configs x 3 seeds) with resumable checkpoints."""
from __future__ import annotations

import gc
import itertools
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.dataset import TrainOnlyScaler, feature_matrix, make_windows
from .io_utils import read_json, sha256_file, utc_now, write_csv, write_json
from .models import build_model, count_parameters

ARCHS = ("LSTM", "Transformer")
FEATURE_SETS = ("R", "RV")
MODEL_CODES = {("LSTM", "R"): "L_R", ("LSTM", "RV"): "L_RV", ("Transformer", "R"): "T_R", ("Transformer", "RV"): "T_RV"}


@dataclass(frozen=True)
class RunSpec:
    arch: str
    feature_set: str
    lr: float
    dropout: float
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.arch}_{self.feature_set}_lr{self.lr:g}_do{self.dropout:g}_s{self.seed}"

    @property
    def config_key(self) -> str:
        return f"lr{self.lr:g}_do{self.dropout:g}"


def grid_specs(cfg: dict) -> list[RunSpec]:
    g = cfg["training"]["grid"]
    return [RunSpec(a, f, lr, do, s) for a, f, lr, do, s in
            itertools.product(ARCHS, FEATURE_SETS, g["learning_rate"], g["dropout"], cfg["training"]["seeds"])]


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            # memory-efficient / flash attention backward is non-deterministic on CUDA; use the math kernel
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------ data
def fit_scalers(panel: pd.DataFrame, samples: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = cfg["features"]["lookback"]
    train_pos = samples.loc[samples["split"] == "train", "pos"].to_numpy()
    rows = TrainOnlyScaler.training_rows(train_pos, L)
    res = {}
    for fs in FEATURE_SETS:
        feat, cols = feature_matrix(panel, cfg, fs)
        sc = TrainOnlyScaler().fit(feat, rows, cols, panel.index)
        write_json(out_dir / f"scaler_{fs}.json", sc.to_dict())
        res[fs] = sc.to_dict()
    return res


def load_split_arrays(panel, samples, cfg, fs: str, scaler_dir: Path, splits=("train", "validation")) -> dict:
    sc = TrainOnlyScaler.from_dict(read_json(scaler_dir / f"scaler_{fs}.json"))
    feat, cols = feature_matrix(panel, cfg, fs)
    assert cols == sc.columns
    fz = sc.transform(feat)
    L = cfg["features"]["lookback"]
    out = {}
    for s in splits:
        part = samples[samples["split"] == s]
        pos = part["pos"].to_numpy()
        out[s] = {"X": make_windows(fz, pos, L), "z": part["z"].to_numpy(np.float32), "pos": pos,
                  "dates": part["origin_date"].to_numpy()}
    return out


def windows_for_positions(panel, cfg, fs: str, scaler_dir: Path, positions: np.ndarray) -> np.ndarray:
    sc = TrainOnlyScaler.from_dict(read_json(scaler_dir / f"scaler_{fs}.json"))
    feat, _ = feature_matrix(panel, cfg, fs)
    return make_windows(sc.transform(feat), positions, cfg["features"]["lookback"])


# ------------------------------------------------------------------------------------------ train
@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).float().cpu().numpy())
        del xb
    return np.concatenate(out) if out else np.empty(0)


def _mse(model, X, y, device, batch=2048) -> float:
    p = predict(model, X, device, batch)
    return float(np.mean((p - y) ** 2))


def train_one(spec: RunSpec, arrays: dict, cfg: dict, run_dir: Path, device=None, log=print) -> dict:
    """Train one run; resumable from run_dir/last.pt. Returns the status dict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "completed":
        return read_json(status_path)
    device = device or device_auto()
    tc = cfg["training"]
    set_seed(spec.seed, tc.get("deterministic", True))
    Xtr, ztr = arrays["train"]["X"], arrays["train"]["z"]
    Xva, zva = arrays["validation"]["X"], arrays["validation"]["z"]
    model = build_model(spec.arch, Xtr.shape[2], cfg, spec.dropout).to(device)
    if tc.get("init_output_bias_to_train_mean", False):
        with torch.no_grad():
            model.head[-1].bias.fill_(float(np.mean(ztr)))
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(spec.seed)
    start_epoch, best, best_epoch, wait, history, best_state, elapsed_prev = 1, np.inf, 0, 0, [], None, 0.0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best, best_epoch, wait = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["wait"]
        history, best_state, elapsed_prev = ck["history"], ck["best_state"], ck.get("elapsed", 0.0)
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        log(f"[resume] {spec.run_id} from epoch {ck['epoch']}")
    write_json(run_dir / "spec.json", {**asdict(spec), "run_id": spec.run_id, "model_code": MODEL_CODES[(spec.arch, spec.feature_set)]})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ztr_t = torch.from_numpy(ztr).to(device)
    bs = tc["batch_size"]
    t0 = time.time()
    stopped_early = False
    for epoch in range(start_epoch, tc["max_epochs"] + 1):
        if wait >= tc["patience"]:
            stopped_early = True
            break
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen) if tc.get("shuffle_train_batches", True) else torch.arange(len(Xtr))
        perm = perm.to(device)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr_t[idx]), ztr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        tr_loss = tot / cnt
        va = _mse(model, Xva, zva, device)
        tr_eval = _mse(model, Xtr, ztr, device)
        improved = va < best
        if improved:
            best, best_epoch, wait = va, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best.pt")
        else:
            wait += 1
        history.append({"epoch": epoch, "train_logmse_batches": tr_loss, "train_logmse_eval": tr_eval,
                        "val_logmse": va, "improved": improved})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch, "wait": wait, "history": history, "best_state": best_state,
                    "gen_state": gen.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "elapsed": elapsed_prev + time.time() - t0}, last)
    else:
        stopped_early = wait >= tc["patience"]
    model.load_state_dict(best_state)                 # restore best validation epoch
    val_pred = predict(model, Xva, device)
    tr_pred = predict(model, Xtr, device)
    write_csv(run_dir / "history.csv", pd.DataFrame(history))
    pd.DataFrame({"origin_date": arrays["validation"]["dates"], "z_true": zva, "z_pred": val_pred}).to_parquet(run_dir / "val_pred.parquet")
    pd.DataFrame({"origin_date": arrays["train"]["dates"], "z_true": ztr, "z_pred": tr_pred}).to_parquet(run_dir / "train_pred.parquet")
    status = {"run_id": spec.run_id, **asdict(spec), "model_code": MODEL_CODES[(spec.arch, spec.feature_set)],
              "status": "completed", "finished_utc": utc_now(), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "best_epoch": int(best_epoch), "best_val_logmse": float(best), "epochs_run": len(history),
              "stopped_early": bool(stopped_early), "n_params": count_parameters(model),
              "train_time_sec": round(elapsed_prev + time.time() - t0, 2),
              "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
              "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
              "n_train": int(len(Xtr)), "n_val": int(len(Xva))}
    write_json(status_path, status)
    # free memory between runs (Colab T4)
    del model, opt, Xtr_t, ztr_t, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        last.unlink()                                 # best.pt + status are the durable artifacts
    except OSError:
        pass
    return status


def run_grid(panel, samples, cfg, out_dir: Path, scaler_dir: Path, only: list[str] | None = None,
             max_runs: int | None = None, log=print) -> pd.DataFrame:
    """Train every pending run of the grid. Safe to interrupt and call again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, done = {}, 0
    for spec in grid_specs(cfg):
        if only and spec.run_id not in only:
            continue
        sp = out_dir / spec.run_id / "status.json"
        if sp.exists() and read_json(sp).get("status") == "completed":
            continue
        if max_runs is not None and done >= max_runs:
            break
        if spec.feature_set not in cache:
            cache[spec.feature_set] = load_split_arrays(panel, samples, cfg, spec.feature_set, scaler_dir)
        log(f"[train] {spec.run_id}")
        st = train_one(spec, cache[spec.feature_set], cfg, out_dir / spec.run_id, log=log)
        log(f"[done]  {spec.run_id} best_epoch={st['best_epoch']} val_logmse={st['best_val_logmse']:.5f} "
            f"time={st['train_time_sec']}s")
        done += 1
    return grid_summary(cfg, out_dir)


def grid_summary(cfg, out_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in grid_specs(cfg):
        sp = out_dir / spec.run_id / "status.json"
        st = read_json(sp) if sp.exists() else {}
        rows.append({"run_id": spec.run_id, **asdict(spec), "model_code": MODEL_CODES[(spec.arch, spec.feature_set)],
                     "config_key": spec.config_key, "status": st.get("status", "pending"),
                     "best_val_logmse": st.get("best_val_logmse"), "best_epoch": st.get("best_epoch"),
                     "epochs_run": st.get("epochs_run"), "n_params": st.get("n_params"),
                     "train_time_sec": st.get("train_time_sec"), "peak_gpu_mem_mb": st.get("peak_gpu_mem_mb")})
    df = pd.DataFrame(rows)
    write_csv(out_dir / "grid_summary.csv", df)
    return df


def select_configs(cfg, grid: pd.DataFrame) -> dict:
    """One configuration per architecture: min mean validation log-MSE over both feature sets and all seeds.
    Ties: higher dropout, then lower learning rate."""
    sel = {}
    n_expected = len(FEATURE_SETS) * len(cfg["training"]["seeds"])
    for arch in ARCHS:
        g = grid[grid["arch"] == arch]
        agg = g.groupby(["lr", "dropout"]).agg(mean_val_logmse=("best_val_logmse", "mean"),
                                               n_completed=("status", lambda s: int((s == "completed").sum()))).reset_index()
        if (agg["n_completed"] < n_expected).any():
            sel[arch] = {"status": "pending", "table": agg.to_dict(orient="records")}
            continue
        agg = agg.sort_values(["mean_val_logmse", "dropout", "lr"], ascending=[True, False, True], kind="mergesort")
        top = agg.iloc[0]
        ties = agg[np.isclose(agg["mean_val_logmse"], top["mean_val_logmse"], rtol=0, atol=1e-12)]
        top = ties.sort_values(["dropout", "lr"], ascending=[False, True]).iloc[0]
        sel[arch] = {"status": "selected", "lr": float(top["lr"]), "dropout": float(top["dropout"]),
                     "mean_val_logmse": float(top["mean_val_logmse"]), "table": agg.to_dict(orient="records")}
    return sel


def ensemble_runs(cfg, selection: dict, arch: str, fs: str) -> list[RunSpec]:
    s = selection[arch]
    return [RunSpec(arch, fs, s["lr"], s["dropout"], seed) for seed in cfg["training"]["seeds"]]


def ensemble_predict_log(cfg, selection, arch, fs, X: np.ndarray, train_dir: Path, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of predicted log-vol over the selected config's seeds; returns (ensemble_z, per_seed_z[k, n])."""
    device = device or device_auto()
    per = []
    for spec in ensemble_runs(cfg, selection, arch, fs):
        model = build_model(arch, X.shape[2], cfg, spec.dropout).to(device)
        model.load_state_dict(torch.load(train_dir / spec.run_id / "best.pt", map_location=device, weights_only=True))
        per.append(predict(model, X, device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    per = np.vstack(per)
    return per.mean(axis=0), per


def to_vol(z: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(np.exp(z), eps)
