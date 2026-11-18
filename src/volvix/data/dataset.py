"""Forecast origins, forward target, input windows, chronological splits with purge, train-only scaler.

Alignment at origin t (report appendix B):
    input rows  : sessions t-59 .. t            (row s = returns of s, VIX of s-1 for set RV)
    target      : r_ref of sessions t+1 .. t+5  -> y_t = sd(ddof=1) * sqrt(252)
    target_end  : session t+5 (stored explicitly)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io_utils import write_csv, write_json

SPLITS = ("train", "validation", "test")


def forward_target(r: np.ndarray, valid: np.ndarray, horizon: int = 5, ddof: int = 1, ann: int = 252):
    """y[t] = sd(r[t+1..t+h], ddof) * sqrt(ann); NaN when any of those returns is invalid or missing."""
    n = len(r)
    y = np.full(n, np.nan)
    for t in range(n - horizon):
        seg = slice(t + 1, t + 1 + horizon)
        if valid[seg].all():
            y[t] = np.std(r[seg], ddof=ddof) * np.sqrt(ann)
    return y


def _window_all(mask: np.ndarray, lookback: int) -> np.ndarray:
    """out[t] = mask[t-lookback+1 .. t].all(); False when the window starts before index 0."""
    m = mask.astype(int)
    c = np.concatenate([[0], np.cumsum(m)])
    out = np.zeros(len(mask), dtype=bool)
    idx = np.arange(lookback - 1, len(mask))
    out[idx] = (c[idx + 1] - c[idx + 1 - lookback]) == lookback
    return out


def build_samples(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tick = cfg["data"]["tickers"]
    L, H = cfg["features"]["lookback"], cfg["target"]["horizon"]
    eps = cfg["target"]["log_epsilon"]
    dates = panel.index
    n = len(dates)
    valid = panel["ret_valid"].to_numpy(bool)
    r_ref = panel["r_ref"].to_numpy(float)
    vix_ok = panel["vix_lag1"].notna().to_numpy(bool)

    in_R = _window_all(valid, L)
    in_RV = in_R & _window_all(vix_ok, L)
    hv_ok = _window_all(valid, cfg["baselines"]["hv_window"])
    y = forward_target(r_ref, valid, H, cfg["target"]["ddof"], cfg["target"]["annualization"])

    pos = np.arange(n)
    df = pd.DataFrame({"pos": pos, "origin_date": dates})
    df["input_start_date"] = [dates[p - L + 1] if p - L + 1 >= 0 else pd.NaT for p in pos]
    df["input_end_date"] = dates
    df["target_start_date"] = [dates[p + 1] if p + 1 < n else pd.NaT for p in pos]
    df["target_end_date"] = [dates[p + H] if p + H < n else pd.NaT for p in pos]
    df["input_valid_R"], df["input_valid_RV"], df["hv_valid"] = in_R, in_RV, hv_ok
    df["y"] = y
    df["target_valid"] = np.isfinite(y)
    df["z"] = np.log(np.maximum(y, eps))
    df.loc[~df["target_valid"], "z"] = np.nan
    # common information set for every model (report: same evaluation dates for all models)
    df["input_valid_common"] = in_R & in_RV & hv_ok

    df["period"] = None
    df["split"] = None
    df["purged_boundary"] = False
    for s in SPLITS:
        sp = cfg["splits"][s]
        lo = pd.Timestamp(sp["start"]) if sp.get("start") else dates[0]
        hi = pd.Timestamp(sp["end"])
        in_period = (df["origin_date"] >= lo) & (df["origin_date"] <= hi)
        df.loc[in_period, "period"] = s
        inside = df["target_end_date"].notna() & (df["target_end_date"] <= hi)
        ok = in_period & df["input_valid_common"] & df["target_valid"] & inside
        df.loc[ok, "split"] = s
        df.loc[in_period & df["input_valid_common"] & ~inside, "purged_boundary"] = True
    df["score_eligible"] = df["split"].notna()
    # trading origins in the test period: forecasts may drive the strategy even without a complete label
    df["trade_eligible"] = (df["period"] == "test") & df["input_valid_common"]
    return df


def split_manifest(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        per = samples[samples["period"] == s]
        sel = samples[samples["split"] == s]
        rows.append({
            "split": s,
            "period_first_origin": per["origin_date"].min(), "period_last_origin": per["origin_date"].max(),
            "candidate_origins": len(per),
            "dropped_input_window_incomplete": int((~per["input_valid_common"]).sum()),
            "dropped_target_invalid_or_missing": int((per["input_valid_common"] & ~per["target_valid"] & ~per["purged_boundary"]).sum()),
            "purged_target_crosses_boundary": int(per["purged_boundary"].sum()),
            "net_samples": len(sel),
            "first_origin": sel["origin_date"].min(), "last_origin": sel["origin_date"].max(),
            "last_target_end": sel["target_end_date"].max(),
        })
    return pd.DataFrame(rows)


def feature_matrix(panel: pd.DataFrame, cfg: dict, feature_set: str) -> tuple[np.ndarray, list[str]]:
    cols = [f"ret_{t}" for t in cfg["data"]["tickers"]]
    if cfg["features"]["sets"][feature_set]["use_vix"]:
        cols = cols + ["vix_lag1"]
    return panel[cols].to_numpy(float), cols


class TrainOnlyScaler:
    """StandardScaler (population sd, ddof=0) fitted on the unique daily rows inside training windows."""

    def __init__(self, mean=None, scale=None, columns=None, n_rows=None, first_row=None, last_row=None):
        self.mean_, self.scale_, self.columns = mean, scale, columns
        self.n_rows, self.first_row, self.last_row = n_rows, first_row, last_row

    @staticmethod
    def training_rows(train_pos: np.ndarray, lookback: int) -> np.ndarray:
        rows = set()
        for p in train_pos:
            rows.update(range(p - lookback + 1, p + 1))
        return np.array(sorted(rows), dtype=int)

    def fit(self, feat: np.ndarray, rows: np.ndarray, columns: list[str], dates=None):
        x = feat[rows]
        if not np.isfinite(x).all():
            raise ValueError("non-finite values inside training input rows")
        self.mean_ = x.mean(axis=0)
        sd = x.std(axis=0, ddof=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        self.columns, self.n_rows = list(columns), int(len(rows))
        if dates is not None:
            self.first_row, self.last_row = str(dates[rows[0]].date()), str(dates[rows[-1]].date())
        return self

    def transform(self, feat: np.ndarray) -> np.ndarray:
        return (feat - self.mean_) / self.scale_

    def to_dict(self) -> dict:
        return {"type": "StandardScaler(ddof=0)", "columns": self.columns, "mean": list(map(float, self.mean_)),
                "scale": list(map(float, self.scale_)), "n_fit_rows": self.n_rows,
                "fit_first_row_date": self.first_row, "fit_last_row_date": self.last_row}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainOnlyScaler":
        return cls(np.array(d["mean"]), np.array(d["scale"]), d["columns"], d["n_fit_rows"],
                   d.get("fit_first_row_date"), d.get("fit_last_row_date"))


def make_windows(feat_scaled: np.ndarray, positions: np.ndarray, lookback: int) -> np.ndarray:
    """X[k] = feat_scaled[p-lookback+1 .. p] for p in positions -> (N, lookback, F) float32."""
    out = np.empty((len(positions), lookback, feat_scaled.shape[1]), dtype=np.float32)
    for k, p in enumerate(positions):
        out[k] = feat_scaled[p - lookback + 1: p + 1]
    if not np.isfinite(out).all():
        raise ValueError("non-finite values inside model input windows")
    return out


def alignment_checks(samples: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Date-level checks executed on the real data before training. Each row records pass/fail as computed."""
    L, H = cfg["features"]["lookback"], cfg["target"]["horizon"]
    dates = panel.index
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    sc = samples[samples["score_eligible"]]
    pos = sc["pos"].to_numpy()
    add("input_ends_at_origin", (sc["input_end_date"] == sc["origin_date"]).all(), "input_end_date == origin t")
    add("input_has_lookback_rows", all(dates[p - L + 1] == s for p, s in zip(pos, sc["input_start_date"])),
        f"input_start_date == session t-{L-1}")
    add("target_starts_next_session", all(dates[p + 1] == s for p, s in zip(pos, sc["target_start_date"])), "t+1")
    add("target_ends_t_plus_5", all(dates[p + H] == s for p, s in zip(pos, sc["target_end_date"])), f"t+{H}")
    r = panel["r_ref"]
    alt = (r.rolling(H).std(ddof=1).shift(-H) * np.sqrt(cfg["target"]["annualization"])).to_numpy()
    add("target_equals_rolling_std_shift_minus5", np.allclose(sc["y"].to_numpy(), alt[pos], rtol=1e-10, atol=1e-14),
        "y == rolling(5).std(ddof=1).shift(-5)*sqrt(252)")
    for s in SPLITS:
        hi = pd.Timestamp(cfg["splits"][s]["end"])
        part = samples[samples["split"] == s]
        add(f"{s}_target_end_inside_split", (part["target_end_date"] <= hi).all(), f"max={part['target_end_date'].max()}")
    tr_end = samples.loc[samples["split"] == "train", "target_end_date"].max()
    va_first = samples.loc[samples["split"] == "validation", "target_start_date"].min()
    va_end = samples.loc[samples["split"] == "validation", "target_end_date"].max()
    te_first = samples.loc[samples["split"] == "test", "target_start_date"].min()
    add("no_target_overlap_train_validation", tr_end < va_first, f"train last target end {tr_end} < val first target start {va_first}")
    add("no_target_overlap_validation_test", va_end < te_first, f"val last target end {va_end} < test first target start {te_first}")
    v = panel["vix_lag1"].to_numpy()
    vv = panel["vix"].to_numpy()
    add("vix_lag1_is_previous_session", np.allclose(v[1:], vv[:-1], equal_nan=True), "vix_lag1[s] == vix[s-1]")
    tr = samples[samples["split"] == "train"]
    add("first_train_origin_uses_first_60_returns", tr["pos"].min() >= L, f"first train origin pos {tr['pos'].min()} >= {L}")
    return pd.DataFrame(checks)


def example_rows(samples: pd.DataFrame) -> pd.DataFrame:
    """A few concrete origins (first/last of each split) for the Thai report."""
    picks = []
    for s in SPLITS:
        part = samples[samples["split"] == s]
        if len(part):
            picks += [part.iloc[0], part.iloc[-1]]
    cols = ["split", "origin_date", "input_start_date", "input_end_date", "target_start_date", "target_end_date", "y"]
    return pd.DataFrame(picks)[cols]


def save_dataset_artifacts(samples, panel, cfg, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out_dir / "samples.parquet")
    man = split_manifest(samples)
    write_csv(out_dir / "split_manifest.csv", man)
    chk = alignment_checks(samples, panel, cfg)
    write_csv(out_dir / "alignment_checks.csv", chk)
    write_csv(out_dir / "alignment_examples.csv", example_rows(samples))
    res = {"split_manifest": man.to_dict(orient="records"), "alignment_checks_all_passed": bool(chk["passed"].all()),
           "failed_checks": chk.loc[~chk["passed"], "check"].tolist()}
    write_json(out_dir / "split_manifest.json", res)
    return res
