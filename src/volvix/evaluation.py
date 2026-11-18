"""Forecast metrics, high-risk alerts, block bootstrap, and supplementary DM / VaR backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------------- forecast metrics
def forecast_metrics(y: np.ndarray, f: np.ndarray, eps: float = 1e-6) -> dict:
    y, f = np.asarray(y, float), np.asarray(f, float)
    e = f - y
    ly, lf = np.log(np.maximum(y, eps)), np.log(np.maximum(f, eps))
    v, vh = y ** 2, np.maximum(f, eps) ** 2
    return {"n": int(len(y)),
            "MAE_pp": float(np.mean(np.abs(e)) * 100), "RMSE_pp": float(np.sqrt(np.mean(e ** 2)) * 100),
            "log_RMSE": float(np.sqrt(np.mean((lf - ly) ** 2))),
            "QLIKE": float(np.mean(np.log(vh) + v / vh)),
            "mean_forecast_pp": float(np.mean(f) * 100), "mean_actual_pp": float(np.mean(y) * 100)}


def loss_series(y, f, kind: str, eps: float = 1e-6) -> np.ndarray:
    y, f = np.asarray(y, float), np.asarray(f, float)
    if kind == "log_se":
        return (np.log(np.maximum(f, eps)) - np.log(np.maximum(y, eps))) ** 2
    if kind == "se":
        return (f - y) ** 2
    if kind == "ae":
        return np.abs(f - y)
    if kind == "qlike":
        vh = np.maximum(f, eps) ** 2
        return np.log(vh) + y ** 2 / vh
    raise ValueError(kind)


# ------------------------------------------------------------------------------- alerts
def tau_from_train(y_train: np.ndarray, q: float = 0.90) -> float:
    return float(np.quantile(np.asarray(y_train, float), q))     # numpy 'linear' interpolation


def confusion(h: np.ndarray, a: np.ndarray) -> dict:
    h, a = np.asarray(h, bool), np.asarray(a, bool)
    return {"TP": int((h & a).sum()), "FP": int((~h & a).sum()), "FN": int((h & ~a).sum()), "TN": int((~h & ~a).sum())}


def prf(h, a) -> dict:
    c = confusion(h, a)
    p = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0     # no alert -> precision 0
    pos = c["TP"] + c["FN"]
    r = c["TP"] / pos if pos else np.nan                                  # no positive -> not evaluable
    f1 = (2 * p * r / (p + r) if (p + r) else 0.0) if pos else np.nan
    if pos and c["TP"] + c["FP"] == 0:
        f1 = 0.0
    return {**c, "precision": p, "recall": r, "F1": f1}


def select_cutoff(f_val: np.ndarray, h_val: np.ndarray, tau: float) -> dict:
    """Max-F1 cutoff on validation forecasts. Alert iff forecast > c. Candidates: every unique forecast plus a
    value below the minimum (alert on all). Ties -> highest cutoff. No validation positive -> c = tau."""
    f_val, h_val = np.asarray(f_val, float), np.asarray(h_val, bool)
    if not h_val.any():
        return {"cutoff": float(tau), "rule": "fallback_tau_no_validation_positive", "val_F1": np.nan}
    uniq = np.unique(f_val)
    cands = np.concatenate([[np.nextafter(uniq[0], -np.inf)], uniq])
    best_c, best_f1 = None, -1.0
    for c in cands:                       # ascending, so '>=' keeps the highest cutoff among ties
        f1 = prf(h_val, f_val > c)["F1"]
        if f1 >= best_f1:
            best_c, best_f1 = c, f1
    return {"cutoff": float(best_c), "rule": "max_F1_validation_highest_on_tie", "val_F1": float(best_f1)}


def alert_metrics(y: np.ndarray, f: np.ndarray, tau: float, cutoff: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    h, a = np.asarray(y) > tau, np.asarray(f) > cutoff
    out = prf(h, a)
    n = len(h)
    out.update({"n": int(n), "prevalence": float(h.mean()) if n else np.nan, "alert_rate": float(a.mean()) if n else np.nan,
                "false_alerts_per_100": float(100 * out["FP"] / n) if n else np.nan, "missed_high_risk": out["FN"]})
    both = h.any() and (~h).any()
    out["AP"] = float(average_precision_score(h, f)) if h.any() else np.nan
    out["ROC_AUC_supplementary"] = float(roc_auc_score(h, f)) if both else np.nan
    return out


# ------------------------------------------------------------------------------- bootstrap
def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=k)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def paired_block_bootstrap(d: np.ndarray, block: int = 20, reps: int = 1000, seed: int = 2026) -> dict:
    """CI of mean(d) where d is a paired loss difference on the same dates (moving-block bootstrap)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = np.array([d[moving_block_indices(len(d), block, rng)].mean() for _ in range(reps)])
    return {"mean_diff": float(d.mean()), "ci_low_2.5": float(np.quantile(means, 0.025)),
            "ci_high_97.5": float(np.quantile(means, 0.975)), "share_boot_below_0": float((means < 0).mean()),
            "block": block, "reps": reps}


# ------------------------------------------------------------------------------- supplementary
def diebold_mariano(l1: np.ndarray, l2: np.ndarray, h: int = 5, hac_lag: int | None = None) -> dict:
    """DM test on d = l1 - l2 (negative mean -> model 1 better). Newey-West variance with lag h-1 (overlapping
    targets) and the Harvey-Leybourne-Newbold small-sample correction, t(n-1) reference. SUPPLEMENTARY ONLY."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    n = len(d)
    L = h - 1 if hac_lag is None else hac_lag
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(L + 1)]
    var = gamma[0] + 2 * sum((1 - k / (L + 1)) * gamma[k] for k in range(1, L + 1))
    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": float(d.mean()), "n": n}
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm_stat": float(stat), "p_value": float(p), "mean_diff": float(d.mean()), "n": n, "hac_lag": L}


def kupiec_pof(violations: np.ndarray, p: float) -> dict:
    x = np.asarray(violations, bool)
    n, k = len(x), int(x.sum())
    ph = k / n if n else np.nan
    def ll(pp):
        pp = min(max(pp, 1e-12), 1 - 1e-12)
        return (n - k) * np.log(1 - pp) + k * np.log(pp)
    lr = -2 * (ll(p) - ll(ph)) if n else np.nan
    return {"n": n, "violations": k, "rate": ph, "expected": p, "LR_pof": float(lr),
            "p_value": float(stats.chi2.sf(lr, 1)) if n else np.nan}


def christoffersen(violations: np.ndarray, p: float) -> dict:
    x = np.asarray(violations, int)
    a, b = x[:-1], x[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    def s(v):
        return min(max(v, 1e-12), 1 - 1e-12)
    p01 = s(n01 / (n00 + n01)) if (n00 + n01) else 1e-12
    p11 = s(n11 / (n10 + n11)) if (n10 + n11) else 1e-12
    p1 = s((n01 + n11) / max(n00 + n01 + n10 + n11, 1))
    ll0 = (n00 + n10) * np.log(1 - p1) + (n01 + n11) * np.log(p1)
    ll1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    lr_ind = float(-2 * (ll0 - ll1))
    pof = kupiec_pof(x.astype(bool), p)
    lr_cc = pof["LR_pof"] + lr_ind
    return {"LR_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)), "LR_cc": float(lr_cc),
            "p_cc": float(stats.chi2.sf(lr_cc, 2)), "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def var_backtest(forecast_ann_vol: np.ndarray, fwd_ret_1d: np.ndarray, fwd_ret_5d: np.ndarray,
                 levels=(0.95, 0.99), ann: int = 252) -> pd.DataFrame:
    """Zero-mean Gaussian VaR from the 5-day volatility forecast. 1-day: r_{t+1}; 5-day: sum r_{t+1..t+5}.
    SUPPLEMENTARY: the 5-day violations overlap, so independence tests on them are not valid."""
    rows = []
    sd1 = np.asarray(forecast_ann_vol) / np.sqrt(ann)
    for hzn, ret, sd in ((1, fwd_ret_1d, sd1), (5, fwd_ret_5d, sd1 * np.sqrt(5))):
        for lvl in levels:
            var = stats.norm.ppf(1 - lvl) * sd
            viol = np.asarray(ret) < var
            k = kupiec_pof(viol, 1 - lvl)
            c = christoffersen(viol, 1 - lvl)
            rows.append({"horizon_days": hzn, "level": lvl, **k, **{f"cc_{kk}": vv for kk, vv in c.items()},
                         "note": "overlapping 5-day windows; tests indicative only" if hzn == 5 else ""})
    return pd.DataFrame(rows)
