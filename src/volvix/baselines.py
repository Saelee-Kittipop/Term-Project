"""HV20, EWMA(0.94) and GARCH(1,1) forecasts of the annualized forward five-day sample SD.

All use the reference-portfolio return r_ref and information up to the origin t only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def hv_forecast(r: np.ndarray, valid: np.ndarray, window: int = 20, ann: int = 252) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for t in range(window - 1, len(r)):
        seg = slice(t - window + 1, t + 1)
        if valid[seg].all():
            out[t] = np.std(r[seg], ddof=1) * np.sqrt(ann)
    return out


def ewma_forecast(r: np.ndarray, valid: np.ndarray, lam: float = 0.94, init_n: int = 60, ann: int = 252,
                  first_return_pos: int = 1) -> tuple[np.ndarray, list[int]]:
    """Zero-mean RiskMetrics EWMA.

    At the origin of return number `init_n` (position first_return_pos+init_n-1) the variance forecast is
    mean(r_1^2..r_init_n^2); afterwards var_t = lam*var_{t-1} + (1-lam)*r_t^2. The first init_n returns are
    not re-processed. An invalid return leaves the variance unchanged and is reported.
    """
    n = len(r)
    var = np.full(n, np.nan)
    p0 = first_return_pos + init_n - 1
    init = r[first_return_pos:p0 + 1]
    if not valid[first_return_pos:p0 + 1].all():
        raise ValueError("EWMA initialisation window contains invalid returns")
    var[p0] = np.mean(init ** 2)
    skipped = []
    for t in range(p0 + 1, n):
        if valid[t]:
            var[t] = lam * var[t - 1] + (1 - lam) * r[t] ** 2
        else:
            var[t] = var[t - 1]
            skipped.append(t)
    return np.sqrt(ann * var), skipped


def garch_forecast(r: pd.Series, valid: pd.Series, train_end: str, horizon: int = 5, ann: int = 252,
                   p: int = 1, q: int = 1, dist: str = "normal") -> tuple[pd.Series, dict]:
    """GARCH(1,1), constant mean, Gaussian; parameters estimated on training returns only (date <= train_end),
    then the conditional variance is filtered forward with the parameters held fixed.

    sigma_hat_{t,5} = sqrt(252/5 * sum_{h=1..5} v_{t+h|t}); returns are scaled to percent inside arch
    and variances divided by 10000 back to decimal.
    """
    from arch import arch_model

    rv = r[valid.astype(bool)].dropna() * 100.0
    gaps = int((~valid.astype(bool)).sum())
    train = rv[rv.index <= pd.Timestamp(train_end)]
    info = {"n_train_returns": int(len(train)), "train_first": str(train.index.min().date()),
            "train_last": str(train.index.max().date()), "invalid_returns_dropped_from_filter": gaps,
            "scale": "percent returns (x100); variance /10000 back to decimal"}
    am = arch_model(train, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = am.fit(disp="off", options={"maxiter": 2000})
        info["fit_warnings"] = [str(x.message) for x in w]
    info.update({"params": {k: float(v) for k, v in res.params.items()},
                 "loglikelihood": float(res.loglikelihood), "convergence_flag": int(res.convergence_flag),
                 "converged": bool(res.convergence_flag == 0)})
    if not info["converged"]:
        # report the failure; do not silently substitute another model
        return pd.Series(np.nan, index=r.index, name="GARCH"), info
    full = arch_model(rv, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    fixed = full.fix(res.params)
    fc = fixed.forecast(horizon=horizon, start=rv.index[0], reindex=False)
    v = fc.variance / 10000.0
    sig = np.sqrt(ann / horizon * v.sum(axis=1))
    out = pd.Series(np.nan, index=r.index, name="GARCH")
    out.loc[sig.index] = sig.to_numpy()
    info["persistence_alpha_plus_beta"] = float(res.params.get("alpha[1]", np.nan) + res.params.get("beta[1]", np.nan))
    return out, info


def all_baselines(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    b = cfg["baselines"]
    ann = cfg["target"]["annualization"]
    r = panel["r_ref"].to_numpy(float)
    valid = panel["ret_valid"].to_numpy(bool)
    hv = hv_forecast(np.nan_to_num(r), valid, b["hv_window"], ann)
    ew, skipped = ewma_forecast(np.nan_to_num(r), valid, b["ewma_lambda"], b["ewma_init_returns"], ann)
    g, ginfo = garch_forecast(panel["r_ref"], panel["ret_valid"], cfg["splits"]["train"]["end"],
                              cfg["target"]["horizon"], ann, b["garch"]["p"], b["garch"]["q"], b["garch"]["dist"])
    out = pd.DataFrame({"HV20": hv, "EWMA": ew, "GARCH": g.to_numpy()}, index=panel.index)
    info = {"ewma_skipped_invalid_positions": skipped, "garch": ginfo}
    return out, info
