"""Self-financing daily-rebalanced backtest with execution delay and proportional costs (report appendix C).

Timeline: signal after close t -> trade at close t+1 -> new weights earn the close-to-close return of t+2.
Each day: holdings first earn that day's return, then trade at the close. Costs are paid from NAV by
solving z = 1 - c * sum_i |z*w_i - u_i| (bisection on [0, 1]); NAV_after = z * NAV_before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def solve_z(w: np.ndarray, u: np.ndarray, c: float, tol: float = 1e-12) -> float:
    if c == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    f = lambda z: z - 1.0 + c * np.abs(z * w - u).sum()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def simulate(returns: pd.DataFrame, target: pd.DataFrame, cost_bps: float, start_date, initial_nav: float = 100.0,
             tol: float = 1e-12) -> pd.DataFrame:
    """returns: daily simple returns (sessions x assets). target: weights to trade INTO at the close of each
    session (NaN row = no trade). Invests from cash at the close of start_date; ends at the last session
    without a final trade."""
    c = cost_bps / 1e4
    dates = returns.loc[returns.index >= pd.Timestamp(start_date)].index
    R = returns.loc[dates].to_numpy(float)
    W = target.reindex(dates).to_numpy(float)
    h = np.zeros(R.shape[1])
    rows = []
    for k, d in enumerate(dates):
        if k == 0:
            v_pre = initial_nav
        else:
            if not np.isfinite(R[k]).all():
                raise ValueError(f"missing return on trading date {d.date()} - fix data before backtest")
            h = h * (1 + R[k])
            v_pre = float(h.sum())
        u = h / v_pre
        last = k == len(dates) - 1
        if not last and np.isfinite(W[k]).all():
            w = W[k]
            z = solve_z(w, u, c, tol)
            traded = float(np.abs(z * w - u).sum())        # gross traded notional / NAV before trading
            v_post = z * v_pre
            h = w * v_post
            traded_flag = True
        else:
            z, traded, v_post, traded_flag = 1.0, 0.0, v_pre, False
        rows.append({"date": d, "nav_pre_trade": v_pre, "nav": v_post, "cost": v_pre - v_post,
                     "turnover_frac": traded, "traded": traded_flag, "cash_start": k == 0,
                     **{f"w_post_{a}": h[i] / v_post for i, a in enumerate(returns.columns)},
                     **{f"u_pre_{a}": u[i] for i, a in enumerate(returns.columns)}})
    out = pd.DataFrame(rows).set_index("date")
    out["net_ret"] = out["nav"].pct_change()
    out.iloc[0, out.columns.get_loc("net_ret")] = np.nan
    return out


def drawdown_stats(nav: pd.Series) -> dict:
    peak = nav.cummax()
    dd = nav / peak - 1
    trough = dd.idxmin()
    mdd = float(-dd.min())
    peak_date = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    rec = after[after >= nav.loc[peak_date]]
    if mdd == 0:
        return {"max_drawdown": 0.0, "dd_peak": None, "dd_trough": None, "dd_recovery": None, "recovery_sessions": 0}
    if len(rec):
        rd = rec.index[0]
        sess = int(nav.index.get_loc(rd) - nav.index.get_loc(peak_date))
        return {"max_drawdown": mdd, "dd_peak": str(peak_date.date()), "dd_trough": str(trough.date()),
                "dd_recovery": str(rd.date()), "recovery_sessions": sess}
    return {"max_drawdown": mdd, "dd_peak": str(peak_date.date()), "dd_trough": str(trough.date()),
            "dd_recovery": "not_recovered_within_test", "recovery_sessions": None}


def summarize(bt: pd.DataFrame, rf: pd.Series, initial_nav: float = 100.0, ann: int = 252) -> dict:
    r = bt["net_ret"].iloc[1:]
    N = len(r)
    cagr = (bt["nav"].iloc[-1] / initial_nav) ** (ann / N) - 1
    vol = r.std(ddof=1) * np.sqrt(ann)
    rfa = rf.reindex(r.index)
    ex = (r - rfa).dropna()
    sharpe = np.sqrt(ann) * ex.mean() / ex.std(ddof=1) if len(ex) > 1 else np.nan
    wcols = [c for c in bt.columns if c.startswith("w_post_")]
    out = {"start": str(bt.index[0].date()), "end": str(bt.index[-1].date()), "return_days_N": N,
           "final_nav": float(bt["nav"].iloc[-1]), "net_CAGR": float(cagr), "ann_vol": float(vol),
           "sharpe": float(sharpe), "sharpe_days_used": int(len(ex)), "sharpe_days_missing_rf": int(N - len(ex)),
           "turnover_per_year": float(bt["turnover_frac"].sum() * ann / N),
           "cumulative_cost": float(bt["cost"].sum()),
           "cumulative_cost_pct_initial": float(bt["cost"].sum() / initial_nav * 100),
           **drawdown_stats(bt["nav"])}
    for c in wcols:
        out[f"avg_{c.replace('w_post_', 'w_')}"] = float(bt[c].iloc[:-1].mean())
    return out


def build_targets(dates: pd.DatetimeIndex, signal_by_origin: pd.Series | None, w_normal: np.ndarray,
                  w_alert: np.ndarray, columns, static_w: np.ndarray | None = None, delay: int = 1):
    """Target weights for the trade at the close of each session. Signal from origin t drives the trade at
    session t+delay. Missing signal -> keep the previous target (logged)."""
    T = pd.DataFrame(np.nan, index=dates, columns=columns)
    log = []
    if static_w is not None:
        T.loc[:, :] = static_w
        return T, pd.Series(False, index=dates), log
    state = pd.Series(False, index=dates)
    prev = None
    for k in range(delay, len(dates)):
        o = dates[k - delay]
        sig = signal_by_origin.get(o, np.nan) if signal_by_origin is not None else np.nan
        if pd.isna(sig):
            log.append({"trade_date": dates[k], "origin": o, "issue": "missing_signal_hold_previous"})
            if prev is None:
                continue
            alert = prev
        else:
            alert = bool(sig)
        T.iloc[k] = w_alert if alert else w_normal
        state.iloc[k] = alert
        prev = alert
    return T, state, log
