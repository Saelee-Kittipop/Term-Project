"""Self-financing accounting, costs and execution timing (report appendix C)."""
import numpy as np
import pandas as pd

from volvix.backtest import build_targets, simulate, solve_z, summarize

TICK = ["A", "B"]


def test_first_day_from_cash_z_equals_one_over_one_plus_c():
    w = np.array([0.6, 0.4])
    c = 0.001
    z = solve_z(w, np.zeros(2), c)
    assert np.isclose(z, 1 / (1 + c), atol=1e-11)


def test_z_solves_cost_equation():
    w, u, c = np.array([0.2, 0.8]), np.array([0.7, 0.3]), 0.002
    z = solve_z(w, u, c)
    assert np.isclose(z, 1 - c * np.abs(z * w - u).sum(), atol=1e-11)


def _rets(n=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=idx, columns=TICK)


def test_zero_cost_fixed_weights_equals_daily_rebalanced_portfolio():
    R = _rets(30)
    w = np.array([0.6, 0.4])
    T, _, _ = build_targets(R.index, None, w, w, TICK, static_w=w)
    res = simulate(R, T, 0.0, R.index[0])
    expected = 100 * np.prod(1 + R.iloc[1:].to_numpy() @ w)
    assert np.isclose(res["nav"].iloc[-1], expected)
    assert np.allclose(res["nav_pre_trade"].iloc[1:], (res["nav"].shift(1) * (1 + R @ w)).iloc[1:])


def test_costs_reduce_nav_and_no_external_cash():
    R = _rets(30)
    w = np.array([0.6, 0.4])
    T, _, _ = build_targets(R.index, None, w, w, TICK, static_w=w)
    a = simulate(R, T, 0.0, R.index[0])
    b = simulate(R, T, 10.0, R.index[0])
    assert b["nav"].iloc[-1] < a["nav"].iloc[-1]
    assert np.allclose(b["nav_pre_trade"] - b["nav"], b["cost"])
    assert np.isclose(b["cost"].iloc[0], 100 - 100 / 1.001)            # initial purchase cost charged
    wpost = b[[f"w_post_{t}" for t in TICK]].iloc[:-1]
    assert np.allclose(wpost.sum(axis=1), 1.0) and np.allclose(wpost.to_numpy(), w)


def test_signal_at_t_changes_weights_earning_t_plus_2():
    R = _rets(12, seed=4)
    d = R.index
    wn, wa = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    sig = pd.Series(False, index=d)
    sig.iloc[5] = True                                   # alert after close of d[5]
    T, state, log = build_targets(d, sig, wn, wa, TICK, delay=1)
    assert state.iloc[6] and not state.iloc[5] and not state.iloc[7]   # trade at close d[6]
    res = simulate(R, T, 0.0, d[1])
    # NAV growth on d[7] uses alert weights (asset B); on d[6] still normal weights (asset A)
    g = res["nav_pre_trade"] / res["nav"].shift(1) - 1
    assert np.isclose(g.loc[d[7]], R.loc[d[7], "B"]) and np.isclose(g.loc[d[6]], R.loc[d[6], "A"])
    assert np.isclose(g.loc[d[8]], R.loc[d[8], "A"])


def test_last_day_has_no_trade_and_summary_fields():
    R = _rets(40)
    w = np.array([0.5, 0.5])
    T, _, _ = build_targets(R.index, None, w, w, TICK, static_w=w)
    res = simulate(R, T, 10.0, R.index[0])
    assert not res["traded"].iloc[-1] and res["cost"].iloc[-1] == 0
    s = summarize(res, pd.Series(0.0, index=R.index))
    assert s["return_days_N"] == 39
    assert np.isclose(s["net_CAGR"], (res["nav"].iloc[-1] / 100) ** (252 / 39) - 1)
