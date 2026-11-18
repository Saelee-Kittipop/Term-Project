"""Baselines (HV20, EWMA, GARCH aggregation) and evaluation formulas."""
import numpy as np
import pandas as pd
import pytest

from volvix.baselines import ewma_forecast, garch_forecast, hv_forecast
from volvix.evaluation import (alert_metrics, christoffersen, diebold_mariano, forecast_metrics, kupiec_pof, prf,
                               select_cutoff, tau_from_train)


def test_hv20_is_trailing_sample_sd_to_t():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 100)
    hv = hv_forecast(r, np.ones(100, bool), 20)
    assert np.isnan(hv[18]) and np.isclose(hv[19], np.std(r[:20], ddof=1) * np.sqrt(252))
    assert np.isclose(hv[50], np.std(r[31:51], ddof=1) * np.sqrt(252))


def test_ewma_initialisation_and_recursion():
    rng = np.random.default_rng(2)
    r = np.concatenate([[np.nan], rng.normal(0, 0.01, 99)])  # position 0 has no return
    valid = np.isfinite(r)
    f, skipped = ewma_forecast(np.nan_to_num(r), valid, 0.94, 60, 252)
    v60 = np.mean(r[1:61] ** 2)                          # returns 1..60 at origin of return 60 (pos 60)
    assert np.isnan(f[59]) and np.isclose(f[60], np.sqrt(252 * v60))
    v61 = 0.94 * v60 + 0.06 * r[61] ** 2
    assert np.isclose(f[61], np.sqrt(252 * v61)) and skipped == []


def test_garch_five_day_aggregation_matches_manual_recursion():
    pytest.importorskip("arch")
    from arch import arch_model
    sim = arch_model(None, mean="Constant", vol="GARCH", p=1, q=1)
    data = sim.simulate([0.03, 0.02, 0.08, 0.9], 1500, burn=200)["data"] / 100.0
    idx = pd.bdate_range("2010-01-01", periods=1500)
    r = pd.Series(data.to_numpy(), index=idx)
    valid = pd.Series(True, index=idx)
    f, info = garch_forecast(r, valid, str(idx[999].date()))
    assert info["converged"] and info["n_train_returns"] == 1000
    p = info["params"]
    mu, om, a, b = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
    am = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fix([mu, om, a, b])
    t = 1200
    h1 = am.conditional_volatility.iloc[t] ** 2
    e = r.iloc[t] * 100 - mu
    v = [om + a * e ** 2 + b * h1]                      # one-step-ahead variance (percent^2)
    for _ in range(4):
        v.append(om + (a + b) * v[-1])
    manual = np.sqrt(252 / 5 * np.sum(v) / 10000)       # average the 5 daily variances, then root
    assert np.isclose(f.iloc[t], manual, rtol=1e-8)
    assert not np.isclose(f.iloc[t], np.sqrt(252 * v[-1] / 10000), rtol=1e-6)   # not the day-5 vol only


def test_forecast_metric_definitions():
    y = np.array([0.10, 0.20, 0.30])
    f = np.array([0.12, 0.18, 0.33])
    m = forecast_metrics(y, f)
    assert np.isclose(m["MAE_pp"], np.mean(np.abs(f - y)) * 100)
    assert np.isclose(m["RMSE_pp"], np.sqrt(np.mean((f - y) ** 2)) * 100)
    assert np.isclose(m["log_RMSE"], np.sqrt(np.mean((np.log(f) - np.log(y)) ** 2)))
    assert np.isclose(m["QLIKE"], np.mean(np.log(f ** 2) + y ** 2 / f ** 2))   # on variance, not volatility


def test_cutoff_selection_rules():
    f = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    h = np.array([0, 0, 0, 1, 1], bool)
    c = select_cutoff(f, h, tau=0.35)
    assert np.isclose(c["cutoff"], 0.3) and np.isclose(c["val_F1"], 1.0)   # alert iff forecast > c
    # ties -> highest cutoff: two cutoffs give the same F1
    f2 = np.array([0.1, 0.2, 0.3, 0.4])
    h2 = np.array([1, 0, 1, 0], bool)
    c2 = select_cutoff(f2, h2, tau=0.0)
    f1s = {cc: prf(h2, f2 > cc)["F1"] for cc in [np.nextafter(0.1, -1), 0.1, 0.2, 0.3, 0.4]}
    best = max(f1s.values())
    assert np.isclose(c2["cutoff"], max(k for k, v in f1s.items() if np.isclose(v, best)))
    # alert-on-all candidate exists (value below the minimum)
    c3 = select_cutoff(np.array([0.1, 0.2]), np.array([1, 1], bool), 0.0)
    assert c3["cutoff"] < 0.1
    # no validation positive -> fallback to tau
    c4 = select_cutoff(f, np.zeros(5, bool), tau=0.42)
    assert c4["cutoff"] == 0.42 and c4["rule"].startswith("fallback")


def test_alert_conventions_no_alert_and_no_positive():
    m = prf(np.array([1, 0, 1], bool), np.array([0, 0, 0], bool))
    assert m["precision"] == 0 and m["F1"] == 0 and m["recall"] == 0
    m2 = alert_metrics(np.array([0.1, 0.1]), np.array([0.2, 0.05]), tau=0.5, cutoff=0.15)
    assert np.isnan(m2["recall"]) and np.isnan(m2["AP"])              # not evaluable, not zero


def test_tau_is_training_p90():
    y = np.arange(1, 101, dtype=float)
    assert np.isclose(tau_from_train(y, 0.9), np.quantile(y, 0.9))


def test_kupiec_and_christoffersen():
    v = np.zeros(1000, bool)
    v[::20] = True                                      # exactly 5% violations
    k = kupiec_pof(v, 0.05)
    assert k["violations"] == 50 and k["LR_pof"] < 1e-8 and k["p_value"] > 0.99
    c = christoffersen(v, 0.05)
    assert c["n11"] == 0 and np.isfinite(c["LR_cc"])


def test_dm_sign():
    rng = np.random.default_rng(3)
    l2 = rng.chisquare(1, 800)
    l1 = l2 * 0.5
    d = diebold_mariano(l1, l2, h=5)
    assert d["dm_stat"] < 0 and d["p_value"] < 0.01
