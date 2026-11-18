"""Target definition, alignment, purge and missing-session handling."""
import numpy as np
import pandas as pd

from volvix.data.dataset import TrainOnlyScaler, alignment_checks, feature_matrix, forward_target, make_windows


def test_annualization_changes_units_not_horizon():
    # daily SD of 1% over 5 future days -> 15.87% annualized
    r = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.0])
    r[1:] = r[1:] - r[1:].mean()
    r[1:] = r[1:] / np.std(r[1:], ddof=1) * 0.01
    y = forward_target(r, np.ones(6, bool), 5, 1, 252)
    assert np.isclose(y[0], 0.01 * np.sqrt(252), rtol=1e-12)
    assert np.isclose(y[0], 0.1587, atol=1e-4)
    assert np.isnan(y[1:]).all()                      # no complete forward window


def test_target_uses_t_plus_1_to_t_plus_5_with_ddof1():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 50)
    y = forward_target(r, np.ones(50, bool))
    t = 10
    assert np.isclose(y[t], np.std(r[t + 1:t + 6], ddof=1) * np.sqrt(252))
    # pandas equivalent from report appendix B: rolling(5).std().shift(-5); shift(-1) would include the past
    alt = pd.Series(r).rolling(5).std(ddof=1).shift(-5) * np.sqrt(252)
    assert np.allclose(y[:-5], alt.to_numpy()[:-5])
    wrong = pd.Series(r).rolling(5).std(ddof=1).shift(-1) * np.sqrt(252)
    assert not np.isclose(y[t], wrong.iloc[t])


def test_target_invalid_when_window_contains_invalid_return():
    r = np.full(20, 0.01)
    valid = np.ones(20, bool)
    valid[8] = False
    y = forward_target(r, valid)
    assert np.isnan(y[3:8]).all() and np.isfinite(y[2]) and np.isfinite(y[8])


def test_alignment_checks_pass_on_synthetic(synth, cfg):
    chk = alignment_checks(synth["samples"], synth["panel"], cfg)
    assert chk["passed"].all(), chk[~chk["passed"]]


def test_purge_drops_boundary_crossing_targets(synth, cfg):
    s = synth["samples"]
    for split in ("train", "validation", "test"):
        hi = pd.Timestamp(cfg["splits"][split]["end"])
        part = s[s["split"] == split]
        assert (part["target_end_date"] <= hi).all()
        # exactly the last 5 origins of each period are purged (no other gaps near boundaries in synthetic data)
        per = s[(s["period"] == split)]
        assert per.tail(5)["purged_boundary"].all()
        assert per.tail(5)["split"].isna().all()


def test_first_origin_uses_first_60_returns(synth, cfg):
    s, panel = synth["samples"], synth["panel"]
    first = s[s["split"] == "train"].iloc[0]
    assert first["pos"] == 60                         # positions 1..60 = the first 60 returns
    assert first["input_start_date"] == panel.index[1]


def test_missing_session_never_produces_multiday_return(synth):
    panel = synth["panel"]
    # synthetic XLV is missing one session: its return that day and the next must be NaN (not a 2-day return)
    miss = panel.index[panel["price_XLV"].isna()]
    assert len(miss) == 1
    i = panel.index.get_loc(miss[0])
    assert np.isnan(panel["ret_XLV"].iloc[i]) and np.isnan(panel["ret_XLV"].iloc[i + 1])
    assert not panel["ret_valid"].iloc[i] and not panel["ret_valid"].iloc[i + 1]
    # samples whose input or target touches those days are excluded
    s = synth["samples"]
    bad = s[s["score_eligible"] & (s["input_start_date"] <= panel.index[i + 1]) & (s["origin_date"] >= panel.index[i])]
    assert len(bad) == 0


def test_no_fill_and_duplicates_logged(synth):
    excl = pd.read_csv(synth["processed"] / "exclusion_log.csv")
    assert ((excl["item"] == "GLD") & (excl["stage"] == "duplicates")).sum() == 1
    assert synth["quality"]["missing_sessions_by_ticker"]["XLV"] == 1
    assert synth["quality"]["dividend_double_count_suspects"] == 0


def test_vix_is_lagged_one_session(synth):
    p = synth["panel"]
    assert np.allclose(p["vix_lag1"].to_numpy()[1:], p["vix"].to_numpy()[:-1])


def test_input_window_ends_at_origin_and_contains_no_future(synth, cfg):
    s, panel = synth["samples"], synth["panel"]
    feat, _ = feature_matrix(panel, cfg, "RV")
    sc = s[s["split"] == "validation"].iloc[:3]
    X = make_windows(feat, sc["pos"].to_numpy(), 60)
    for k, p in enumerate(sc["pos"]):
        assert np.allclose(X[k, -1], feat[p])            # last row = origin t
        assert np.allclose(X[k, 0], feat[p - 59])        # first row = t-59


def test_scaler_fit_only_on_training_rows(synth, cfg):
    s, panel = synth["samples"], synth["panel"]
    train_pos = s.loc[s["split"] == "train", "pos"].to_numpy()
    rows = TrainOnlyScaler.training_rows(train_pos, 60)
    assert rows.max() == train_pos.max()                 # input rows stop at the last training origin
    assert panel.index[rows.max()] <= pd.Timestamp(cfg["splits"]["train"]["end"])
    feat, cols = feature_matrix(panel, cfg, "RV")
    a = TrainOnlyScaler().fit(feat, rows, cols)
    feat2 = feat.copy()
    feat2[rows.max() + 1:] *= 50.0                       # change everything after training rows
    b = TrainOnlyScaler().fit(feat2, rows, cols)
    assert np.allclose(a.mean_, b.mean_) and np.allclose(a.scale_, b.scale_)
    assert len(np.unique(rows)) == len(rows)            # unique daily rows, not window-weighted
