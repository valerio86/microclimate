import numpy as np
import pandas as pd
import pytest

from microclimate import features


def paired(hours: int = 24 * 40, lead: int = 1, start: str = "2025-01-01"):
    times = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    actual = 50 + 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
    return pd.DataFrame(
        {
            "valid_time": times,
            "lead_days": lead,
            "temp_f_actual": actual,
            "temp_f_forecast": actual + 1.0,
            "temp_f_error": np.full(hours, 1.0),
        }
    )


def test_features_cannot_see_the_future():
    """The test that matters: tamper with everything after a cutoff and prove
    that features for forecasts issued before it do not move."""
    base = paired()
    cutoff = base["valid_time"].iloc[len(base) // 2]

    tampered = base.copy()
    future = tampered["valid_time"] > cutoff
    tampered.loc[future, "temp_f_actual"] += 100.0
    tampered.loc[future, "temp_f_error"] -= 100.0

    a = features.add_lagged_observations(base)
    b = features.add_lagged_observations(tampered)
    columns = features.feature_columns(a)

    unaffected = a["issue_time"] <= cutoff
    assert unaffected.sum() > 100, "need a meaningful number of rows to check"

    pd.testing.assert_frame_equal(
        a.loc[unaffected, columns].reset_index(drop=True),
        b.loc[unaffected, columns].reset_index(drop=True),
    )


def test_tampering_does_change_later_features():
    """Guards the guard: if nothing changed anywhere, the test above is vacuous."""
    base = paired()
    cutoff = base["valid_time"].iloc[len(base) // 2]
    tampered = base.copy()
    tampered.loc[tampered["valid_time"] > cutoff, "temp_f_actual"] += 100.0

    a = features.add_lagged_observations(base)
    b = features.add_lagged_observations(tampered)

    later = a["issue_time"] > cutoff + pd.Timedelta(hours=30)
    assert not np.allclose(
        a.loc[later, "obs_latest"].to_numpy(),
        b.loc[later, "obs_latest"].to_numpy(),
        equal_nan=True,
    )


def test_issue_time_respects_the_lead():
    frame = paired(hours=200, lead=3)
    out = features.add_lagged_observations(frame)
    gap = out["valid_time"] - out["issue_time"]
    assert (gap == pd.Timedelta(days=3)).all()


def test_lead_zero_is_treated_as_a_full_day_of_separation():
    # Lead 0 is the analysis, not a forecast; giving it observations from the
    # hour it describes would be leakage dressed up as a feature.
    frame = paired(hours=200, lead=0)
    out = features.add_lagged_observations(frame)
    assert (out["valid_time"] - out["issue_time"] == pd.Timedelta(days=1)).all()


def test_persistence_is_the_observation_one_lead_earlier():
    frame = paired(hours=24 * 10, lead=2)
    out = features.add_lagged_observations(frame)

    observed = frame.set_index("valid_time")["temp_f_actual"]
    expected = (out["valid_time"] - pd.Timedelta(days=2)).map(observed)
    valid = out["persistence"].notna()
    assert valid.sum() > 50
    assert np.allclose(out.loc[valid, "persistence"], expected[valid])


def test_latest_observation_equals_the_value_at_issue_time():
    # With complete hourly data the most recent observation at the cutoff is
    # exactly the observation at the cutoff.
    frame = paired(hours=24 * 10, lead=1)
    out = features.add_lagged_observations(frame)
    observed = frame.set_index("valid_time")["temp_f_actual"]
    expected = out["issue_time"].map(observed)
    valid = out["obs_latest"].notna() & expected.notna()
    assert np.allclose(out.loc[valid, "obs_latest"], expected[valid])


def test_diurnal_range_matches_the_constructed_signal():
    frame = paired(hours=24 * 10, lead=1)
    out = features.add_lagged_observations(frame)
    settled = out["obs_range_24h"].dropna()
    # The synthetic series swings +/-10 F, so a full day spans about 20 F.
    assert settled.iloc[-1] == pytest.approx(20.0, abs=0.5)


def test_recent_error_carries_the_previous_days_bias():
    frame = paired(hours=24 * 10, lead=1)
    frame.loc[frame["valid_time"].dt.day <= 5, "temp_f_error"] = 3.0
    out = features.add_lagged_observations(frame)
    early = out[(out["valid_time"].dt.day == 5) & out["recent_error"].notna()]
    assert (early["recent_error"] == 3.0).all()


def test_gaps_do_not_pull_values_across_them_incorrectly():
    frame = paired(hours=24 * 20, lead=1)
    # Remove two days of observations; features may be stale but must not be
    # invented, and must never come from after the issue time.
    hole = (frame["valid_time"] >= "2025-01-05") & (frame["valid_time"] < "2025-01-07")
    frame.loc[hole, "temp_f_actual"] = np.nan

    out = features.add_lagged_observations(frame)
    assert out["obs_latest"].notna().any()
    features.assert_no_lookahead(out)  # must not raise


def test_assert_no_lookahead_catches_an_injected_leak():
    frame = paired(hours=24 * 5, lead=1)
    out = features.add_lagged_observations(frame)
    # Fabricate a feature value for a forecast issued before any observation.
    out.loc[out.index[0], "issue_time"] = pd.Timestamp("2000-01-01", tz="UTC")
    out.loc[out.index[0], "obs_latest"] = 42.0

    with pytest.raises(AssertionError, match="lookahead"):
        features.assert_no_lookahead(out)


def test_empty_input_passes_through():
    assert features.add_lagged_observations(pd.DataFrame()).empty
