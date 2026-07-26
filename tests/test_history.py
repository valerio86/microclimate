import numpy as np
import pandas as pd
import pytest

from microclimate import history


def paired(days: int = 90, bias: float = 4.0, seed: int = 0):
    """Hourly paired data where the forecast runs `bias` degrees warm at night."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
    hours = times.hour.to_numpy()
    actual = 45 - 8 * np.cos(2 * np.pi * (hours - 5) / 24) + rng.normal(0, 1, len(times))
    night = ((hours < 7) | (hours >= 20)).astype(float)
    error = bias * night

    frame = pd.DataFrame(
        {
            "valid_time": times,
            "lead_days": 1,
            "temp_f_actual": actual,
            "temp_f_forecast": actual + error,
            "temp_f_error": error,
            "local_hour": hours,
            "local_doy": times.dayofyear,
            "local_month": times.month,
            "cloud_cover": rng.uniform(0, 100, len(times)),
            "wind_mph_forecast": rng.uniform(0, 12, len(times)),
            "solar_wm2_forecast": np.clip(600 * np.sin(np.pi * (hours - 6) / 12), 0, None),
            "rain_in": 0.0,
            "precip_in": 0.0,
            "showers_in": 0.0,
            "snowfall_in": 0.0,
            "rh_forecast": 70.0,
            "vpd_kpa": 0.5,
        }
    )
    return {"primary": frame}


def test_verification_returns_only_the_window():
    out = history.verification(paired(90), "UTC", "primary", window_days=20)
    nights = out["nights"]
    assert not nights.empty
    span = pd.to_datetime(nights["night"] if "night" in nights else nights["day"])
    assert (span.max() - span.min()).days <= 21


def test_correction_beats_the_raw_forecast_on_a_learnable_bias():
    out = history.verification(paired(120, bias=5.0), "UTC", "primary", window_days=30)
    summary = out["summary"]
    assert summary["corrected_low_mae"] < summary["raw_low_mae"]
    assert summary["low_improved"] is True


def test_no_improvement_claimed_when_there_is_no_bias():
    out = history.verification(paired(120, bias=0.0), "UTC", "primary", window_days=30)
    summary = out["summary"]
    # Within noise; the point is that it must not invent an improvement.
    assert summary["corrected_low_mae"] == pytest.approx(summary["raw_low_mae"], abs=0.6)


def test_predictions_are_out_of_sample():
    """The window must not be part of what the corrector was fitted on.

    Tampering with the training period has to change the window's predictions;
    if it does not, the model is being fitted to the window itself.
    """
    base = paired(120, bias=4.0)
    tampered = {"primary": base["primary"].copy()}
    frame = tampered["primary"]
    cutoff = frame["valid_time"].max() - pd.Timedelta(days=30)
    early = frame["valid_time"] < cutoff
    frame.loc[early, "temp_f_error"] = -6.0  # opposite bias in training only

    a = history.verification(base, "UTC", "primary", window_days=30)["nights"]
    b = history.verification(tampered, "UTC", "primary", window_days=30)["nights"]

    assert not np.allclose(a["corrected_min"], b["corrected_min"]), (
        "training data must influence the window's predictions"
    )
    assert np.allclose(a["actual_min"], b["actual_min"]), "actuals must be untouched"


def test_frost_counts_are_consistent():
    out = history.verification(paired(120, bias=4.0), "UTC", "primary", window_days=30)
    s = out["summary"]
    assert s["frost_caught"] + s["frost_missed"] == s["frost_nights"]


def test_short_history_degrades_gracefully():
    out = history.verification(paired(5), "UTC", "primary", window_days=30)
    assert out["nights"].empty
    assert out["summary"] == {}


def test_missing_source_is_an_error_not_a_silent_empty():
    with pytest.raises(KeyError):
        history.verification(paired(60), "UTC", "absent", window_days=10)
