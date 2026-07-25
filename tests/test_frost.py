import numpy as np
import pandas as pd
import pytest

from microclimate.analysis import frost


def test_contingency_counts_and_rates():
    predicted = np.array([True, True, False, False, True])
    observed = np.array([True, False, True, False, True])

    result = frost.contingency(predicted, observed)

    assert (result["hits"], result["misses"], result["false_alarms"]) == (2, 1, 1)
    assert result["correct_neg"] == 1
    assert result["pod"] == pytest.approx(2 / 3)
    assert result["far"] == pytest.approx(1 / 3)
    assert result["csi"] == pytest.approx(2 / 4)


def test_contingency_perfect_forecast_has_full_skill():
    observed = np.array([True, False, True, False])
    result = frost.contingency(observed.copy(), observed)

    assert result["pss"] == pytest.approx(1.0)
    assert result["misses"] == 0 and result["false_alarms"] == 0


def test_contingency_always_no_has_zero_skill():
    observed = np.array([True, False, True, False])
    result = frost.contingency(np.zeros(4, dtype=bool), observed)

    assert result["pss"] == pytest.approx(0.0)
    assert result["pod"] == pytest.approx(0.0)


def _hourly(nights: int, offset: float, base: float = 35.0):
    """Hourly frame spanning whole nights, forecast offset from actual."""
    times = pd.date_range("2025-03-01 12:00", periods=nights * 24, freq="1h", tz="UTC")
    hours = times.hour.to_numpy()
    # Coldest just before dawn, warmest mid-afternoon.
    actual = base - 6 * np.cos(2 * np.pi * (hours - 5) / 24)
    return pd.DataFrame(
        {
            "valid_time": times,
            "temp_f_actual": actual,
            "temp_f_forecast": actual + offset,
        }
    )


def test_nightly_minima_groups_evening_with_next_morning():
    data = _hourly(nights=5, offset=0.0)
    nights = frost.nightly_minima(data, "UTC")

    assert len(nights) >= 3
    # The 5am minimum of the constructed curve is base - 6.
    assert nights["actual_min"].iloc[1] == pytest.approx(29.0, abs=0.5)


def test_correction_is_applied_before_taking_the_minimum():
    data = _hourly(nights=5, offset=4.0)  # forecast runs 4 F warm
    corrections = np.full(len(data), 4.0)

    uncorrected = frost.nightly_minima(data, "UTC")
    corrected = frost.nightly_minima(data, "UTC", corrections=corrections)

    assert (uncorrected["raw_min"] - uncorrected["actual_min"]).mean() == pytest.approx(4.0)
    assert (corrected["corrected_min"] - corrected["actual_min"]).mean() == pytest.approx(
        0.0, abs=1e-6
    )


def test_skill_curve_rewards_the_corrected_forecast_when_raw_runs_warm():
    # A forecast that reads 5 F warm will miss frosts; correcting it should
    # catch them without inventing many false alarms.
    data = _hourly(nights=60, offset=5.0, base=36.0)
    nights = frost.nightly_minima(data, "UTC", corrections=np.full(len(data), 5.0))

    curve = frost.skill_curve(nights, warn_levels=(32.0,))
    raw = curve[curve.forecast == "raw"].iloc[0]
    corrected = curve[curve.forecast == "corrected"].iloc[0]

    assert raw["misses"] > 0, "a warm-biased forecast should miss frosts"
    assert corrected["pod"] > raw["pod"]


def test_partial_nights_are_dropped():
    data = _hourly(nights=3, offset=0.0).iloc[:8]  # truncated, no complete night
    assert frost.nightly_minima(data, "UTC").empty
