import numpy as np
import pandas as pd
import pytest

from microclimate import align


def test_circular_mean_wraps_around_north():
    # The whole reason this helper exists: a plain mean would give 180°.
    result = align.circular_mean_deg(pd.Series([350.0, 10.0]))
    assert result == pytest.approx(0.0, abs=0.01) or result == pytest.approx(360.0, abs=0.01)


def test_circular_mean_ignores_missing():
    assert align.circular_mean_deg(pd.Series([90.0, np.nan, 90.0])) == pytest.approx(90.0)


def test_circular_mean_empty_is_nan():
    assert np.isnan(align.circular_mean_deg(pd.Series([], dtype=float)))


def test_hourly_station_aggregates_by_physical_meaning():
    timestamps = pd.date_range("2024-01-01 00:00", periods=12, freq="5min", tz="UTC")
    station = pd.DataFrame(
        {
            "ts": timestamps,
            "temp_f": [50.0] * 6 + [60.0] * 6,
            "dewpoint_f": [40.0] * 12,
            "rh": [80.0] * 12,
            "wind_mph": [5.0] * 12,
            "gust_mph": [10.0] * 11 + [25.0],
            "wind_dir_deg": [0.0] * 12,
            "pressure_hpa": [1013.0] * 12,
            "rain_hourly_in": [0.0] * 11 + [0.2],
            "solar_wm2": [100.0] * 12,
            "uv": [1.0] * 12,
        }
    )

    hourly = align.hourly_station(station)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["temp_f"] == pytest.approx(55.0)  # mean
    assert row["gust_mph"] == pytest.approx(25.0)  # peak, not mean
    assert row["rain_hourly_in"] == pytest.approx(0.2)  # trailing total
    assert row["n_readings"] == 12


def test_pair_computes_forecast_minus_actual():
    valid_time = pd.to_datetime(["2024-01-01 00:00"], utc=True)
    actual = pd.DataFrame({"valid_time": valid_time, "temp_f": [50.0]})
    forecast = pd.DataFrame(
        {"valid_time": valid_time, "temp_f": [53.0], "lead_days": [1]}
    )

    paired = align.pair(actual, forecast)

    assert paired["temp_f_error"].iloc[0] == pytest.approx(3.0)
    assert paired["temp_f_forecast"].iloc[0] == pytest.approx(53.0)
    assert paired["temp_f_actual"].iloc[0] == pytest.approx(50.0)


def test_pair_with_no_overlap_is_empty():
    actual = pd.DataFrame(
        {"valid_time": pd.to_datetime(["2024-01-01 00:00"], utc=True), "temp_f": [50.0]}
    )
    forecast = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(["2024-06-01 00:00"], utc=True),
            "temp_f": [53.0],
            "lead_days": [1],
        }
    )
    assert align.pair(actual, forecast).empty
