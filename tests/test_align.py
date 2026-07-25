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
            # Daily accumulator: 0.05 in falls, then another 0.15 in.
            "rain_daily_in": [0.0] * 4 + [0.05] * 4 + [0.20] * 4,
            "solar_wm2": [100.0] * 12,
            "uv": [1.0] * 12,
        }
    )

    hourly = align.hourly_station(station)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["temp_f"] == pytest.approx(55.0)  # mean
    assert row["gust_mph"] == pytest.approx(25.0)  # peak, not mean
    assert row["rain_in"] == pytest.approx(0.20)  # summed increments
    assert row["n_readings"] == 12


def test_hourly_rain_sums_increments_without_double_counting():
    # Regression: rain used to be read off the trailing-60-minute field with a
    # max, which counted the same rainfall in two adjacent clock hours.
    timestamps = pd.date_range("2024-01-01 00:00", periods=24, freq="5min", tz="UTC")
    daily = [0.0] * 6 + [0.10] * 6 + [0.10] * 6 + [0.30] * 6  # 0.10 then 0.20
    station = pd.DataFrame(
        {
            "ts": timestamps,
            "temp_f": [50.0] * 24,
            "rain_daily_in": daily,
        }
    )

    hourly = align.hourly_station(station)

    assert len(hourly) == 2
    assert hourly["rain_in"].iloc[0] == pytest.approx(0.10)
    assert hourly["rain_in"].iloc[1] == pytest.approx(0.20)
    assert hourly["rain_in"].sum() == pytest.approx(0.30)


def test_midnight_reset_of_the_daily_counter_is_not_counted_as_rain():
    timestamps = pd.date_range("2024-01-01 23:30", periods=12, freq="5min", tz="UTC")
    # Counter climbs to 0.40, then resets to 0 at midnight and climbs again.
    daily = [0.35, 0.40, 0.40, 0.40, 0.40, 0.40, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05]
    station = pd.DataFrame(
        {"ts": timestamps, "temp_f": [40.0] * 12, "rain_daily_in": daily}
    )

    hourly = align.hourly_station(station)

    # The reset must not read as negative rain, nor the rebound as a downpour.
    assert (hourly["rain_in"] >= 0).all()
    assert hourly["rain_in"].sum() == pytest.approx(0.10, abs=1e-9)


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
