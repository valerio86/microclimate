"""Align 5-minute station readings with hourly forecasts.

This is where the training set gets made: one row per (valid_time, lead_days)
holding what was forecast and what actually happened, side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns present in both the station and forecast tables — the ones we can
# actually score a forecast against.
COMPARABLE = [
    "temp_f",
    "dewpoint_f",
    "rh",
    "wind_mph",
    "gust_mph",
    "pressure_hpa",
    "solar_wm2",
]


def circular_mean_deg(values: pd.Series) -> float:
    """Mean of compass bearings.

    A plain mean of 350° and 10° gives 180° — the opposite direction. Averaging
    the unit vectors instead gives 0°, which is the answer we want.
    """
    clean = values.dropna()
    if clean.empty:
        return np.nan
    radians = np.deg2rad(clean.to_numpy(dtype=float))
    angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(angle) % 360)


def hourly_station(station: pd.DataFrame) -> pd.DataFrame:
    """Downsample 5-minute station readings to hourly.

    Each variable gets the aggregation that matches its physical meaning:
    gusts are peaks, not averages; directions are circular; the rest are means.
    """
    if station.empty:
        return station

    frame = station.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    grouped = frame.set_index("ts").resample("1h")

    out = grouped.agg(
        {
            "temp_f": "mean",
            "dewpoint_f": "mean",
            "rh": "mean",
            "wind_mph": "mean",
            "gust_mph": "max",
            "pressure_hpa": "mean",
            "solar_wm2": "mean",
            "uv": "mean",
            # `hourlyrainin` is already a trailing-60-minute total, so the
            # largest value seen in the hour approximates that hour's rain.
            "rain_hourly_in": "max",
        }
    )
    out["wind_dir_deg"] = grouped["wind_dir_deg"].apply(circular_mean_deg)
    out["n_readings"] = grouped["temp_f"].count()

    return out.reset_index().rename(columns={"ts": "valid_time"})


def pair(station_hourly: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Join hourly actuals to forecasts, adding one error column per variable.

    Sign convention: error = forecast - actual. Positive means the public
    forecast runs high at your location.
    """
    if station_hourly.empty or forecasts.empty:
        return pd.DataFrame()

    actual = station_hourly.rename(
        columns={column: f"{column}_actual" for column in COMPARABLE}
    )
    predicted = forecasts.rename(
        columns={column: f"{column}_forecast" for column in COMPARABLE}
    )

    merged = predicted.merge(actual, on="valid_time", how="inner")

    for column in COMPARABLE:
        forecast_column = f"{column}_forecast"
        actual_column = f"{column}_actual"
        if forecast_column in merged and actual_column in merged:
            merged[f"{column}_error"] = merged[forecast_column] - merged[actual_column]

    return merged


def add_time_features(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Add local-time columns — biases are usually strongest by hour and season."""
    if frame.empty:
        return frame

    out = frame.copy()
    local = pd.to_datetime(out["valid_time"], utc=True).dt.tz_convert(timezone)
    out["local_hour"] = local.dt.hour
    out["local_month"] = local.dt.month
    out["local_doy"] = local.dt.dayofyear
    out["local_date"] = local.dt.date
    return out
