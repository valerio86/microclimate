"""Open-Meteo client — the public forecast baseline we correct against.

Two endpoints matter here, and the distinction is the whole point of the
project:

  historical-forecast-api   What the model's own archive says for a past date.
                            Effectively the freshest run (lead time ~0-1 day).

  previous-runs-api         What the model predicted N days *before* the fact,
                            via `<variable>_previous_day{N}` for N in 1..7.

The second is what makes bias correction trainable without waiting: for any
past hour we can recover the forecast issued 1-7 days ahead of it and pair it
with what the station actually measured. Both accept historical date ranges
back to 2022, and neither needs an API key.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Iterator, Sequence

import httpx
import pandas as pd

from ..config import Location

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Open-Meteo variable -> our canonical column name.
VARIABLES = {
    "temperature_2m": "temp_f",
    "dew_point_2m": "dewpoint_f",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_mph",
    "wind_gusts_10m": "gust_mph",
    "wind_direction_10m": "wind_dir_deg",
    # Sea-level-adjusted, to match the station's `baromrelin`. Using
    # `surface_pressure` instead compares against pressure at 534 m and
    # produces a flat ~56 hPa offset that looks like a forecast bias but is
    # purely a reference mismatch.
    "pressure_msl": "pressure_hpa",
    "precipitation": "precip_in",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "solar_wm2",
}

# Match the station's native units so comparisons need no conversion.
UNIT_PARAMS = {
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "UTC",
}

MAX_LEAD_DAYS = 7
CHUNK_DAYS = 90  # Keep responses and URLs a manageable size.


class OpenMeteoClient:
    def __init__(self, location: Location, timeout: float = 60.0):
        self._location = location
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _base_params(self) -> dict:
        params = {
            "latitude": self._location.latitude,
            "longitude": self._location.longitude,
            **UNIT_PARAMS,
        }
        # Passing elevation stops Open-Meteo from downscaling to the grid
        # cell's mean height, which is a large temperature bias in hilly terrain.
        if self._location.elevation_m is not None:
            params["elevation"] = self._location.elevation_m
        return params

    def _get(self, url: str, params: dict, max_retries: int = 4) -> dict:
        for attempt in range(max_retries):
            response = self._client.get(url, params=params)
            if response.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
            return payload
        raise RuntimeError(f"Open-Meteo still rate-limiting after {max_retries} tries")

    def fetch_analysis(self, start: date, end: date) -> pd.DataFrame:
        """Archived best-match forecast for a past range (lead_days = 0)."""
        payload = self._get(
            HISTORICAL_FORECAST_URL,
            {
                **self._base_params(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": ",".join(VARIABLES),
            },
        )
        frame = _hourly_to_frame(payload, {v: c for v, c in VARIABLES.items()})
        frame["lead_days"] = 0
        return frame

    def fetch_lead(self, start: date, end: date, lead_days: int) -> pd.DataFrame:
        """Forecast issued `lead_days` before each valid hour (1..7)."""
        if not 1 <= lead_days <= MAX_LEAD_DAYS:
            raise ValueError(f"lead_days must be 1..{MAX_LEAD_DAYS}, got {lead_days}")

        suffixed = {
            f"{variable}_previous_day{lead_days}": column
            for variable, column in VARIABLES.items()
        }
        payload = self._get(
            PREVIOUS_RUNS_URL,
            {
                **self._base_params(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": ",".join(suffixed),
            },
        )
        frame = _hourly_to_frame(payload, suffixed)
        frame["lead_days"] = lead_days
        return frame

    def fetch_range(
        self,
        start: date,
        end: date,
        leads: Sequence[int] = (0, 1, 2, 3, 5, 7),
    ) -> Iterator[pd.DataFrame]:
        """Yield forecast frames chunk by chunk for the whole range and leads.

        Yields rather than accumulating so a multi-year pull can be written to
        the database incrementally and resumed after an interruption.
        """
        for chunk_start, chunk_end in _chunks(start, end, CHUNK_DAYS):
            for lead in leads:
                if lead == 0:
                    frame = self.fetch_analysis(chunk_start, chunk_end)
                else:
                    frame = self.fetch_lead(chunk_start, chunk_end, lead)
                if not frame.empty:
                    frame["source"] = "open-meteo"
                    yield frame


def _hourly_to_frame(payload: dict, name_map: dict[str, str]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        return pd.DataFrame()

    frame = pd.DataFrame({"valid_time": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, column in name_map.items():
        frame[column] = pd.to_numeric(hourly.get(api_name), errors="coerce")
    return frame


def _chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)
