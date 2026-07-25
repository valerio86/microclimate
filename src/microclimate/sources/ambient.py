"""Ambient Weather Network API client.

The WS-2902 has no local API — it uploads to Ambient's cloud, and history is
read back from there. The device history endpoint returns at most 288 records
(one day at the station's 5-minute cadence) ending at a given timestamp, so a
multi-year backfill is a backward walk: fetch a day, step the cursor to just
before the oldest record returned, repeat.

Ambient rate-limits to roughly one request per second per API key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator

import httpx
import pandas as pd

from ..config import AmbientCredentials

BASE_URL = "https://rt.ambientweather.net/v1"
MAX_RECORDS_PER_CALL = 288
MIN_REQUEST_INTERVAL = 1.1  # seconds; Ambient allows ~1 req/s
INHG_TO_HPA = 33.8639

# Ambient field name -> our canonical column name.
FIELD_MAP = {
    "tempf": "temp_f",
    "dewPoint": "dewpoint_f",
    "humidity": "rh",
    "windspeedmph": "wind_mph",
    "windgustmph": "gust_mph",
    "winddir": "wind_dir_deg",
    "hourlyrainin": "rain_hourly_in",
    "dailyrainin": "rain_daily_in",
    "solarradiation": "solar_wm2",
    "uv": "uv",
}


@dataclass
class Device:
    mac: str
    name: str
    location: str | None


class AmbientClient:
    """Thin, rate-limited client for the Ambient Weather REST API."""

    def __init__(self, credentials: AmbientCredentials, timeout: float = 30.0):
        self._credentials = credentials
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout)
        self._last_request_at = 0.0

    def __enter__(self) -> AmbientClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, path: str, params: dict, max_retries: int = 5) -> list | dict:
        params = {
            "apiKey": self._credentials.api_key,
            "applicationKey": self._credentials.application_key,
            **params,
        }
        for attempt in range(max_retries):
            self._throttle()
            response = self._client.get(path, params=params)

            # Ambient answers a tripped rate limit with 429, and sheds load
            # with 502/503 under real traffic. Both are worth waiting out.
            if response.status_code in (429, 502, 503):
                backoff = min(2**attempt, 30)
                time.sleep(backoff)
                continue

            response.raise_for_status()
            return response.json()

        raise RuntimeError(
            f"Ambient API still rate-limiting after {max_retries} attempts ({path})"
        )

    def devices(self) -> list[Device]:
        """List the devices on this account (use to find your MAC address)."""
        payload = self._get("/devices", {})
        devices = []
        for entry in payload:
            info = entry.get("info", {})
            coords = info.get("coords", {})
            devices.append(
                Device(
                    mac=entry.get("macAddress", ""),
                    name=info.get("name", ""),
                    location=coords.get("address") or info.get("location"),
                )
            )
        return devices

    def fetch_page(self, end: datetime, limit: int = MAX_RECORDS_PER_CALL) -> list[dict]:
        """Fetch up to `limit` records at or before `end`, newest first."""
        mac = self._credentials.mac
        if not mac:
            raise ValueError("AMBIENT_MAC is required to fetch device history")

        end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
        payload = self._get(f"/devices/{mac}", {"endDate": end_ms, "limit": limit})
        return payload if isinstance(payload, list) else []

    def iter_history(
        self,
        start: datetime,
        end: datetime | None = None,
        on_page: Callable[[datetime, int], None] | None = None,
        max_gap_days: int = 45,
    ) -> Iterator[list[dict]]:
        """Walk history backward from `end` to `start`, yielding pages.

        Yields raw record dicts so the caller decides how to normalize and
        store them; `on_page` is called with (oldest timestamp, record count)
        after each page for progress reporting.

        An empty page means the station reported nothing in that window, which
        is an outage far more often than it is the start of the record — power
        cuts, console resets, and dead console batteries all produce multi-day
        holes. So a gap is stepped over a day at a time, and the walk stops
        only after `max_gap_days` of continuous silence.
        """
        cursor = (end or datetime.now(timezone.utc)).astimezone(timezone.utc)
        start = start.astimezone(timezone.utc)
        empty_days = 0

        while cursor > start:
            page = self.fetch_page(cursor)
            if not page:
                empty_days += 1
                if empty_days > max_gap_days:
                    return  # Long silence — treat as the start of the record.
                cursor -= timedelta(days=1)
                continue

            empty_days = 0
            oldest_ms = min(record["dateutc"] for record in page)
            oldest = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)

            yield page
            if on_page:
                on_page(oldest, len(page))

            # Step just past the oldest record to avoid re-fetching it.
            next_cursor = oldest - timedelta(milliseconds=1)
            if next_cursor >= cursor:
                return  # No progress — guard against an infinite loop.
            cursor = next_cursor


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Normalize raw Ambient records into the obs_station schema."""
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(frame["dateutc"], unit="ms", utc=True)

    for source_field, column in FIELD_MAP.items():
        out[column] = (
            pd.to_numeric(frame[source_field], errors="coerce")
            if source_field in frame.columns
            else pd.NA
        )

    # Prefer relative (sea-level-adjusted) pressure; fall back to absolute.
    pressure_in = None
    for field in ("baromrelin", "baromabsin"):
        if field in frame.columns:
            pressure_in = pd.to_numeric(frame[field], errors="coerce")
            break
    out["pressure_hpa"] = pressure_in * INHG_TO_HPA if pressure_in is not None else pd.NA

    return out.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
