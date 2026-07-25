"""PurpleAir API client.

Secondary source for now: the sensor was installed 2026-07-24, so it has no
history to train on yet. PM2.5 becomes a genuinely useful forecast target once
there's a season or two behind it (smoke, inversions, wood smoke at night).

No scheduled polling is needed to preserve data. PurpleAir stores sensor
history in its own cloud — network-wide back to 2016 — so the record can be
pulled retroactively at any point via `history()`. Sensor owners read their own
sensor for free under PurpleAir's points-based API billing.

Note the two PM2.5 flavors: `_atm` is the reading most dashboards show, `_cf_1`
reads high outdoors but is the input the EPA correction expects. Store both;
decide which to trust at analysis time.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

from ..config import PurpleAirCredentials

BASE_URL = "https://api.purpleair.com/v1"

FIELDS = ["pm2.5_atm", "pm2.5_cf_1", "pm10.0_atm", "temperature", "humidity"]

FIELD_MAP = {
    "pm2.5_atm": "pm25",
    "pm2.5_cf_1": "pm25_cf1",
    "pm10.0_atm": "pm10",
    "temperature": "temp_f",
    "humidity": "rh",
}


class PurpleAirClient:
    def __init__(self, credentials: PurpleAirCredentials, timeout: float = 30.0):
        self._credentials = credentials
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"X-API-Key": credentials.api_key},
        )

    def __enter__(self) -> PurpleAirClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def current(self) -> pd.DataFrame:
        """Latest reading from the configured sensor."""
        response = self._client.get(
            f"/sensors/{self._credentials.sensor_index}",
            params={"fields": ",".join(FIELDS)},
        )
        response.raise_for_status()
        sensor = response.json().get("sensor", {})

        row = {"ts": datetime.now(timezone.utc), "sensor_index": self._credentials.sensor_index}
        for field, column in FIELD_MAP.items():
            row[column] = sensor.get(field)
        return pd.DataFrame([row])

    def history(
        self, start: datetime, end: datetime, average_minutes: int = 60
    ) -> pd.DataFrame:
        """Historical readings for one time range.

        PurpleAir caps rows per response, so a long backfill needs to be walked
        in chunks the way the Ambient client does. Not yet implemented — this
        returns a single page, which is enough while the sensor is days old.
        """
        response = self._client.get(
            f"/sensors/{self._credentials.sensor_index}/history",
            params={
                "start_timestamp": int(start.timestamp()),
                "end_timestamp": int(end.timestamp()),
                "average": average_minutes,
                "fields": ",".join(FIELDS),
            },
        )
        response.raise_for_status()
        payload = response.json()

        columns = payload.get("fields", [])
        rows = payload.get("data", [])
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=columns)
        out = pd.DataFrame()
        out["ts"] = pd.to_datetime(frame["time_stamp"], unit="s", utc=True)
        out["sensor_index"] = self._credentials.sensor_index
        for field, column in FIELD_MAP.items():
            out[column] = pd.to_numeric(frame.get(field), errors="coerce")
        return out.sort_values("ts").reset_index(drop=True)
