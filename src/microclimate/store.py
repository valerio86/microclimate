"""DuckDB-backed storage for observations and forecasts.

Three tables:
  obs_station   5-minute readings from the Ambient Weather station (ground truth)
  obs_air       PurpleAir readings
  forecasts     Open-Meteo model output, one row per (valid_time, lead_days)

All timestamps are stored as UTC. Local-time questions are answered at query
time using the configured timezone, so the stored data stays unambiguous.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from .config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS obs_station (
    ts              TIMESTAMPTZ NOT NULL,
    temp_f          DOUBLE,
    dewpoint_f      DOUBLE,
    rh              DOUBLE,
    wind_mph        DOUBLE,
    gust_mph        DOUBLE,
    wind_dir_deg    DOUBLE,
    pressure_hpa    DOUBLE,
    rain_hourly_in  DOUBLE,
    rain_daily_in   DOUBLE,
    solar_wm2       DOUBLE,
    uv              DOUBLE,
    PRIMARY KEY (ts)
);

CREATE TABLE IF NOT EXISTS obs_air (
    ts              TIMESTAMPTZ NOT NULL,
    sensor_index    INTEGER NOT NULL,
    pm25            DOUBLE,
    pm25_cf1        DOUBLE,
    pm10            DOUBLE,
    temp_f          DOUBLE,
    rh              DOUBLE,
    PRIMARY KEY (ts, sensor_index)
);

CREATE TABLE IF NOT EXISTS forecasts (
    valid_time      TIMESTAMPTZ NOT NULL,
    lead_days       INTEGER NOT NULL,  -- 0 = analysis/current run, N = run from N days earlier
    source          VARCHAR NOT NULL,  -- e.g. 'open-meteo'
    temp_f          DOUBLE,
    dewpoint_f      DOUBLE,
    rh              DOUBLE,
    wind_mph        DOUBLE,
    gust_mph        DOUBLE,
    wind_dir_deg    DOUBLE,
    pressure_hpa    DOUBLE,
    precip_in       DOUBLE,
    cloud_cover     DOUBLE,
    solar_wm2       DOUBLE,
    wind_100m_mph   DOUBLE,
    direct_wm2      DOUBLE,
    diffuse_wm2     DOUBLE,
    vpd_kpa         DOUBLE,
    PRIMARY KEY (valid_time, lead_days, source)
);
"""

# Columns added after the original schema. CREATE TABLE IF NOT EXISTS will not
# add them to a database that already exists, so they are applied explicitly on
# every connect. Adding a column is cheap and idempotent; leave entries here
# permanently so any older database can still be opened.
MIGRATIONS: dict[str, dict[str, str]] = {
    "forecasts": {
        "wind_100m_mph": "DOUBLE",
        "direct_wm2": "DOUBLE",
        "diffuse_wm2": "DOUBLE",
        "vpd_kpa": "DOUBLE",
        "snowfall_in": "DOUBLE",
        "showers_in": "DOUBLE",
        "weather_code": "DOUBLE",
    },
}


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Bring an existing database up to the current column set."""
    for table, columns in MIGRATIONS.items():
        for name, sql_type in columns.items():
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}"
            )


@contextmanager
def connect(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the database, creating tables if needed."""
    conn = duckdb.connect(str(path or db_path()))
    try:
        conn.execute(SCHEMA)
        migrate(conn)
        yield conn
    finally:
        conn.close()


def upsert(
    conn: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame
) -> int:
    """Insert rows, replacing any that collide on the primary key.

    Backfills are re-runnable, and the station occasionally re-reports a
    timestamp, so idempotency here matters more than insert speed.
    """
    if frame.empty:
        return 0

    columns = [
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
    ]
    aligned = frame.reindex(columns=columns)

    conn.register("_incoming", aligned)
    conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _incoming")
    conn.unregister("_incoming")
    return len(aligned)


def coverage(
    conn: duckdb.DuckDBPyConnection, table: str, time_column: str = "ts"
) -> tuple[datetime | None, datetime | None, int]:
    """Return (earliest, latest, row count) for a table."""
    row = conn.execute(
        f"SELECT min({time_column}), max({time_column}), count(*) FROM {table}"
    ).fetchone()
    return (row[0], row[1], row[2]) if row else (None, None, 0)


def earliest_station_ts(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    """Oldest station reading on record — the resume point for a backfill."""
    return coverage(conn, "obs_station")[0]


def export_parquet(conn: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    """Write a table to Parquet for portability and external analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
