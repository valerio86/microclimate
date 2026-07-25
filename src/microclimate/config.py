"""Configuration loaded from .env / environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    elevation_m: float | None
    timezone: str

    @classmethod
    def from_env(cls) -> Location:
        elevation = os.getenv("ELEVATION_M", "").strip()
        return cls(
            latitude=float(_require("LATITUDE")),
            longitude=float(_require("LONGITUDE")),
            elevation_m=float(elevation) if elevation else None,
            timezone=os.getenv("TIMEZONE", "UTC").strip() or "UTC",
        )


@dataclass(frozen=True)
class AmbientCredentials:
    api_key: str
    application_key: str
    mac: str | None

    @classmethod
    def from_env(cls, require_mac: bool = True) -> AmbientCredentials:
        return cls(
            api_key=_require("AMBIENT_API_KEY"),
            application_key=_require("AMBIENT_APPLICATION_KEY"),
            mac=_require("AMBIENT_MAC") if require_mac else os.getenv("AMBIENT_MAC"),
        )


@dataclass(frozen=True)
class PurpleAirCredentials:
    api_key: str
    sensor_index: int

    @classmethod
    def from_env(cls) -> PurpleAirCredentials:
        return cls(
            api_key=_require("PURPLEAIR_API_KEY"),
            sensor_index=int(_require("PURPLEAIR_SENSOR_INDEX")),
        )


def data_dir() -> Path:
    path = Path(os.getenv("DATA_DIR", "./data")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "microclimate.duckdb"
