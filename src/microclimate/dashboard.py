"""Assemble everything the dashboard page needs.

The page has one job: answer "why did it say that?". So every element here
exists to back an alert — the overnight curve explains a frost warning, the
per-model totals explain a rain probability, and the trust panel states what the
system cannot do, because a forecast tool that hides its limits invites being
over-trusted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"

# WMO weather code -> a small icon vocabulary the page knows how to draw.
# This is the raw model's own classification, not a locally-corrected one —
# unlike the frost and rain calls, nothing here has been verified against the
# station, so the landing page must not present it with the same confidence.
WEATHER_ICONS = {
    0: "clear", 1: "clear",
    2: "partly-cloudy",
    3: "cloudy",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle", 56: "drizzle", 57: "drizzle",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    80: "rain", 81: "rain", 82: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow", 85: "snow", 86: "snow",
    95: "storm", 96: "storm", 99: "storm",
}


def _weather_icon(code) -> str | None:
    if code is None or code != code:  # NaN != NaN
        return None
    return WEATHER_ICONS.get(int(round(code)), "cloudy")


# Verified numbers, shown on the page so the reader knows how far to trust it.
TRUST = {
    "rain": "Brier skill 0.41 ± 0.15 across walk-forward folds; 0.52 on sealed data.",
    "frost": "Warning at 34 °F caught 14 of 14 frosts on held-out marginal nights.",
    "temperature": "Correction skill 0.06 ± 0.08 — real but modest, and mostly a summer effect.",
    "cannot": [
        "Wind — the anemometer reads about half of true speed.",
        "Snow — an unheated gauge does not register frozen precipitation.",
        "Air quality — the PurpleAir sensor needs a winter of history first.",
        "Hourly shower timing — convective cells miss a point sensor most of the time.",
    ],
}


def _json_safe(value):
    """Convert to something `json.dumps` can emit, with NaN becoming null.

    Plain Python floats must be handled as well as numpy ones: `iterrows` yields
    object-dtype Series, so a NaN arrives as a builtin float and would otherwise
    slip through as the literal `NaN`, which is not valid JSON and leaves the
    page silently failing to parse with nothing logged anywhere.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if value != value else round(float(value), 3)  # NaN != NaN
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_payload(
    *,
    station: pd.DataFrame,
    nights: pd.DataFrame,
    hourly: pd.DataFrame,
    rain_days: pd.DataFrame,
    per_model: dict[str, pd.Series],
    alerts: list,
    timezone_name: str,
    forecast_through: pd.Timestamp | None,
    history: dict | None = None,
) -> dict:
    """Everything the page renders, as plain JSON-safe structures."""
    now = datetime.now(timezone.utc)

    latest = station.dropna(subset=["temp_f"]).tail(1)
    current = {}
    if not latest.empty:
        row = latest.iloc[0]
        observed_at = pd.to_datetime(row["ts"], utc=True)
        current = {
            "temp_f": _json_safe(row.get("temp_f")),
            "rh": _json_safe(row.get("rh")),
            "wind_mph": _json_safe(row.get("wind_mph")),
            "observed_at": observed_at.tz_convert(timezone_name).isoformat(),
            "age_hours": round((now - observed_at).total_seconds() / 3600, 1),
        }

    # Rain so far today, straight from the station's own daily accumulator.
    # Summing derived hourly increments would work too, but the counter is what
    # the console itself shows, so the page agrees with the hardware.
    rain_today = None
    if not station.empty and "rain_daily_in" in station.columns:
        recent = station.dropna(subset=["rain_daily_in"]).tail(1)
        if not recent.empty:
            rain_today = float(recent.iloc[0]["rain_daily_in"])

    curve = []
    for _, row in hourly.iterrows():
        curve.append(
            {
                "t": pd.to_datetime(row["valid_time"], utc=True)
                .tz_convert(timezone_name)
                .isoformat(),
                "raw": _json_safe(row["temp_f_forecast"]),
                "corrected": _json_safe(row["corrected"]),
                # Uncorrected model output — the day-zoom detail, not a claim.
                "dewpoint_f": _json_safe(row.get("dewpoint_f")),
                "wind_mph": _json_safe(row.get("wind_mph_forecast")),
                "gust_mph": _json_safe(row.get("gust_mph")),
                "cloud_cover": _json_safe(row.get("cloud_cover")),
                "weather_code": _json_safe(row.get("weather_code")),
            }
        )

    rain = []
    for _, row in rain_days.iterrows():
        day = row["day"]
        rain.append(
            {
                "day": _json_safe(day),
                "chance": _json_safe(row.get("rain_chance")),
                "mean_in": _json_safe(row.get("model_mean_in")),
                "spread_in": _json_safe(row.get("model_spread_in")),
                "models_wet": _json_safe(row.get("models_wet")),
                "per_model": {
                    name.replace("open-meteo:", ""): _json_safe(series.get(day, np.nan))
                    for name, series in per_model.items()
                },
            }
        )

    night_rows = [
        {
            "day": _json_safe(row["day"]),
            "raw_min": _json_safe(row["raw_min"]),
            "corrected_min": _json_safe(row["corrected_min"]),
        }
        for _, row in nights.iterrows()
    ]
    daily = _daily_summary(hourly, timezone_name, night_rows, rain, alerts)

    return {
        "generated": now.astimezone().isoformat(),
        "timezone": timezone_name,
        "current": current,
        "rain_today_in": _json_safe(rain_today) if rain_today is not None else None,
        "forecast_through": _json_safe(forecast_through),
        "alerts": [
            {
                "day": _json_safe(alert.day),
                "kind": alert.kind,
                "severity": alert.severity,
                "headline": alert.headline,
                "detail": alert.detail,
            }
            for alert in alerts
        ],
        "nights": night_rows,
        "curve": curve,
        "rain": rain,
        "daily": daily,
        "history": _history_block(history),
        "trust": TRUST,
    }


def _daily_summary(
    hourly: pd.DataFrame,
    timezone_name: str,
    night_rows: list[dict],
    rain: list[dict],
    alerts: list,
) -> list[dict]:
    """One card's worth of numbers per day, for the landing page.

    The overnight low is `night_rows`' own `corrected_min` — the same number
    the frost alert is built from — rather than a fresh calendar-day minimum,
    so this card can never show a different low than the alert above it
    explains. The high and the day-type icon have no existing verified
    counterpart, so they are computed fresh from the hourly corrected curve
    and the raw model's `weather_code`, respectively.
    """
    if hourly.empty:
        return []

    local = pd.to_datetime(hourly["valid_time"], utc=True).dt.tz_convert(timezone_name)
    by_day = hourly.assign(_day=local.dt.date.map(lambda d: d.isoformat()), _hour=local.dt.hour)
    groups = {day: group for day, group in by_day.groupby("_day")}

    lows = {row["day"]: row["corrected_min"] for row in night_rows}
    # Severity, not a bare flag, so the day card's badge is drawn from the same
    # `--critical`/`--warning` tokens the alert above it already uses — it can
    # color the same night differently only by disagreeing about the fact.
    frost_severity = {_json_safe(alert.day): alert.severity for alert in alerts if alert.kind == "frost"}

    out = []
    for r in rain:  # `rain` already carries exactly the requested day window
        day = r["day"]
        group = groups.get(day)
        max_f, icon = None, None
        if group is not None:
            corrected = group["corrected"].dropna()
            if not corrected.empty:
                max_f = _json_safe(corrected.max())
            if "weather_code" in group.columns and group["_hour"].notna().any():
                rep = group.loc[(group["_hour"] - 14).abs().idxmin()]
                icon = _weather_icon(rep.get("weather_code"))
        out.append(
            {
                "day": day,
                "min_f": lows.get(day),
                "max_f": max_f,
                "icon": icon,
                "rain_chance": r["chance"],
                "rain_in": r["mean_in"],
                "frost": frost_severity.get(day),
            }
        )
    return out


def _history_block(history: dict | None) -> dict:
    """Past predictions against outcomes, for the verification panel."""
    if not history:
        return {}

    nights = history.get("nights", pd.DataFrame())
    rain_rows = history.get("rain", pd.DataFrame())
    curve = history.get("curve", pd.DataFrame())
    key = "night" if "night" in getattr(nights, "columns", []) else "day"

    return {
        "summary": {k: _json_safe(v) for k, v in (history.get("summary") or {}).items()},
        "curve": [
            {
                "t": pd.to_datetime(row["valid_time"], utc=True).isoformat(),
                "raw": _json_safe(row["temp_f_forecast"]),
                "corrected": _json_safe(row["corrected"]),
                "actual": _json_safe(row["temp_f_actual"]),
                "rain": _json_safe(row.get("rain_in")),
            }
            for _, row in curve.iterrows()
        ]
        if not curve.empty
        else [],
        "nights": [
            {
                "day": _json_safe(row[key]),
                "actual": _json_safe(row["actual_min"]),
                "raw": _json_safe(row["raw_min"]),
                "corrected": _json_safe(row["corrected_min"]),
            }
            for _, row in nights.iterrows()
        ]
        if not nights.empty
        else [],
        "rain": [
            {
                "day": _json_safe(row["day"]),
                "chance": _json_safe(row.get("rain_chance")),
                "wet": bool(row["wet"]),
            }
            for _, row in rain_rows.iterrows()
        ]
        if not rain_rows.empty
        else [],
    }


def render(payload: dict, destination: Path) -> Path:
    """Write the self-contained page. No network, no build step, opens from disk."""
    template = TEMPLATE.read_text()
    if "__DATA__" not in template:
        raise ValueError("dashboard template is missing its __DATA__ placeholder")

    # allow_nan=False turns a stray NaN into a loud failure here rather than an
    # unparseable page discovered later. ensure_ascii=False keeps degree signs
    # readable in the source, which the UTF-8 charset already supports.
    encoded = json.dumps(
        payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__DATA__", encoded), encoding="utf-8")
    return destination
