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
        "nights": [
            {
                "day": _json_safe(row["day"]),
                "raw_min": _json_safe(row["raw_min"]),
                "corrected_min": _json_safe(row["corrected_min"]),
            }
            for _, row in nights.iterrows()
        ],
        "curve": curve,
        "rain": rain,
        "history": _history_block(history),
        "trust": TRUST,
    }


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
