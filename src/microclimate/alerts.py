"""Conditions worth interrupting someone for.

The project's two verified results are both *decisions*, not readings: whether to
cover the plants tonight, and whether Tuesday is worth planning around. A
dashboard would bury them among numbers available from any weather app. So the
default is silence, and the bar for speaking is that something is actionable.

Three rules, each tied to something measured rather than assumed:

  frost       The overnight minimum, corrected. Verified: warning at 34 °F
              caught 14 of 14 frosts on held-out marginal nights, where taking
              the raw 32 °F line at face value missed 10 of 14.

  rain        Daily rain probability from the frozen three-feature model, which
              scored Brier skill 0.41 across folds and 0.52 on sealed data.

  deviation   Where the corrected forecast departs from the public one. This is
              the project's whole reason for existing, and it is the alert no
              off-the-shelf app can produce: not "it will be cold" but "the
              forecast says 38 °F and it will be 33 °F in your field".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# Verified on held-out nights: see README, "The frost call, verified".
FROST_WARN_F = 34.0
HARD_FREEZE_F = 28.0
FREEZE_F = 32.0

# Only speak about rain when the models substantially agree. Below this the
# honest answer is "maybe", which is not worth a notification.
RAIN_LIKELY = 0.70
RAIN_HEAVY_IN = 0.50

# Below this the correction is not distinguishable from noise: cross-validated
# temperature skill is 0.06 with a standard deviation of 0.08, so a small
# disagreement with the public forecast means nothing.
DEVIATION_F = 3.0

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Alert:
    day: date
    kind: str
    severity: str
    headline: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.day:%a %d %b}: {self.headline}"


def upcoming_nights(
    hourly: pd.DataFrame, timezone: str, corrections: np.ndarray | None = None
) -> pd.DataFrame:
    """Nightly minima from a live forecast, raw and corrected.

    A night runs 18:00 to 09:00 the following morning, the same window the frost
    verification used. Separate from `frost.nightly_minima` because that one
    needs observations to score against, and these nights have not happened yet.

    The correction is applied per hour *before* taking the minimum: the coldest
    corrected hour need not be the coldest raw hour.
    """
    if hourly.empty:
        return pd.DataFrame(columns=["day", "raw_min", "corrected_min", "hours"])

    frame = hourly.copy()
    local = pd.to_datetime(frame["valid_time"], utc=True).dt.tz_convert(timezone)
    frame["local_hour"] = local.dt.hour
    frame["night"] = (local - pd.Timedelta(hours=12)).dt.date
    frame["corrected"] = frame["temp_f_forecast"] - (
        0.0 if corrections is None else corrections
    )

    overnight = frame[(frame["local_hour"] >= 18) | (frame["local_hour"] <= 9)]
    grouped = overnight.groupby("night").agg(
        raw_min=("temp_f_forecast", "min"),
        corrected_min=("corrected", "min"),
        hours=("temp_f_forecast", "size"),
    )
    # A partial night would report a minimum from the evening only and miss the
    # pre-dawn low, which is exactly the hour that matters.
    complete = grouped[grouped["hours"] >= 12].reset_index()
    return complete.rename(columns={"night": "day"})


def frost_alerts(nights: pd.DataFrame) -> list[Alert]:
    """Frost risk from corrected overnight minima.

    `nights` needs `day`, `corrected_min` and `raw_min`.
    """
    out = []
    for _, row in nights.iterrows():
        corrected = row["corrected_min"]
        if pd.isna(corrected) or corrected > FROST_WARN_F:
            continue

        if corrected <= HARD_FREEZE_F:
            severity, label = "critical", "Hard freeze"
        elif corrected <= FREEZE_F:
            severity, label = "critical", "Freeze"
        else:
            severity, label = "warning", "Frost risk"

        detail = (
            f"Corrected low {corrected:.0f} °F "
            f"(public forecast {row['raw_min']:.0f} °F). "
            f"Warning threshold is {FROST_WARN_F:.0f} °F because the overnight "
            f"minimum here runs below forecast on clear, calm nights."
        )
        out.append(
            Alert(row["day"], "frost", severity, f"{label} — {corrected:.0f} °F", detail)
        )
    return out


def rain_alerts(days: pd.DataFrame) -> list[Alert]:
    """Rain worth planning around: likely, or heavy, or both."""
    out = []
    for _, row in days.iterrows():
        chance = row.get("rain_chance", np.nan)
        amount = row.get("model_mean_in", np.nan)
        if pd.isna(chance):
            continue

        heavy = not pd.isna(amount) and amount >= RAIN_HEAVY_IN
        if chance < RAIN_LIKELY and not heavy:
            continue

        agreement = f"{int(row['models_wet'])}/3 models"
        if heavy and chance >= RAIN_LIKELY:
            severity = "warning"
            headline = f"Heavy rain likely — {chance:.0%}, {amount:.2f} in"
        elif heavy:
            severity = "info"
            headline = f"Heavy rain possible — {chance:.0%}, {amount:.2f} in"
        else:
            severity = "info"
            headline = f"Rain likely — {chance:.0%}"

        detail = (
            f"{agreement} forecast rain, mean {amount:.2f} in, "
            f"spread {row.get('model_spread_in', float('nan')):.2f}. "
            f"Model agreement is the strongest available signal here."
        )
        out.append(Alert(row["day"], "rain", severity, headline, detail))
    return out


def deviation_alerts(nights: pd.DataFrame) -> list[Alert]:
    """Where this location will differ materially from the public forecast.

    The alert nothing off the shelf can give: not that it will be cold, but that
    the published number is wrong for this field, and by how much.
    """
    out = []
    for _, row in nights.iterrows():
        corrected, raw = row["corrected_min"], row["raw_min"]
        if pd.isna(corrected) or pd.isna(raw):
            continue
        gap = raw - corrected
        if abs(gap) < DEVIATION_F:
            continue

        direction = "colder" if gap > 0 else "warmer"
        out.append(
            Alert(
                row["day"],
                "deviation",
                "info",
                f"Overnight {abs(gap):.0f} °F {direction} than forecast",
                f"Public forecast {raw:.0f} °F, expected here {corrected:.0f} °F. "
                f"Cold air pools in this terrain on clear, calm nights, which a "
                f"grid forecast cannot see.",
            )
        )
    return out


def evaluate(nights: pd.DataFrame, days: pd.DataFrame) -> list[Alert]:
    """All alerts, most urgent first, then soonest.

    Frost outranks a deviation note about the same night: if it is going to
    freeze, that is the headline and the reason is detail.
    """
    alerts = frost_alerts(nights) + rain_alerts(days) + deviation_alerts(nights)

    frost_days = {alert.day for alert in alerts if alert.kind == "frost"}
    alerts = [
        alert
        for alert in alerts
        if not (alert.kind == "deviation" and alert.day in frost_days)
    ]

    return sorted(alerts, key=lambda a: (SEVERITY_ORDER[a.severity], a.day))
