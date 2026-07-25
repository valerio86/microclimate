"""Does the correction actually improve the frost call?

MAE is the wrong yardstick for this. Frost is a *decision*: cover the plants or
don't. What matters is how often the call is right, and the two ways of being
wrong are not equally bad — a missed frost kills something, a false alarm costs
an evening's inconvenience. So this scores the binary decision directly.

Definitions, all evaluated on held-out nights the corrector never saw:

    night      18:00 through 09:00 the following morning, local time
    frost      the night's minimum observed temperature <= threshold
    predicted  the night's minimum forecast temperature <= warning level

Raising the warning level above freezing buys sensitivity at the cost of false
alarms; `skill_curve` reports that trade so the threshold can be chosen against
real numbers rather than a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FROST_F = 32.0
MIN_HOURS_PER_NIGHT = 10


def nightly_minima(
    paired: pd.DataFrame,
    timezone: str,
    corrections: np.ndarray | None = None,
) -> pd.DataFrame:
    """Collapse hourly rows to one row per night, keeping the minima.

    `corrections` is the per-hour predicted error; it is subtracted from the
    forecast before taking the minimum, because the coldest *corrected* hour is
    not necessarily the coldest raw hour.
    """
    frame = paired.copy()
    local = pd.to_datetime(frame["valid_time"], utc=True).dt.tz_convert(timezone)
    frame["local_hour"] = local.dt.hour

    # Shift by 12 hours so an evening and the following morning share a night.
    frame["night"] = (local - pd.Timedelta(hours=12)).dt.date

    frame["corrected"] = frame["temp_f_forecast"] - (
        0.0 if corrections is None else corrections
    )

    overnight = frame[(frame["local_hour"] >= 18) | (frame["local_hour"] <= 9)]
    grouped = overnight.groupby("night").agg(
        actual_min=("temp_f_actual", "min"),
        raw_min=("temp_f_forecast", "min"),
        corrected_min=("corrected", "min"),
        hours=("temp_f_actual", "count"),
    )
    return grouped[grouped["hours"] >= MIN_HOURS_PER_NIGHT].dropna().reset_index()


def contingency(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """Standard 2x2 verification scores for a yes/no forecast."""
    predicted, observed = predicted.astype(bool), observed.astype(bool)
    hits = int((predicted & observed).sum())
    misses = int((~predicted & observed).sum())
    false_alarms = int((predicted & ~observed).sum())
    correct_negatives = int((~predicted & ~observed).sum())

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    pod = ratio(hits, hits + misses)  # of real frosts, how many were caught
    far = ratio(false_alarms, hits + false_alarms)  # of warnings, how many were wrong
    pofd = ratio(false_alarms, false_alarms + correct_negatives)

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_neg": correct_negatives,
        "pod": pod,
        "far": far,
        # Critical success index: hits over everything that mattered.
        "csi": ratio(hits, hits + misses + false_alarms),
        # Peirce skill: hit rate minus false-alarm rate. 0 = no better than
        # always saying no; 1 = perfect. Immune to how rare frost is.
        "pss": pod - pofd if not (np.isnan(pod) or np.isnan(pofd)) else np.nan,
    }


def skill_curve(
    nights: pd.DataFrame,
    warn_levels: tuple[float, ...] = (32.0, 34.0, 36.0, 38.0),
    threshold: float = FROST_F,
) -> pd.DataFrame:
    """Score raw and corrected forecasts across a range of warning levels."""
    observed = (nights["actual_min"] <= threshold).to_numpy()

    rows = []
    for level in warn_levels:
        for label, column in (("raw", "raw_min"), ("corrected", "corrected_min")):
            predicted = (nights[column] <= level).to_numpy()
            rows.append(
                {"warn_at": level, "forecast": label, **contingency(predicted, observed)}
            )
    return pd.DataFrame(rows)


def summarize(nights: pd.DataFrame, threshold: float = FROST_F) -> dict:
    observed = (nights["actual_min"] <= threshold).to_numpy()
    return {
        "nights": int(len(nights)),
        "frost_nights": int(observed.sum()),
        "base_rate": float(observed.mean()) if len(nights) else np.nan,
        "raw_bias_f": float((nights["raw_min"] - nights["actual_min"]).mean()),
        "corrected_bias_f": float((nights["corrected_min"] - nights["actual_min"]).mean()),
        "raw_mae_f": float((nights["raw_min"] - nights["actual_min"]).abs().mean()),
        "corrected_mae_f": float(
            (nights["corrected_min"] - nights["actual_min"]).abs().mean()
        ),
    }
