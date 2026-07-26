"""Rain chance: will measurable rain fall *here*?

This is a different problem from temperature, and needs different everything.

**The target is binary and the metric is not MAE.** Rain is zero most hours and
heavy-tailed when it is not, so a mean error is meaningless. What is wanted is a
calibrated probability, scored with the Brier score and checked with a
reliability curve — when the model says 30%, it should rain about 30% of the
time.

**Two things look like the same problem and are not.** Comparing ICON's
forecast against the gauge:

    below 36 °F   the gauge sees 0-50% of forecast-wet hours
    above 50 °F   the gauge sees 29% of them

The first is broken ground truth: an unheated tipping bucket does not register
snow until it melts, so the precipitation happened and the record says dry.
Those hours are *excluded* — a model cannot learn from a label that is wrong.

The second is not a fault at all. A convective cell inside a 13 km grid square
genuinely misses a point sensor most of the time. "ICON says rain, it falls here
29% of the time" is a true and useful statement, and reproducing it is precisely
the job. That is signal, and it is kept.

**Aggregation.** Daily is the primary target. Hourly timing of a shower at a
point is close to unpredictable, while "will it rain today" is answerable, and
daily aggregation averages out the timing noise without discarding the event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The gauge's own resolution is 0.01 in; require a clear tip to call it wet.
WET_THRESHOLD_IN = 0.01
# Below this the bucket under-catches badly even when snowfall is not forecast.
COLD_CUTOFF_F = 34.0


def usable_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows where the gauge can be trusted to have seen liquid precipitation.

    Excluded when snow is forecast, or when it is cold enough that frozen
    precipitation is plausible and the forecast simply did not say so.
    """
    snowfall = pd.to_numeric(frame.get("snowfall_in"), errors="coerce").fillna(0.0)
    temperature = pd.to_numeric(frame.get("temp_f_actual"), errors="coerce")

    cold = temperature.lt(COLD_CUTOFF_F).fillna(False)
    snowy = snowfall.gt(0.0)
    return ~(cold | snowy)


def daily_targets(paired: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """One row per local day: did measurable rain fall, and what was forecast.

    A day is dropped entirely if any of its hours were snow-affected — a partial
    day would understate the total and mislabel the day as dry.
    """
    if paired.empty:
        return pd.DataFrame()

    frame = paired.copy()
    local = pd.to_datetime(frame["valid_time"], utc=True).dt.tz_convert(timezone)
    frame["day"] = local.dt.date
    frame["usable"] = usable_mask(frame)

    grouped = frame.groupby(["day", "lead_days"])
    daily = grouped.agg(
        observed_in=("rain_in", "sum"),
        forecast_in=("precip_in", "sum"),
        showers_in=("showers_in", "sum"),
        hours=("rain_in", "size"),
        usable_hours=("usable", "sum"),
        mean_temp_f=("temp_f_actual", "mean"),
    ).reset_index()

    complete = (daily["hours"] >= 20) & (daily["usable_hours"] == daily["hours"])
    daily = daily[complete].copy()
    daily["wet"] = daily["observed_in"] >= WET_THRESHOLD_IN
    # Fraction of the day's forecast precipitation that was convective — high
    # values mean scattered cells that may miss the station entirely.
    daily["convective_share"] = np.where(
        daily["forecast_in"] > 0, daily["showers_in"] / daily["forecast_in"], 0.0
    ).clip(0, 1)
    return daily.reset_index(drop=True)


def brier_score(probability: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of a probability forecast. Lower is better."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    valid = ~(np.isnan(probability) | np.isnan(outcome))
    if not valid.any():
        return float("nan")
    return float(((probability[valid] - outcome[valid]) ** 2).mean())


def brier_skill(
    probability: np.ndarray, outcome: np.ndarray, reference: np.ndarray | None = None
) -> float:
    """Brier score against a reference, positive meaning better.

    The default reference is climatology — always predicting the base rate.
    Beating that is the minimum bar for a probability forecast to be worth
    anything at all.
    """
    outcome = np.asarray(outcome, dtype=float)
    if reference is None:
        reference = np.full(outcome.shape, np.nanmean(outcome))
    baseline = brier_score(reference, outcome)
    if not baseline:
        return float("nan")
    return 1 - brier_score(probability, outcome) / baseline


def reliability(
    probability: np.ndarray, outcome: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Predicted probability against observed frequency, bin by bin.

    A well-calibrated forecast sits on the diagonal. This catches the failure
    the Brier score hides: a model can score respectably while being
    systematically overconfident.
    """
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    valid = ~(np.isnan(probability) | np.isnan(outcome))
    probability, outcome = probability[valid], outcome[valid]

    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probability, edges) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        selected = index == b
        if not selected.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(selected.sum()),
                "mean_predicted": float(probability[selected].mean()),
                "observed_rate": float(outcome[selected].mean()),
            }
        )
    return pd.DataFrame(rows)


def climatological_probability(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Base rate from the training period — the floor any model must clear."""
    return np.full(len(test), float(train["wet"].mean()))


def forecast_amount_probability(
    train: pd.DataFrame, test: pd.DataFrame, bins: tuple[float, ...] = (0, 0.005, 0.02, 0.05, 0.15, 1e9)
) -> np.ndarray:
    """Calibrate forecast amount into a probability, learned from the training set.

    Deliberately simple, and the honest baseline for any ML model: bucket the
    forecast total and look up how often it actually rained in that bucket. Most
    of the achievable skill is usually here, because the useful information is
    that "ICON says 0.1 in" means something quite different at this location
    than it does on the grid.
    """
    train_bins = pd.cut(train["forecast_in"], bins=list(bins), include_lowest=True)
    rates = train.groupby(train_bins, observed=False)["wet"].mean()
    overall = float(train["wet"].mean())

    test_bins = pd.cut(test["forecast_in"], bins=list(bins), include_lowest=True)
    return test_bins.map(rates).astype(float).fillna(overall).to_numpy()
