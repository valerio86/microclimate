"""How right has this thing been?

A history view showing last week's weather would duplicate every weather app in
existence. What none of them can show is whether *this* system's corrections and
probabilities actually held up. So this is a verification log rather than a
record of conditions.

It matters for two reasons. It earns trust that is deserved rather than assumed —
and it fails visibly. If the rain gauge clogs again, or the station drifts, or a
model changes upstream, the verification panel degrades where you can see it,
instead of the corrections quietly learning from broken ground truth.

**Honest by construction.** Every prediction shown is reconstructed from the
archived lead-1 forecast, using models fitted only on data from *before* the
window. Nothing here is a replay of values the model was fitted to, which would
look excellent and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import backtest, frost, rain, rain_model

DEFAULT_WINDOW_DAYS = 30


def verification(
    paired_by_source: dict[str, pd.DataFrame],
    timezone: str,
    primary: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict:
    """Reconstruct what would have been predicted, and what happened.

    Returns `nights` (overnight minima, predicted and actual), `rain` (daily
    probability against outcome) and `summary` (skill over the window).
    """
    primary_frame = paired_by_source[primary]
    at_lead = primary_frame[primary_frame["lead_days"] == 1]
    if at_lead.empty:
        return {"nights": pd.DataFrame(), "rain": pd.DataFrame(), "summary": {}}

    times = pd.to_datetime(at_lead["valid_time"], utc=True)
    cutoff = times.max() - pd.Timedelta(days=window_days)

    train = at_lead[times < cutoff]
    test = at_lead[times >= cutoff]
    if train.empty or test.empty:
        return {"nights": pd.DataFrame(), "rain": pd.DataFrame(), "summary": {}}

    # --- temperature: corrected overnight minima against actual ---
    usable = train.dropna(subset=["temp_f_error"])
    nights = pd.DataFrame()
    if not usable.empty:
        corrector = backtest.RegimeCorrector().fit(usable, "temp_f_error")
        nights = frost.nightly_minima(test, timezone, corrections=corrector.predict(test))

    # --- rain: probability against outcome ---
    daily = rain_model.daily_features(
        paired_by_source, timezone, primary=primary, lead_days=1
    )
    rain_rows = pd.DataFrame()
    if not daily.empty:
        day_times = pd.to_datetime(daily["valid_time"], utc=True)
        rain_train = daily[day_times < cutoff]
        rain_test = daily[day_times >= cutoff]
        if not rain_train.empty and not rain_test.empty:
            rain_rows = rain_test.copy()
            rain_rows["rain_chance"] = rain_model.fit_predict_frozen(
                rain_train, rain_test
            ).clip(0.01, 0.99)

    return {
        "nights": nights,
        "rain": rain_rows,
        "summary": _summarise(nights, rain_rows, window_days),
    }


def _summarise(nights: pd.DataFrame, rain_rows: pd.DataFrame, window_days: int) -> dict:
    """Headline accuracy over the window, with the honest comparison alongside.

    The corrected error alone means little; what matters is whether it beat the
    public forecast it was meant to improve on.
    """
    summary = {"window_days": window_days}

    if not nights.empty:
        raw_error = (nights["raw_min"] - nights["actual_min"]).abs()
        corrected_error = (nights["corrected_min"] - nights["actual_min"]).abs()
        summary["nights"] = int(len(nights))
        summary["raw_low_mae"] = float(raw_error.mean())
        summary["corrected_low_mae"] = float(corrected_error.mean())
        summary["low_improved"] = bool(corrected_error.mean() < raw_error.mean())

        # Frost calls, at the verified 34 °F warning threshold.
        predicted = nights["corrected_min"] <= 34.0
        observed = nights["actual_min"] <= 32.0
        summary["frost_nights"] = int(observed.sum())
        summary["frost_caught"] = int((predicted & observed).sum())
        summary["frost_missed"] = int((~predicted & observed).sum())
        summary["frost_false_alarms"] = int((predicted & ~observed).sum())

    if not rain_rows.empty and "rain_chance" in rain_rows:
        outcome = rain_rows["wet"].to_numpy(dtype=float)
        probability = rain_rows["rain_chance"].to_numpy(dtype=float)
        summary["rain_days"] = int(len(rain_rows))
        summary["rain_wet_days"] = int(outcome.sum())
        summary["rain_brier"] = rain.brier_score(probability, outcome)
        # Undefined when every day in the window went the same way, since
        # climatology is then perfect and there is nothing to improve on.
        summary["rain_skill"] = (
            rain.brier_skill(probability, outcome) if 0 < outcome.mean() < 1 else None
        )

    return summary
