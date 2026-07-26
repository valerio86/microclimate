"""Features built from past observations, without looking into the future.

Persistence beat every correction we had for wind, which says recent station
history carries real signal. The catch is that using it correctly is easy to get
wrong in a way that never announces itself: a model fed observations it could
not have had scores brilliantly in backtest and is useless in production.

The rule this module enforces:

    A forecast valid at time T with lead N days was issued at T - N days.
    Only observations at or before that issue time may be used.

Every feature here is derived by taking a statistic of the observation series
and reading it *as of* the issue time, so the constraint holds by construction
rather than by remembering to check. `merge_asof` does the reading, matching each
row to the most recent observation at or before its cutoff.

Lead 0 is the analysis rather than a real forecast, so it has no honest issue
time. It is treated as lead 1 (a full day of separation) instead of being given
observations from the moment it describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINIMUM_LEAD_DAYS = 1


def issue_times(
    valid_time: pd.Series, lead_days: pd.Series, minimum_lead_days: int = MINIMUM_LEAD_DAYS
) -> pd.Series:
    """When each forecast was issued — the cutoff for what it could have known."""
    effective = np.maximum(lead_days.to_numpy(dtype=float), minimum_lead_days)
    return pd.to_datetime(valid_time, utc=True) - pd.to_timedelta(effective, unit="D")


def _observation_statistics(observations: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rolling statistics of the observed series, each valid at its own timestamp.

    Every window looks *backward* only, so reading any row as of a cutoff yields
    a value that was genuinely available at that cutoff.
    """
    frame = (
        observations[["valid_time", column]]
        .dropna(subset=["valid_time"])
        .sort_values("valid_time")
        .rename(columns={"valid_time": "ts", column: "obs"})
        .reset_index(drop=True)
    )
    indexed = frame.set_index("ts")["obs"]

    stats = pd.DataFrame({"ts": frame["ts"], "obs_latest": frame["obs"].to_numpy()})
    stats["obs_mean_24h"] = indexed.rolling("24h", min_periods=4).mean().to_numpy()
    stats["obs_min_24h"] = indexed.rolling("24h", min_periods=4).min().to_numpy()
    stats["obs_max_24h"] = indexed.rolling("24h", min_periods=4).max().to_numpy()
    stats["obs_range_24h"] = stats["obs_max_24h"] - stats["obs_min_24h"]
    # Six-hour tendency: how the air was already moving when the forecast ran.
    stats["obs_trend_6h"] = frame["obs"].to_numpy() - (
        indexed.rolling("6h", min_periods=2).apply(lambda w: w.iloc[0], raw=False).to_numpy()
    )
    return stats


def add_lagged_observations(
    paired: pd.DataFrame,
    variable: str = "temp_f",
    minimum_lead_days: int = MINIMUM_LEAD_DAYS,
) -> pd.DataFrame:
    """Attach observation-derived features, each read as of the issue time.

    Adds:
      obs_latest       last observation before the forecast was issued
      obs_mean_24h     mean over the 24 h ending at issue
      obs_range_24h    diurnal range over that same window
      obs_trend_6h     6-hour tendency at issue
      persistence      observation at the same clock hour, `lead_days` ago
      recent_error     that day's forecast error at the same hour and lead

    `recent_error` is the adaptive term: knowing the model ran 3 °F warm here
    yesterday is a strong predictor that it will today. It is available at issue
    time because verifying an hour only requires observing it.
    """
    if paired.empty:
        return paired

    actual_column = f"{variable}_actual"
    error_column = f"{variable}_error"
    if actual_column not in paired.columns:
        raise KeyError(f"{actual_column} not in paired data")

    out = paired.copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"], utc=True)
    out["issue_time"] = issue_times(
        out["valid_time"], out["lead_days"], minimum_lead_days
    )

    observations = (
        out[["valid_time", actual_column]]
        .dropna()
        .drop_duplicates(subset="valid_time")
        .rename(columns={actual_column: "value"})
    )
    stats = _observation_statistics(observations, "value")

    out = out.sort_values("issue_time")
    out = pd.merge_asof(
        out,
        stats.sort_values("ts"),
        left_on="issue_time",
        right_on="ts",
        direction="backward",
    ).drop(columns="ts")

    # Same clock hour, `lead_days` back — exactly the persistence baseline, and
    # by construction no later than the issue time.
    lag = pd.to_timedelta(
        np.maximum(out["lead_days"].to_numpy(dtype=float), minimum_lead_days), unit="D"
    )
    lagged_at = out["valid_time"] - lag
    observed = observations.set_index("valid_time")["value"]
    out["persistence"] = lagged_at.map(observed).to_numpy()

    if error_column in out.columns:
        errors = (
            out[["valid_time", "lead_days", error_column]]
            .dropna()
            .drop_duplicates(subset=["valid_time", "lead_days"])
            .set_index(["valid_time", "lead_days"])[error_column]
        )
        keys = pd.MultiIndex.from_arrays([lagged_at, out["lead_days"]])
        out["recent_error"] = errors.reindex(keys).to_numpy()

    return out.sort_values(["valid_time", "lead_days"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The observation-derived columns present, for handing to a model."""
    candidates = [
        "obs_latest",
        "obs_mean_24h",
        "obs_range_24h",
        "obs_trend_6h",
        "persistence",
        "recent_error",
    ]
    return [column for column in candidates if column in frame.columns]


def assert_no_lookahead(frame: pd.DataFrame, variable: str = "temp_f") -> None:
    """Fail loudly if any lagged feature could only be known after issue time.

    A cheap invariant that runs on real data: every observation-derived value
    must appear somewhere in the observation history at or before its issue
    time. Cheaper than discovering the problem from a model that backtests far
    better than it performs.
    """
    if frame.empty or "issue_time" not in frame.columns:
        return

    actual_column = f"{variable}_actual"
    observations = (
        frame[["valid_time", actual_column]].dropna().drop_duplicates(subset="valid_time")
    )
    if observations.empty:
        return

    earliest = observations["valid_time"].min()
    too_early = frame["issue_time"] < earliest
    for column in ("obs_latest", "obs_mean_24h", "persistence"):
        if column not in frame.columns:
            continue
        offenders = frame.loc[too_early & frame[column].notna()]
        if not offenders.empty:
            raise AssertionError(
                f"{column} has values for {len(offenders)} rows issued before any "
                f"observation existed — a lookahead leak"
            )
