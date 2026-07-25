"""Quantify how the public forecast misses at this specific location.

Phase 2 of the project: before any model, just measure the gap. A systematic
offset that depends on hour-of-day or season is exactly what a grid-cell
forecast cannot capture about a microclimate, and it is also the easiest thing
to correct — often most of the available win comes from subtracting a
conditional mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_columns(paired: pd.DataFrame) -> list[str]:
    return [column for column in paired.columns if column.endswith("_error")]


def _stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "p90_abs": np.nan}
    return {
        "n": int(clean.size),
        "bias": float(clean.mean()),
        "mae": float(clean.abs().mean()),
        "rmse": float(np.sqrt((clean**2).mean())),
        "p90_abs": float(clean.abs().quantile(0.90)),
    }


def overall(paired: pd.DataFrame) -> pd.DataFrame:
    """Headline error per variable per lead time."""
    rows = []
    for (lead,), group in paired.groupby(["lead_days"]):
        for column in error_columns(group):
            rows.append(
                {
                    "variable": column.removesuffix("_error"),
                    "lead_days": int(lead),
                    **_stats(group[column]),
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "lead_days"]).reset_index(drop=True)


def by_group(paired: pd.DataFrame, group_column: str, variable: str) -> pd.DataFrame:
    """Error for one variable broken out by hour, month, or any other column."""
    error_column = f"{variable}_error"
    if error_column not in paired.columns:
        raise KeyError(f"{error_column} not in paired data")

    rows = []
    for key, group in paired.groupby([group_column, "lead_days"]):
        group_value, lead = key
        rows.append(
            {
                group_column: group_value,
                "lead_days": int(lead),
                **_stats(group[error_column]),
            }
        )
    return pd.DataFrame(rows).sort_values([group_column, "lead_days"]).reset_index(drop=True)


def correction_table(
    paired: pd.DataFrame, variable: str, by: tuple[str, ...] = ("local_hour", "local_month")
) -> pd.DataFrame:
    """A first, deliberately simple bias correction.

    The mean error within each (hour, month, lead) bucket, to be subtracted
    from future forecasts. Naive, but it is the honest baseline that any
    fancier model has to beat — and it is frequently hard to beat by much.
    """
    error_column = f"{variable}_error"
    grouped = paired.groupby([*by, "lead_days"])[error_column]
    table = grouped.agg(["mean", "count"]).reset_index()
    return table.rename(columns={"mean": "correction", "count": "n"})


def apply_correction(
    forecasts: pd.DataFrame,
    table: pd.DataFrame,
    variable: str,
    by: tuple[str, ...] = ("local_hour", "local_month"),
    min_samples: int = 10,
) -> pd.DataFrame:
    """Subtract the learned bias from a forecast frame.

    Buckets with too few samples are left uncorrected rather than trusted —
    a correction fit on three observations is noise.
    """
    trusted = table[table["n"] >= min_samples]
    merged = forecasts.merge(trusted, on=[*by, "lead_days"], how="left")
    merged["correction"] = merged["correction"].fillna(0.0)
    merged[f"{variable}_corrected"] = merged[variable] - merged["correction"]
    return merged
