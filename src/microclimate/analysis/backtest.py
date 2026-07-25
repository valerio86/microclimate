"""Does a correction actually help? — honest evaluation.

Any correction fitted and then scored on the same data will look like an
improvement, and the more freedom it has the better it will look. A table of
288 per-bucket means will "beat" a 13-parameter curve every time on its own
training data, purely by memorising noise. So nothing here is scored on data it
was fitted to.

The split is **chronological**, never random. Randomly holding out hours would
put 2pm in the training set and 3pm of the same afternoon in the test set —
they share the same weather, so the model would be graded on conditions it had
effectively already seen. Fitting on earlier data and scoring on later data is
also the question we actually care about: does last year's correction improve
tomorrow's forecast?

Everything is reported as a **skill score** against the raw forecast:

    skill = 1 - MAE(corrected) / MAE(raw)

Positive means the correction helped. Zero means it achieved nothing. Negative
means we made the public forecast worse, which is a real possible outcome and
the reason this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The error column convention from align.pair(): forecast - actual.
# A corrector predicts that error; subtracting it from the forecast corrects it.


def chronological_split(
    paired: pd.DataFrame, train_fraction: float = 0.7, train_end: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) by time, never at random.

    `train_end` (an ISO date) pins the boundary explicitly; otherwise the
    earliest `train_fraction` of the time span becomes the training set.
    """
    frame = paired.sort_values("valid_time")
    times = pd.to_datetime(frame["valid_time"], utc=True)

    if train_end is not None:
        boundary = pd.Timestamp(train_end, tz="UTC")
    else:
        span = times.max() - times.min()
        boundary = times.min() + span * train_fraction

    return frame[times <= boundary].copy(), frame[times > boundary].copy()


# --------------------------------------------------------------------------
# Correctors: each learns to predict the forecast error, per lead time.
# --------------------------------------------------------------------------


class Corrector:
    name = "corrector"

    def fit(self, train: pd.DataFrame, error_column: str) -> Corrector:
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ConstantCorrector(Corrector):
    """A single mean error per lead — the dashboard headline, as a model.

    Included precisely because it is the naive thing, so everything else has
    something honest to beat.
    """

    name: str = "constant"
    _means: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, error_column: str) -> ConstantCorrector:
        self._means = train.groupby("lead_days")[error_column].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["lead_days"].map(self._means).fillna(0.0).to_numpy()


@dataclass
class BucketCorrector(Corrector):
    """Mean error per (hour, month, lead) cell.

    Flexible, but each cell is estimated from ~30 observations, and cells with
    fewer than `min_samples` fall back to the lead's overall mean rather than
    trusting a handful of points.
    """

    min_samples: int = 20
    name: str = "buckets"
    _table: pd.DataFrame | None = None
    _fallback: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, error_column: str) -> BucketCorrector:
        grouped = train.groupby(["local_hour", "local_month", "lead_days"])[error_column]
        table = grouped.agg(["mean", "count"]).reset_index()
        self._table = table[table["count"] >= self.min_samples].drop(columns="count")
        self._fallback = train.groupby("lead_days")[error_column].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = ["local_hour", "local_month", "lead_days"]
        merged = frame[keys].merge(self._table, on=keys, how="left")
        fallback = frame["lead_days"].map(self._fallback).fillna(0.0).to_numpy()
        return merged["mean"].fillna(pd.Series(fallback, index=merged.index)).to_numpy()


def harmonic_design(
    hours: np.ndarray,
    doy: np.ndarray,
    n_diurnal: int = 2,
    n_annual: int = 2,
    interaction: bool = True,
) -> np.ndarray:
    """Smooth cyclical basis for hour-of-day and day-of-year.

    Replaces 288 independently estimated cells with a handful of coefficients
    that every observation informs. It also enforces something the buckets
    cannot: 23:00 and 00:00 are adjacent, and December and January are
    adjacent. The interaction terms let the size of the daily swing vary
    through the year, which the data says it does.
    """
    columns = [np.ones(len(hours))]
    for k in range(1, n_diurnal + 1):
        columns += [np.sin(2 * np.pi * k * hours / 24), np.cos(2 * np.pi * k * hours / 24)]
    for j in range(1, n_annual + 1):
        columns += [
            np.sin(2 * np.pi * j * doy / 365.25),
            np.cos(2 * np.pi * j * doy / 365.25),
        ]
    if interaction:
        sh, ch = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
        sa, ca = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        columns += [sh * sa, sh * ca, ch * sa, ch * ca]
    return np.column_stack(columns)


@dataclass
class HarmonicCorrector(Corrector):
    """Least-squares fit of the cyclical basis, one model per lead time."""

    n_diurnal: int = 2
    n_annual: int = 2
    interaction: bool = True
    name: str = "harmonic"
    _coefficients: dict = field(default_factory=dict)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        return harmonic_design(
            frame["local_hour"].to_numpy(dtype=float),
            frame["local_doy"].to_numpy(dtype=float),
            self.n_diurnal,
            self.n_annual,
            self.interaction,
        )

    def fit(self, train: pd.DataFrame, error_column: str) -> HarmonicCorrector:
        for lead, group in train.groupby("lead_days"):
            clean = group.dropna(subset=[error_column])
            if clean.empty:
                continue
            design = self._design(clean)
            target = clean[error_column].to_numpy(dtype=float)
            self._coefficients[lead], *_ = np.linalg.lstsq(design, target, rcond=None)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame))
        for lead, group in frame.groupby("lead_days"):
            coefficients = self._coefficients.get(lead)
            if coefficients is None:
                continue
            out[frame["lead_days"].to_numpy() == lead] = self._design(group) @ coefficients
        return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _metrics(errors: np.ndarray) -> dict:
    clean = errors[~np.isnan(errors)]
    if clean.size == 0:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan}
    return {
        "n": int(clean.size),
        "bias": float(clean.mean()),
        "mae": float(np.abs(clean).mean()),
        "rmse": float(np.sqrt((clean**2).mean())),
    }


def persistence_errors(test: pd.DataFrame, variable: str) -> np.ndarray:
    """Error of 'tomorrow will be like today' — the classic free baseline.

    For a forecast valid at time t with lead L days, this uses the observation
    from t - L days, which really was available when that forecast was issued.
    """
    actual_column = f"{variable}_actual"
    observed = (
        test[["valid_time", actual_column]]
        .dropna()
        .drop_duplicates(subset="valid_time")
        .set_index("valid_time")[actual_column]
    )

    out = np.full(len(test), np.nan)
    for lead, group in test.groupby("lead_days"):
        lagged_time = pd.to_datetime(group["valid_time"], utc=True) - pd.Timedelta(
            days=int(lead) if lead > 0 else 1
        )
        predicted = lagged_time.map(observed).to_numpy(dtype=float)
        out[test["lead_days"].to_numpy() == lead] = predicted - group[actual_column].to_numpy(
            dtype=float
        )
    return out


def evaluate(
    paired: pd.DataFrame,
    variable: str = "temp_f",
    train_fraction: float = 0.7,
    train_end: str | None = None,
    correctors: list[Corrector] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit every corrector on the training period, score on the held-out period.

    Returns (results, info). Results carry one row per (method, lead) plus an
    'all' lead summary, with a skill score against the raw forecast.
    """
    error_column = f"{variable}_error"
    if error_column not in paired.columns:
        raise KeyError(f"{error_column} not in paired data")

    usable = paired.dropna(subset=[error_column])
    train, test = chronological_split(usable, train_fraction, train_end)
    if train.empty or test.empty:
        raise ValueError("Split produced an empty train or test set")

    if correctors is None:
        correctors = [ConstantCorrector(), BucketCorrector(), HarmonicCorrector()]

    raw = test[error_column].to_numpy(dtype=float)
    columns: dict[str, np.ndarray] = {"raw forecast": raw}

    for corrector in correctors:
        corrector.fit(train, error_column)
        # Corrected error is the raw error minus the correction we predicted.
        columns[corrector.name] = raw - corrector.predict(test)

    columns["persistence"] = persistence_errors(test, variable)

    leads = sorted(test["lead_days"].unique())
    rows = []
    for name, errors in columns.items():
        for lead in [*leads, "all"]:
            mask = (
                np.ones(len(test), dtype=bool)
                if lead == "all"
                else test["lead_days"].to_numpy() == lead
            )
            stats = _metrics(errors[mask])
            reference = _metrics(raw[mask])["mae"]
            skill = (
                np.nan
                if not reference or np.isnan(stats["mae"])
                else 1 - stats["mae"] / reference
            )
            rows.append({"method": name, "lead_days": lead, **stats, "skill": skill})

    info = {
        "variable": variable,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": str(train["valid_time"].min()),
        "train_end": str(train["valid_time"].max()),
        "test_start": str(test["valid_time"].min()),
        "test_end": str(test["valid_time"].max()),
    }
    return pd.DataFrame(rows), info
