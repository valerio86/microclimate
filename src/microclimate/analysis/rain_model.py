"""A learned rain-chance model, and the baseline it has to beat.

The baseline is deliberately hard to beat: bucket the forecast total, look up how
often it historically rained in that bucket. That already scores Brier skill 0.31
across folds. Anything here must clear it *across folds*, not on one split.

Two feature families the baseline cannot use:

  multi-model agreement   three models are stored, and whether they agree that
                          it will rain is a direct read on how confident the
                          forecast deserves to be. One model saying 0.1 in while
                          two say nothing is a different situation from all
                          three saying 0.1 in, and the amount alone cannot tell
                          them apart.

  convective share        a shower inside a 13 km cell frequently misses a point
                          sensor; widespread rain does not. `showers` versus
                          total precipitation separates them.

Everything here comes from the forecast or the calendar. No observation from the
day being predicted may appear — that is the whole discipline, and
`FEATURES` is the list to audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import rain

# The frozen model. Chosen on the development folds, then scored once on the
# sealed holdout: Brier skill +0.52 against climatology, versus +0.45 for the
# amount-lookup baseline and +0.46 for the same model given all fifteen features.
#
# Three features beat fifteen. Whether the models agree is very nearly the entire
# signal, and the extra features mostly add variance: an ablation on the folds
# gave +0.41 for these three against +0.39 for the full set, with a better worst
# fold (+0.26 against +0.19). `model_spread_in` enters negative — when the models
# disagree, rain is less likely — which is what it ought to do.
#
# Two predictions of mine died here. Convective share was supposed to be the key
# discriminator for whether a shower lands on a point sensor; removing it changes
# nothing. And gradient boosting was supposed to beat logistic regression; on 336
# training days it does not (+0.33 against +0.39).
FROZEN_FEATURES = ["models_wet", "model_mean_in", "model_spread_in"]

FEATURES = [
    # ICON, the primary model
    "forecast_in",
    "forecast_max_hourly_in",
    "forecast_wet_hours",
    "showers_in",
    "convective_share",
    "forecast_temp_f",
    "forecast_rh",
    "forecast_cloud",
    "forecast_vpd",
    "forecast_wind_mph",
    # Multi-model agreement
    "models_wet",
    "model_mean_in",
    "model_spread_in",
    # Season
    "doy_sin",
    "doy_cos",
]


def daily_features(
    paired_by_source: dict[str, pd.DataFrame],
    timezone: str,
    primary: str,
    lead_days: int = 1,
) -> pd.DataFrame:
    """Build one row per day: forecast-derived features plus the observed label.

    `paired_by_source` maps a forecast source to its paired frame. The primary
    source supplies the label and the detailed features; the others contribute
    only their daily precipitation totals, for agreement.
    """
    primary_frame = paired_by_source[primary]
    primary_frame = primary_frame[primary_frame["lead_days"] == lead_days]

    daily = rain.daily_targets(primary_frame, timezone)
    if daily.empty:
        return daily

    hourly = primary_frame.copy()
    local = pd.to_datetime(hourly["valid_time"], utc=True).dt.tz_convert(timezone)
    hourly["day"] = local.dt.date

    extra = hourly.groupby("day").agg(
        forecast_max_hourly_in=("precip_in", "max"),
        forecast_wet_hours=("precip_in", lambda s: int((s >= 0.01).sum())),
        # Forecast temperature, never the observed value: using what actually
        # happened that day would be a leak wearing a feature's name.
        forecast_temp_f=("temp_f_forecast", "mean"),
        forecast_rh=("rh_forecast", "mean"),
        forecast_cloud=("cloud_cover", "mean"),
        forecast_vpd=("vpd_kpa", "mean"),
        forecast_wind_mph=("wind_mph_forecast", "mean"),
    )
    daily = daily.merge(extra, on="day", how="left")

    # Daily totals from every stored model, for agreement.
    totals = {}
    for source, frame in paired_by_source.items():
        at_lead = frame[frame["lead_days"] == lead_days].copy()
        day = pd.to_datetime(at_lead["valid_time"], utc=True).dt.tz_convert(timezone).dt.date
        totals[source] = at_lead.groupby(day)["precip_in"].sum()

    agreement = pd.DataFrame(totals)
    agreement.index.name = "day"
    daily = daily.merge(
        pd.DataFrame(
            {
                "models_wet": (agreement >= 0.02).sum(axis=1),
                "model_mean_in": agreement.mean(axis=1),
                "model_spread_in": agreement.std(axis=1),
            }
        ).reset_index(),
        on="day",
        how="left",
    )

    day_of_year = pd.to_datetime(daily["day"]).dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    daily["valid_time"] = pd.to_datetime(daily["day"])
    return daily


def fit_predict_gbm(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Gradient-boosted trees. Small, because the training set is small."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=3,
        learning_rate=0.06,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train[FEATURES], train["wet"].astype(int))
    return model.predict_proba(test[FEATURES])[:, 1]


def build_logistic(features: list[str] | None = None):
    """Regularised logistic regression — fewer ways to overfit than the trees."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=2000),
    )


def fit_predict_logistic(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str] | None = None
) -> np.ndarray:
    columns = features or FEATURES
    model = build_logistic(columns)
    model.fit(train[columns], train["wet"].astype(int))
    return model.predict_proba(test[columns])[:, 1]


def fit_predict_frozen(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The model that was frozen and then scored on the sealed holdout."""
    return fit_predict_logistic(train, test, FROZEN_FEATURES)


METHODS = {
    "climatology": lambda train, test: rain.climatological_probability(train, test),
    "amount lookup": lambda train, test: rain.forecast_amount_probability(train, test),
    "frozen": fit_predict_frozen,
    "logistic (all)": fit_predict_logistic,
    "gbm": fit_predict_gbm,
}
