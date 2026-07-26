import numpy as np
import pandas as pd
import pytest

from microclimate.analysis import rain, rain_model as rm


def test_no_feature_is_derived_from_the_observed_day():
    """The audit that matters: nothing named for an observation may be a feature.

    Using the day's own measurements to predict whether it rained would score
    beautifully and mean nothing.
    """
    # Precise rather than blunt: `forecast_wet_hours` is a count of *forecast*
    # wet hours and is perfectly legitimate, so a bare "wet" substring test
    # would fire on it and get quietly deleted the first time it did.
    forbidden_exact = {"rain_in", "observed_in", "wet", "usable_hours", "mean_temp_f"}
    for name in rm.FEATURES + rm.FROZEN_FEATURES:
        assert not name.endswith("_actual"), name
        assert name not in forbidden_exact, name
        assert "observed" not in name, name


def test_frozen_features_are_a_subset_of_the_full_set():
    assert set(rm.FROZEN_FEATURES) <= set(rm.FEATURES)


def synthetic_daily(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Days where rain probability genuinely rises with model agreement."""
    rng = np.random.default_rng(seed)
    models_wet = rng.integers(0, 4, n)
    spread = rng.uniform(0, 0.3, n)
    probability = np.clip(0.05 + 0.30 * models_wet - 0.4 * spread, 0.01, 0.99)

    frame = pd.DataFrame(
        {
            "valid_time": pd.date_range("2024-01-01", periods=n, freq="1D"),
            "models_wet": models_wet,
            "model_mean_in": models_wet * 0.05 + rng.uniform(0, 0.02, n),
            "model_spread_in": spread,
            "wet": rng.uniform(0, 1, n) < probability,
        }
    )
    for column in rm.FEATURES:
        if column not in frame:
            frame[column] = rng.normal(0, 1, n)
    return frame


def test_frozen_model_beats_climatology_on_learnable_data():
    data = synthetic_daily()
    train, test = data.iloc[:200], data.iloc[200:]
    outcome = test["wet"].to_numpy(float)

    predicted = rm.fit_predict_frozen(train, test)
    assert rain.brier_skill(predicted, outcome) > 0.2


def test_frozen_model_finds_no_skill_in_noise():
    # The guard against a model that always looks good: with the label
    # independent of every feature, skill must be near zero.
    rng = np.random.default_rng(3)
    data = synthetic_daily(300, seed=1)
    data["wet"] = rng.uniform(0, 1, len(data)) < 0.4

    train, test = data.iloc[:200], data.iloc[200:]
    predicted = rm.fit_predict_frozen(train, test)
    assert rain.brier_skill(predicted, test["wet"].to_numpy(float)) < 0.10


def test_predictions_are_probabilities():
    data = synthetic_daily()
    predicted = rm.fit_predict_frozen(data.iloc[:200], data.iloc[200:])
    assert ((predicted >= 0) & (predicted <= 1)).all()


def test_missing_feature_values_do_not_break_prediction():
    data = synthetic_daily()
    data.loc[data.index[:20], "model_spread_in"] = np.nan
    predicted = rm.fit_predict_frozen(data.iloc[:200], data.iloc[200:])
    assert np.isfinite(predicted).all()


def test_every_method_returns_one_probability_per_test_row():
    data = synthetic_daily()
    train, test = data.iloc[:200], data.iloc[200:]
    for name, method in rm.METHODS.items():
        if name == "amount lookup":
            continue  # needs forecast_in, covered in test_rain
        predicted = method(train, test)
        assert len(predicted) == len(test), name
        assert np.isfinite(predicted).all(), name
