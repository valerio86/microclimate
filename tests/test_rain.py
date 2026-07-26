import numpy as np
import pandas as pd
import pytest

from microclimate.analysis import rain


def hourly(days: int = 30, start: str = "2025-05-01"):
    times = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "valid_time": times,
            "lead_days": 1,
            "rain_in": 0.0,
            "precip_in": 0.0,
            "showers_in": 0.0,
            "snowfall_in": 0.0,
            "temp_f_actual": 55.0,
        }
    )


def test_snow_hours_are_excluded():
    frame = hourly(days=2)
    frame.loc[0:5, "snowfall_in"] = 0.4
    usable = rain.usable_mask(frame)
    assert not usable.iloc[0:6].any()
    assert usable.iloc[6:].all()


def test_cold_hours_are_excluded_even_without_forecast_snow():
    frame = hourly(days=2)
    frame.loc[0:5, "temp_f_actual"] = 28.0
    assert not rain.usable_mask(frame).iloc[0:6].any()


def test_warm_hours_are_kept():
    assert rain.usable_mask(hourly(days=2)).all()


def test_daily_target_marks_wet_days():
    frame = hourly(days=5)
    day2 = frame["valid_time"].dt.date == pd.Timestamp("2025-05-02").date()
    frame.loc[day2, "rain_in"] = 0.02  # spread across the day

    daily = rain.daily_targets(frame, "UTC")
    wet_days = daily.loc[daily["wet"], "day"].tolist()

    assert pd.Timestamp("2025-05-02").date() in wet_days
    assert daily["wet"].sum() == 1


def test_a_trace_below_threshold_is_not_wet():
    frame = hourly(days=3)
    frame.loc[0, "rain_in"] = 0.005
    daily = rain.daily_targets(frame, "UTC")
    assert not daily["wet"].any()


def test_days_with_any_snow_hour_are_dropped_entirely():
    # A partial day would sum only the liquid hours and mislabel the day dry.
    frame = hourly(days=4)
    snowy = frame["valid_time"].dt.date == pd.Timestamp("2025-05-02").date()
    frame.loc[snowy & (frame.index % 24 == 3), "snowfall_in"] = 0.2

    daily = rain.daily_targets(frame, "UTC")
    assert pd.Timestamp("2025-05-02").date() not in daily["day"].tolist()
    assert len(daily) >= 2


def test_convective_share_is_bounded_and_meaningful():
    frame = hourly(days=3)
    frame.loc[0:23, "precip_in"] = 0.1
    frame.loc[0:23, "showers_in"] = 0.1  # entirely convective
    daily = rain.daily_targets(frame, "UTC")
    assert daily["convective_share"].between(0, 1).all()
    assert daily["convective_share"].iloc[0] == pytest.approx(1.0)


def test_brier_score_rewards_confident_correctness():
    outcome = np.array([1, 1, 0, 0])
    assert rain.brier_score(np.array([1.0, 1.0, 0.0, 0.0]), outcome) == pytest.approx(0.0)
    assert rain.brier_score(np.array([0.5, 0.5, 0.5, 0.5]), outcome) == pytest.approx(0.25)
    assert rain.brier_score(np.array([0.0, 0.0, 1.0, 1.0]), outcome) == pytest.approx(1.0)


def test_brier_skill_is_zero_for_climatology():
    outcome = np.array([1, 0, 1, 0, 0, 0])
    climatology = np.full(6, outcome.mean())
    assert rain.brier_skill(climatology, outcome) == pytest.approx(0.0)


def test_brier_skill_is_negative_for_a_worse_than_climatology_forecast():
    outcome = np.array([1, 0, 1, 0])
    assert rain.brier_skill(np.array([0.0, 1.0, 0.0, 1.0]), outcome) < 0


def test_reliability_recovers_a_calibrated_forecast():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0, 1, 20000)
    outcome = (rng.uniform(0, 1, 20000) < probability).astype(float)

    table = rain.reliability(probability, outcome, bins=10)
    # A perfectly calibrated forecast sits on the diagonal.
    assert np.allclose(table["mean_predicted"], table["observed_rate"], atol=0.03)


def test_reliability_exposes_overconfidence():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 1, 20000)
    outcome = (rng.uniform(0, 1, 20000) < truth).astype(float)
    overconfident = np.clip((truth - 0.5) * 2.0 + 0.5, 0, 1)

    table = rain.reliability(overconfident, outcome, bins=10)
    low = table[table["mean_predicted"] < 0.2]
    assert (low["observed_rate"] > low["mean_predicted"]).all()


def test_forecast_amount_probability_learns_from_training_only():
    train = pd.DataFrame({"forecast_in": [0.0, 0.0, 0.1, 0.1], "wet": [False, False, True, True]})
    test = pd.DataFrame({"forecast_in": [0.0, 0.1]})

    probability = rain.forecast_amount_probability(train, test)
    assert probability[0] == pytest.approx(0.0)
    assert probability[1] == pytest.approx(1.0)


def test_unseen_forecast_bucket_falls_back_to_the_base_rate():
    train = pd.DataFrame({"forecast_in": [0.0, 0.0], "wet": [False, True]})
    test = pd.DataFrame({"forecast_in": [5.0]})
    assert rain.forecast_amount_probability(train, test)[0] == pytest.approx(0.5)
