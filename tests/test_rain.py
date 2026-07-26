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


def test_outage_days_are_dropped_not_treated_as_dry():
    # The failure this guards: a blocked gauge produces confident wrong labels,
    # which is worse for a model than having no data at all.
    frame = hourly(days=6, start="2025-06-02")
    daily = rain.daily_targets(frame, "UTC")
    assert daily.empty, "June 2025 falls inside the known outage"


def test_days_outside_the_outage_survive():
    frame = hourly(days=6, start="2025-10-01")
    assert not rain.daily_targets(frame, "UTC").empty


def test_in_outage_boundaries_are_half_open():
    days = pd.Series(
        pd.to_datetime(["2025-05-31", "2025-06-01", "2025-09-05", "2025-09-06"])
    )
    assert rain.in_outage(days).tolist() == [False, True, True, False]


def test_gauge_health_flags_a_blocked_month():
    frame = hourly(days=31, start="2025-10-01")
    frame["temp_f_actual"] = 60.0
    frame["precip_in"] = 0.0
    # Forecast rain on many hours; gauge never tips.
    frame.loc[frame.index % 17 == 0, "precip_in"] = 0.05

    health = rain.gauge_health(frame, "UTC")
    assert (health["verdict"] == "BLOCKED").any()


def test_gauge_health_passes_a_working_month():
    frame = hourly(days=31, start="2025-10-01")
    frame["temp_f_actual"] = 60.0
    wet = frame.index % 17 == 0
    frame.loc[wet, "precip_in"] = 0.05
    frame.loc[wet, "rain_in"] = 0.04

    health = rain.gauge_health(frame, "UTC")
    assert (health["verdict"] == "ok").all()


def test_gauge_health_ignores_cold_months():
    # A freezing month with no tips is snow blindness, not a fault, and must
    # not be reported as a blockage.
    frame = hourly(days=31, start="2025-10-01")
    frame["temp_f_actual"] = 20.0
    frame.loc[frame.index % 17 == 0, "precip_in"] = 0.05

    assert rain.gauge_health(frame, "UTC").empty


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
