import numpy as np
import pandas as pd
import pytest

from microclimate.analysis import backtest as bt


def synthetic(days: int = 400, bias_fn=None, noise: float = 0.0, seed: int = 0):
    """Paired data with a known, constructed bias so we can check recovery."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=days * 24, freq="1h", tz="UTC")
    frame = pd.DataFrame({"valid_time": times})
    frame["local_hour"] = frame["valid_time"].dt.hour
    frame["local_month"] = frame["valid_time"].dt.month
    frame["local_doy"] = frame["valid_time"].dt.dayofyear
    frame["lead_days"] = 1

    truth = 50 + 10 * np.sin(2 * np.pi * frame["local_doy"] / 365.25)
    error = bias_fn(frame) if bias_fn else np.zeros(len(frame))
    error = error + rng.normal(0, noise, len(frame))

    frame["temp_f_actual"] = truth
    frame["temp_f_forecast"] = truth + error
    frame["temp_f_error"] = error
    return frame


def test_split_is_chronological_and_disjoint():
    data = synthetic(days=100)
    train, test = bt.chronological_split(data, train_fraction=0.7)

    assert train["valid_time"].max() < test["valid_time"].min()
    assert len(train) + len(test) == len(data)


def test_split_honours_an_explicit_boundary():
    data = synthetic(days=100)
    train, test = bt.chronological_split(data, train_end="2024-02-01")

    assert train["valid_time"].max() <= pd.Timestamp("2024-02-01", tz="UTC")
    assert test["valid_time"].min() > pd.Timestamp("2024-02-01", tz="UTC")


def test_constant_corrector_recovers_a_flat_offset():
    data = synthetic(days=200, bias_fn=lambda f: np.full(len(f), 3.0))
    results, _ = bt.evaluate(data, correctors=[bt.ConstantCorrector()])

    row = results[(results.method == "constant") & (results.lead_days == "all")].iloc[0]
    assert row["mae"] == pytest.approx(0.0, abs=0.01)
    assert row["skill"] == pytest.approx(1.0, abs=0.01)  # removed the error entirely


def test_harmonic_recovers_a_diurnal_bias_on_unseen_data():
    # The bias the real data shows: warm at night, cool in the afternoon.
    data = synthetic(
        days=500, bias_fn=lambda f: 2.5 * np.cos(2 * np.pi * f["local_hour"] / 24)
    )
    results, _ = bt.evaluate(data, correctors=[bt.HarmonicCorrector()])

    row = results[(results.method == "harmonic") & (results.lead_days == "all")].iloc[0]
    assert row["skill"] > 0.95, "should nearly eliminate a purely cyclical bias"


def test_constant_corrector_cannot_fix_a_diurnal_bias():
    # Guards the headline-number trap: a single mean has nothing to say about
    # a bias that averages to zero over the day.
    data = synthetic(
        days=500, bias_fn=lambda f: 2.5 * np.cos(2 * np.pi * f["local_hour"] / 24)
    )
    results, _ = bt.evaluate(data, correctors=[bt.ConstantCorrector()])

    row = results[(results.method == "constant") & (results.lead_days == "all")].iloc[0]
    assert abs(row["skill"]) < 0.05


def test_pure_noise_yields_no_real_skill():
    # The central guard: with nothing to learn, a flexible model must not be
    # rewarded. Scored in-sample, buckets would show spurious positive skill.
    data = synthetic(days=500, bias_fn=None, noise=3.0)
    results, _ = bt.evaluate(
        data, correctors=[bt.BucketCorrector(), bt.HarmonicCorrector()]
    )

    for method in ("buckets", "harmonic"):
        row = results[(results.method == method) & (results.lead_days == "all")].iloc[0]
        assert row["skill"] < 0.05, f"{method} found skill in pure noise"


def test_raw_forecast_scores_exactly_zero_skill():
    data = synthetic(days=200, bias_fn=lambda f: np.full(len(f), 2.0))
    results, _ = bt.evaluate(data)

    row = results[(results.method == "raw forecast") & (results.lead_days == "all")].iloc[0]
    assert row["skill"] == pytest.approx(0.0, abs=1e-9)


def test_bucket_corrector_falls_back_when_a_cell_is_sparse():
    data = synthetic(days=300, bias_fn=lambda f: np.full(len(f), 4.0))
    train, test = bt.chronological_split(data, train_fraction=0.7)

    corrector = bt.BucketCorrector(min_samples=10_000).fit(train, "temp_f_error")
    predicted = corrector.predict(test)

    # Every cell is below the threshold, so all predictions use the lead mean.
    assert np.allclose(predicted, 4.0, atol=0.01)


def test_harmonic_design_shape_and_interaction_terms():
    hours = np.arange(24, dtype=float)
    doy = np.full(24, 100.0)

    with_interaction = bt.harmonic_design(hours, doy, 2, 2, interaction=True)
    without = bt.harmonic_design(hours, doy, 2, 2, interaction=False)

    assert without.shape == (24, 1 + 4 + 4)
    assert with_interaction.shape == (24, 1 + 4 + 4 + 4)


def test_evaluate_rejects_an_empty_split():
    data = synthetic(days=10)
    with pytest.raises(ValueError):
        bt.evaluate(data, train_end="2030-01-01")
