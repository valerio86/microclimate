import numpy as np
import pandas as pd
import pytest

from microclimate import shield


def test_no_sun_means_no_error():
    assert shield.radiation_error([0.0], [0.0])[0] == pytest.approx(0.0)


def test_error_grows_with_sunlight():
    errors = shield.radiation_error([0, 250, 500, 1000], [2, 2, 2, 2])
    assert np.all(np.diff(errors) > 0)


def test_error_shrinks_with_ventilation():
    errors = shield.radiation_error([800] * 4, [0, 2, 5, 15])
    assert np.all(np.diff(errors) < 0)


def test_error_is_never_negative():
    # A shield can warm the sensor above ambient; it cannot cool it below.
    errors = shield.radiation_error([-50, 0, 300], [1, 1, 1])
    assert np.all(errors >= 0)


def test_missing_values_are_handled_as_calm_and_dark():
    errors = shield.radiation_error([np.nan, 500], [np.nan, np.nan])
    assert errors[0] == pytest.approx(0.0)  # no sun recorded -> no error
    assert errors[1] > 0  # missing wind treated as calm, the worst case


def test_magnitude_respects_the_tree_transit_upper_bound():
    # The tree-transit experiment found no detectable shield error and bounds it
    # below ~0.75 F per 1000 W/m². Guards against reinstating the earlier
    # cloud-derived 2.9 F, which measured real air cooling rather than the
    # instrument, or the fitted-but-implausible ~8 F.
    worst = shield.radiation_error([1000.0], [0.0])[0]
    assert 0.0 < worst <= 1.0


def test_correct_observations_lowers_daytime_temperature_only():
    station = pd.DataFrame(
        {
            "ts": pd.date_range("2025-06-01", periods=3, freq="1h", tz="UTC"),
            "temp_f": [70.0, 70.0, 70.0],
            "solar_wm2": [0.0, 500.0, 900.0],
            "wind_mph": [1.0, 1.0, 1.0],
        }
    )

    corrected = shield.correct_observations(station)

    assert corrected["temp_f"].iloc[0] == pytest.approx(70.0)  # night untouched
    assert corrected["temp_f"].iloc[1] < 70.0
    assert corrected["temp_f"].iloc[2] < corrected["temp_f"].iloc[1]
    # The adjustment stays visible rather than being silently absorbed.
    assert "shield_error_f" in corrected


def test_correct_observations_passes_through_without_solar():
    station = pd.DataFrame({"temp_f": [70.0], "wind_mph": [3.0]})
    assert shield.correct_observations(station).equals(station)


def test_wind_scale_accounts_for_the_under_reading_anemometer():
    # The same indicated wind means more true airflow, hence less shield error.
    under_reading = shield.radiation_error([800], [3], wind_scale=0.47)[0]
    accurate = shield.radiation_error([800], [3], wind_scale=1.0)[0]
    assert under_reading < accurate
