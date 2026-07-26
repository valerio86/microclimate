from datetime import date

import numpy as np
import pandas as pd
import pytest

from microclimate import alerts


def nights(rows):
    return pd.DataFrame(rows, columns=["day", "corrected_min", "raw_min"])


def days(rows):
    return pd.DataFrame(
        rows, columns=["day", "rain_chance", "model_mean_in", "models_wet", "model_spread_in"]
    )


def test_silence_is_the_default():
    quiet_nights = nights([(date(2026, 5, 1), 52.0, 54.0)])
    quiet_days = days([(date(2026, 5, 1), 0.20, 0.05, 1, 0.02)])
    assert alerts.evaluate(quiet_nights, quiet_days) == []


def test_frost_warning_above_freezing():
    out = alerts.frost_alerts(nights([(date(2026, 5, 1), 33.0, 37.0)]))
    assert len(out) == 1
    assert out[0].severity == "warning"
    assert "Frost risk" in out[0].headline


def test_freeze_and_hard_freeze_are_critical():
    out = alerts.frost_alerts(
        nights([(date(2026, 5, 1), 31.0, 35.0), (date(2026, 5, 2), 25.0, 30.0)])
    )
    assert [a.severity for a in out] == ["critical", "critical"]
    assert "Hard freeze" in out[1].headline


def test_no_frost_alert_comfortably_above_the_threshold():
    assert alerts.frost_alerts(nights([(date(2026, 5, 1), 36.0, 39.0)])) == []


def test_frost_alert_names_the_public_forecast_too():
    out = alerts.frost_alerts(nights([(date(2026, 5, 1), 31.0, 37.0)]))
    assert "37" in out[0].detail, "must show what the public forecast said"


def test_rain_alert_needs_confidence_or_volume():
    assert alerts.rain_alerts(days([(date(2026, 5, 1), 0.55, 0.10, 2, 0.05)])) == []
    likely = alerts.rain_alerts(days([(date(2026, 5, 1), 0.85, 0.10, 3, 0.02)]))
    assert len(likely) == 1


def test_heavy_rain_alerts_even_when_uncertain():
    # Worth knowing about a possible soaking even at moderate confidence.
    out = alerts.rain_alerts(days([(date(2026, 5, 1), 0.50, 0.90, 2, 0.40)]))
    assert len(out) == 1
    assert "possible" in out[0].headline


def test_heavy_and_likely_is_a_warning_not_an_info():
    out = alerts.rain_alerts(days([(date(2026, 5, 1), 0.90, 0.80, 3, 0.05)]))
    assert out[0].severity == "warning"


def test_deviation_needs_to_exceed_the_noise_floor():
    # Cross-validated temperature skill is 0.06 +/- 0.08, so a 2 F disagreement
    # means nothing and must not be announced.
    assert alerts.deviation_alerts(nights([(date(2026, 5, 1), 45.0, 47.0)])) == []
    out = alerts.deviation_alerts(nights([(date(2026, 5, 1), 42.0, 47.0)]))
    assert len(out) == 1
    assert "colder" in out[0].headline


def test_deviation_reports_warmer_too():
    out = alerts.deviation_alerts(nights([(date(2026, 5, 1), 50.0, 45.0)]))
    assert "warmer" in out[0].headline


def test_frost_suppresses_a_deviation_note_for_the_same_night():
    # If it is going to freeze, that is the headline; the reason is detail.
    out = alerts.evaluate(
        nights([(date(2026, 5, 1), 30.0, 38.0)]),
        days([(date(2026, 5, 1), 0.1, 0.0, 0, 0.0)]),
    )
    kinds = [a.kind for a in out]
    assert "frost" in kinds
    assert "deviation" not in kinds


def test_ordering_is_severity_then_date():
    out = alerts.evaluate(
        nights([(date(2026, 5, 3), 25.0, 33.0), (date(2026, 5, 1), 55.0, 58.0)]),
        days(
            [
                (date(2026, 5, 1), 0.95, 0.10, 3, 0.01),
                (date(2026, 5, 2), 0.95, 0.90, 3, 0.02),
            ]
        ),
    )
    severities = [alerts.SEVERITY_ORDER[a.severity] for a in out]
    assert severities == sorted(severities)
    assert out[0].kind == "frost"


def test_missing_values_are_skipped_not_crashed_on():
    out = alerts.evaluate(
        nights([(date(2026, 5, 1), np.nan, 40.0)]),
        days([(date(2026, 5, 1), np.nan, np.nan, 0, np.nan)]),
    )
    assert out == []
