from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microclimate import dashboard
from microclimate.alerts import Alert


def station_frame():
    times = pd.date_range("2026-05-01", periods=12, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "ts": times,
            "temp_f": np.linspace(50, 55, 12),
            "rh": np.linspace(70, 80, 12),
            "wind_mph": 3.0,
            "rain_daily_in": np.linspace(0, 0.12, 12),
        }
    )


def payload(**overrides):
    times = pd.date_range("2026-05-01", periods=24, freq="1h", tz="UTC")
    base = dict(
        station=station_frame(),
        nights=pd.DataFrame(
            [{"day": date(2026, 5, 1), "raw_min": 38.0, "corrected_min": 33.5}]
        ),
        hourly=pd.DataFrame(
            {
                "valid_time": times,
                "temp_f_forecast": np.linspace(40, 60, 24),
                "corrected": np.linspace(38, 58, 24),
            }
        ),
        rain_days=pd.DataFrame(
            [
                {
                    "day": date(2026, 5, 1),
                    "rain_chance": 0.82,
                    "model_mean_in": 0.31,
                    "model_spread_in": 0.05,
                    "models_wet": 3,
                }
            ]
        ),
        per_model={
            "open-meteo:icon_seamless": pd.Series({date(2026, 5, 1): 0.30}),
            "open-meteo:gfs_seamless": pd.Series({date(2026, 5, 1): 0.32}),
        },
        alerts=[Alert(date(2026, 5, 1), "frost", "critical", "Freeze — 33 °F", "detail")],
        timezone_name="America/New_York",
        forecast_through=pd.Timestamp("2026-05-06", tz="UTC"),
    )
    base.update(overrides)
    return dashboard.build_payload(**base)


def test_payload_is_json_serialisable():
    import json

    json.dumps(payload())  # must not raise on numpy or date types


def test_current_conditions_come_from_the_latest_reading():
    out = payload()
    assert out["current"]["temp_f"] == pytest.approx(55.0)
    assert out["current"]["age_hours"] >= 0


def test_rain_today_uses_the_daily_accumulator():
    assert payload()["rain_today_in"] == pytest.approx(0.12)


def test_alerts_are_carried_through_with_severity():
    out = payload()
    assert len(out["alerts"]) == 1
    assert out["alerts"][0]["severity"] == "critical"


def test_trust_panel_states_limits_as_well_as_skill():
    trust = payload()["trust"]
    assert trust["rain"] and trust["frost"] and trust["temperature"]
    assert len(trust["cannot"]) >= 3, "the page must say what it cannot do"
    assert any("Wind" in item for item in trust["cannot"])


def test_nan_values_become_null_not_nan():
    # NaN is not valid JSON; a page that silently produced it would fail to
    # parse in the browser with no server-side error.
    import json

    nights = pd.DataFrame(
        [{"day": date(2026, 5, 1), "raw_min": np.nan, "corrected_min": 33.5}]
    )
    text = json.dumps(payload(nights=nights))
    assert "NaN" not in text
    assert '"raw_min":null' in text.replace(" ", "")


def test_render_writes_a_self_contained_page(tmp_path: Path):
    destination = dashboard.render(payload(), tmp_path / "sub" / "page.html")
    text = destination.read_text()

    assert destination.exists()
    assert "__DATA__" not in text, "placeholder must be replaced"
    assert "Freeze — 33 °F" in text
    # Self-contained: nothing fetched at view time.
    assert "http://" not in text.replace("http://www.w3.org", "")
    assert "<script src=" not in text


def test_render_rejects_a_template_without_the_placeholder(tmp_path, monkeypatch):
    broken = tmp_path / "broken.html"
    broken.write_text("<html>no placeholder</html>")
    monkeypatch.setattr(dashboard, "TEMPLATE", broken)

    with pytest.raises(ValueError, match="__DATA__"):
        dashboard.render(payload(), tmp_path / "out.html")


def test_empty_station_does_not_break_the_page():
    out = payload(station=pd.DataFrame(columns=["ts", "temp_f", "rain_daily_in"]))
    assert out["current"] == {}
    assert out["rain_today_in"] is None
