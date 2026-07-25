from datetime import date, datetime, timezone

import pandas as pd
import pytest

from microclimate.sources import ambient, openmeteo


def _client_with_data_on(days: set[date]) -> ambient.AmbientClient:
    """A client whose history has records only on `days` — everything else is
    an outage. Built without __init__ so no network or credentials are used."""
    client = ambient.AmbientClient.__new__(ambient.AmbientClient)

    def fake_fetch_page(end, limit=ambient.MAX_RECORDS_PER_CALL):
        if end.date() not in days:
            return []
        midnight = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
        return [{"dateutc": int(midnight.timestamp() * 1000), "tempf": 40.0}]

    client.fetch_page = fake_fetch_page
    return client


def test_iter_history_steps_over_an_outage():
    # Regression: a multi-day station outage used to end the backfill early,
    # silently truncating years of history that were still available.
    client = _client_with_data_on({date(2024, 1, 10), date(2024, 1, 5)})

    pages = list(
        client.iter_history(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, 12, tzinfo=timezone.utc),
        )
    )

    assert len(pages) == 2, "should have crossed the 4-day gap to the older data"


def test_iter_history_gives_up_after_a_long_silence():
    client = _client_with_data_on({date(2024, 3, 1)})

    pages = list(
        client.iter_history(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, 12, tzinfo=timezone.utc),
            max_gap_days=10,
        )
    )

    assert len(pages) == 1  # stops rather than crawling a year of nothing


def test_ambient_to_frame_maps_and_converts():
    records = [
        {
            "dateutc": 1704067200000,  # 2024-01-01T00:00:00Z
            "tempf": 51.8,
            "humidity": 72,
            "windspeedmph": 4.5,
            "windgustmph": 9.0,
            "winddir": 270,
            "baromrelin": 30.0,
            "solarradiation": 120.5,
            "uv": 2,
            "dewPoint": 43.2,
        }
    ]

    frame = ambient.to_frame(records)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ts"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert row["temp_f"] == pytest.approx(51.8)
    assert row["rh"] == pytest.approx(72)
    assert row["pressure_hpa"] == pytest.approx(30.0 * 33.8639)


def test_ambient_to_frame_handles_missing_fields():
    frame = ambient.to_frame([{"dateutc": 1704067200000, "tempf": 50.0}])
    assert frame["temp_f"].iloc[0] == pytest.approx(50.0)
    assert pd.isna(frame["gust_mph"].iloc[0])


def test_ambient_to_frame_deduplicates_and_sorts():
    records = [
        {"dateutc": 1704067200000, "tempf": 50.0},
        {"dateutc": 1704063600000, "tempf": 49.0},
        {"dateutc": 1704067200000, "tempf": 50.0},
    ]
    frame = ambient.to_frame(records)
    assert len(frame) == 2
    assert frame["ts"].is_monotonic_increasing


def test_ambient_to_frame_empty():
    assert ambient.to_frame([]).empty


def test_openmeteo_chunks_cover_range_without_gaps():
    chunks = list(openmeteo._chunks(date(2024, 1, 1), date(2024, 5, 1), 30))

    assert chunks[0][0] == date(2024, 1, 1)
    assert chunks[-1][1] == date(2024, 5, 1)
    for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert (next_start - previous_end).days == 1


def test_openmeteo_rejects_out_of_range_lead():
    client = openmeteo.OpenMeteoClient.__new__(openmeteo.OpenMeteoClient)
    with pytest.raises(ValueError):
        client.fetch_lead(date(2024, 1, 1), date(2024, 1, 2), lead_days=9)
