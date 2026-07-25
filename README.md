# microclimate

Localized weather forecasting for a location that public forecasts get wrong.

Off-the-shelf forecasts predict for a **grid cell**, which knows nothing about
the terrain, elevation, and microclimates of a specific property. This project
uses a personal weather station as ground truth to measure and correct that
error — the same idea as Model Output Statistics (MOS), fit to one location.

## Sources

| Source | Role | History |
|---|---|---|
| Ambient Weather WS-2902 (via AWN cloud API) | ground truth observations | since 2022 |
| Open-Meteo forecast archive | the public forecast baseline | since 2022 |
| PurpleAir | air quality | since 2026-07-24 |

The WS-2902 has no local API — readings are read back from Ambient's cloud, at
288 records (one day) per request.

The key enabler is Open-Meteo's **previous-runs API**: for any past hour it
returns what the model predicted 1–7 days beforehand. That means the
`(forecast, actual)` training pairs can be reconstructed from data that already
exists, rather than accumulated going forward.

## Plan

- **Phase 1 — collect.** Backfill station history and matching forecasts into
  DuckDB. *(this scaffold)*
- **Phase 2 — measure.** Quantify the error: by variable, hour of day, season,
  and lead time. Often actionable on its own.
- **Phase 3 — correct.** Start with conditional-mean bias correction as the
  baseline, then ML once it has to be beaten.
- **Phase 4 — serve.** Web dashboard, then possibly a native iPhone app.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
cp .env.example .env
```

Fill in `.env`:

- `AMBIENT_API_KEY`, `AMBIENT_APPLICATION_KEY` — ambientweather.net → Account → API Keys
- `AMBIENT_MAC` — run `microclimate devices` to list yours
- `LATITUDE`, `LONGITUDE`, `ELEVATION_M`, `TIMEZONE` — elevation matters; it stops
  Open-Meteo from using the grid cell's mean height, a large bias in hilly terrain
- `PURPLEAIR_API_KEY`, `PURPLEAIR_SENSOR_INDEX` — optional for now

`.env` and `data/` are gitignored.

## Usage

```bash
.venv/bin/microclimate devices
```

```bash
.venv/bin/microclimate backfill-station --start 2022-01-01
```

Walks backward through history at ~1 request/second (Ambient's rate limit), so
roughly 25 minutes for four years. It is resumable — interrupt it freely and
re-run to continue from the oldest record stored.

```bash
.venv/bin/microclimate backfill-forecast --start 2022-01-01
```

```bash
.venv/bin/microclimate bias --variable temp_f --by local_hour --lead 1
```

```bash
.venv/bin/microclimate status
```

## Layout

```
src/microclimate/
  config.py          environment + location config
  store.py           DuckDB schema and upserts
  align.py           5-min observations → hourly, paired with forecasts
  sources/
    ambient.py       AWN client, backward-walking history pagination
    openmeteo.py     forecast archive + previous-runs (lead-resolved)
    purpleair.py     air quality
  analysis/
    bias.py          error statistics and conditional-mean correction
```

## Conventions

- All timestamps stored as UTC; local time is derived at query time.
- Error sign is **forecast − actual**: positive means the public forecast runs
  high here.
- Wind direction is averaged circularly — a plain mean of 350° and 10° gives
  180°, the opposite of the right answer.

## Caveats

- Hourly alignment assumes Open-Meteo's hourly values line up with the
  station's hourly aggregates. Instantaneous variables (temperature) are a
  close match; accumulations (precipitation) are attributed to the preceding
  hour and are worth validating before trusting precipitation error stats.
- `rain_hourly_in` from the station is a trailing-60-minute total, approximated
  here by the hourly maximum.
