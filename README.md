# microclimate

Localized weather forecasting for a location that public forecasts get wrong.

Off-the-shelf forecasts predict for a **grid cell**, which knows nothing about
the terrain, elevation, and microclimates of a specific property. This project
uses a personal weather station as ground truth to measure and correct that
error — the same idea as Model Output Statistics (MOS), fit to one location.

## Sources

| Source | Role | Backfill window |
|---|---|---|
| Ambient Weather WS-2902 (via AWN cloud API) | ground truth observations | 2024-01-01 → now |
| Open-Meteo forecast archive | the public forecast baseline | 2024-01-01 → now |
| PurpleAir | air quality | from 2026-07-24 (install date) |

The station has recorded since 2022 and Ambient still serves that history, but
**2024-01-01 is a data-quality boundary, not a convenience one**: the station
was not correctly configured for its first couple of years, so the earlier
readings are unreliable. Bad ground truth is worse than none here — it would
bias the corrections while still looking plausible in aggregate. Widen the
window only if that early data is revisited and validated.

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
.venv/bin/microclimate backfill-station --start 2024-01-01
```

Walks backward through history at ~1 request/second (Ambient's rate limit), so
roughly 17 minutes for the ~940 days back to 2024. It is resumable — interrupt it freely and
re-run to continue from the oldest record stored.

```bash
.venv/bin/microclimate backfill-forecast --start 2024-01-01
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

## Early findings

From the first 8 months of paired data (Nov 2025 – Jul 2026, 5,676 hours),
before the full 2024+ backfill:

- **Wind is the big one.** Open-Meteo runs ~+6.8 mph on sustained wind and
  ~+12.5 mph on gusts, consistently at every lead time. That is the signature
  of a sheltered site being compared against open-terrain 10 m wind.
- **Temperature bias is diurnal, and the daily average hides it.** Near-zero
  mean error at lead 1, but a steady +1.1 to +1.5 °F overnight — the model runs
  warm at night, consistent with cold-air drainage in hilly terrain.
- **Error grows and flips sign with lead time**, from +0.65 °F at lead 0 to
  −2.65 °F at lead 7.

## Pick the model before correcting it

Open-Meteo's `best_match` resolves to **GFS** at this location, and GFS is the
worst of the four models here. Scored against the station, lead 1:

| Model | MAE | Lead 1 → 2 bias |
|---|---|---|
| icon_seamless | **2.33 °F** | +0.73 → +0.48 |
| gem_seamless | 2.42 °F | +0.80 → +0.76 |
| ecmwf_ifs025 | 2.48 °F | −0.61 → −0.82 |
| best_match / GFS | 2.82 °F | +0.02 → **−1.78** |

That last column also explains the lead-1 to lead-2 discontinuity that looked
like a data artifact: it is GFS drifting, and no other model does it.

The two effects compose. At lead 1, raw GFS is 2.80 °F MAE and 2.50 corrected;
raw ICON is 2.34 and **2.10 corrected** — a 25% total reduction, two-thirds of
which came from naming a model rather than from any modelling.

Models are stored side by side (`forecasts.source` is part of the key), so their
disagreement is available as an uncertainty feature. Analysis commands take
`--source` and default to ICON.

## Does correcting actually help?

`microclimate backtest` fits corrections on an earlier period and scores them on
a later one they never saw. Skill is versus the raw forecast: positive helped,
negative made it worse. Fitted on 2024-01 → 2025-10, scored on 2025-10 → 2026-07.

| Variable | Raw MAE | Corrected MAE | Skill | Persistence |
|---|---|---|---|---|
| Temperature (lead 1) | 2.80 °F | 2.59 °F | **+0.07** | −1.60 |
| Wind (lead 1) | 7.73 mph | 2.65 mph | **+0.66** | **+0.72** |

Three things this settled that reasoning could not:

- **A single mean correction makes day-ahead temperature *worse*** (skill −0.03).
  The seasonal and diurnal structure is the whole signal; the average of it is
  actively misleading. The headline numbers above are summary statistics, not a
  model.
- **The harmonic fit matches the bucket table with 13 coefficients instead of
  288 cells**, and edges ahead across all leads. Fewer parameters, same skill,
  far less to overfit.
- **For wind, plain persistence beats every correction we have.** Yesterday's
  wind at this hour predicts today's better than a bias-corrected grid forecast
  does. The +7 mph offset is real and worth removing, but past that the forecast
  adds little for wind at this site. Temperature is the opposite — persistence
  is far worse than the forecast, so the model genuinely knows something.

### Regime conditioning

Calendar position is only a proxy for the physics. Conditioning on the forecast's
own cloud cover and wind speed — both known ahead of time — beats calendar alone
(temperature lead 1: MAE 2.50 vs 2.59, skill 0.10 vs 0.07), and the mechanism
holds up when inspected directly:

| Overnight conditions | Temperature bias | n |
|---|---|---|
| Clear + calm | **+4.58 °F** | 666 |
| Cloudy + calm | +2.66 °F | 754 |
| Clear + windy | −0.72 °F | 98 |
| Cloudy + windy | −0.19 °F | 1,172 |

This is cold-air pooling: on calm nights the surface decouples from the air above
and cold air settles; wind mixes it away. The "+2.0 °F overnight" headline is an
average of +4.6 and −0.2, and describes neither.

### The frost call, verified

`microclimate frost-skill` scores the yes/no decision on held-out nights, which
is a different question from MAE — a missed frost costs a crop, a false alarm
costs an evening.

Across all 255 held-out nights the raw forecast is already good (92% of frosts
caught, 1% false alarms). The interesting case is the 44 **marginal** nights
where the forecast minimum lands between 32 and 40 °F, 14 of which froze:

| Warning level | Raw | Corrected |
|---|---|---|
| 32 °F | 4 hits, **10 misses** | 8 hits, 6 misses |
| 34 °F | 13 hits, 1 miss, 2 false alarms | **14 hits, 0 misses**, 5 false alarms |
| 36 °F | 14 hits, 12 false alarms | 14 hits, 13 false alarms |

**Taking the 32 °F line literally misses most marginal frosts.** Warning at 34 °F
fixes nearly all of it on its own; the correction closes the remainder and
improves the underlying estimate (overnight-minimum bias +1.37 → +0.59 °F, MAE
2.32 → 1.86 on marginal nights).

An earlier note here framed this as "a coin flip at 36 °F". That was an
hour-level statistic — comparing a single hour's forecast against whether frost
occurred at any point overnight — and it overstated the miss rate. The nightly
minimum is far more informative than any single hour, and the night-level
numbers above supersede it.

### Instrument corrections

Ground truth has to be trusted before anything is fitted to it. Three defects
found, in decreasing order of how well they are pinned down:

**Temperature — no detectable radiation-shield error.** Cloud-shadow analysis
initially suggested 2.9 °F per 1000 W/m² at a 5-minute lag, which looked like an
instrument response. A tree ~15 m from the station overturned it. The tree clips
the sensor for one 5-minute sample at a fixed bearing on every clear day (178
transits on record), and it is the control the cloud analysis lacked: **a cloud
shades the whole field and cools the air; a tree shades only the sensor.**
Compositing all 178 transits, solar drops 592 → 378 W/m² and temperature does not
fall (+0.28 ± 0.06 °F, wrong sign) where a 2.9 °F shield would have produced a
~0.4 °F drop. So the cloud figure was mostly real air cooling. Shield error is
bounded below **~0.75 °F per 1000 W/m²** and is consistent with zero.

The corollary is that essentially all of the solar-correlated warmth in the
forecast residual is **real**: the mown open field genuinely runs hotter than a
grid cell averaging in forest and water. See `shield.py`; the correction is
opt-in, sized at the upper bound, and unused by default.

Also worth noting: the daily *minimum* occurs at or before sunrise, with the
house blocking the east, so **frost work is unaffected by any of this** whatever
the shield turns out to do.

**Rain — fixed in code.** `hourlyrainin` is a trailing 60-minute total, so taking
its hourly maximum counted the same rainfall in two adjacent clock hours and
inflated station totals to roughly twice the forecast. Rain is now differenced
from the daily accumulator. Station/forecast went from 1.99 to 0.61, which reads
as ordinary gauge under-catch: 0.71 in spring and summer, **0.29 in winter**,
where an unheated tipping bucket does not register snow until it melts. Treat
winter precipitation volume as unusable.

**Solar — the pyranometer is obstructed, mostly to the east.** Comparing clear-sky
ratios at *matched solar elevation* (equal air mass, so any east/west gap is
obstruction rather than geometry) over 21 clear days:

| Solar elevation | Morning (E) | Afternoon (W) | Gap |
|---|---|---|---|
| 15–20° | 0.23 | 0.66 | +0.43 |
| 20–25° | 0.32 | 0.74 | +0.42 |
| 45–50° | 0.93 | 0.84 | −0.10 |
| 55–60° | 0.82 | 0.82 | 0.00 |

Mornings below 25° elevation get a third to a half of the afternoon equivalent —
a hill or treeline on the eastern horizon. A smaller, broad afternoon deficit
also shows at mid elevations. So **observed solar under-reports true insolation**,
and part of the apparent "model over-forecasts solar radiation" is our own shaded
sensor. Corrections use *forecast* solar, so they are unaffected; do not use
observed `solar_wm2` as a model feature without accounting for this.

**Wind — see below.** Unresolved, and the reason the shield's ventilation term
is still a literature value rather than a measured one.

### Open question: the anemometer reads about half

The station sits on an 8 ft mast in an open field. The log wind profile puts the
expected ratio to Open-Meteo's 10 m wind at **0.69–0.80**. Observed is **0.28**,
and **0.35 even above 15 mph** where a start-up threshold is irrelevant; gusts
read 0.34, and the highest gust in 2.5 years is 32 mph. Siting does not explain a
shortfall that size — worth a physical check of the cups and bearing before
trusting any wind correction.

## Caveats

- Hourly alignment assumes Open-Meteo's hourly values line up with the
  station's hourly aggregates. Instantaneous variables (temperature) are a
  close match; accumulations (precipitation) are attributed to the preceding
  hour and are worth validating before trusting precipitation error stats.
- `rain_hourly_in` from the station is a trailing-60-minute total, approximated
  here by the hourly maximum.
