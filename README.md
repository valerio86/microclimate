# microclimate

Localized weather forecasting for a place that public forecasts get wrong.

Off-the-shelf forecasts predict for a **grid cell**, which knows nothing about the
terrain and elevation of one property. This uses a personal weather station as
ground truth to measure and correct that error, and surfaces only the results
that survived verification.

Station: Ambient WS-2902 at Valerio's farmhouse in Delaware County, on an 8 ft
mast in an open field. Forecasts: Open-Meteo (ICON, ECMWF, GFS).

## What it does

**Alerts** — silent unless something is actionable:

- **Frost** when the corrected overnight minimum is at or below 34 °F
- **Rain** at 70% chance or half an inch
- **Deviation** when this site will differ from the public forecast by over 3 °F

**A dashboard** whose only job is answering *why did it say that* — the overnight
curve behind a frost warning, per-model totals behind a rain probability, a
verification panel showing how right it has been, and a statement of what it
cannot do.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,analysis]"
```

```bash
cp .env.example .env    # then fill in keys and location
```

```bash
.venv/bin/microclimate backfill-station --start 2024-01-01
```

```bash
.venv/bin/microclimate backfill-forecast --start 2024-01-01
```

Then daily use:

```bash
.venv/bin/microclimate refresh      # update data, rebuild the page
```

```bash
.venv/bin/microclimate alerts       # anything worth acting on?
```

```bash
open data/dashboard.html
```

`scripts/com.microclimate.refresh.plist` runs `refresh` at 06:15 daily on macOS.
Install instructions are in the file.

## Headline results

All measured on data the models never saw during fitting.

| | |
|---|---|
| **Rain chance** | Brier skill **0.41 ± 0.15** across folds, **0.52** on sealed data |
| **Frost** | warning at 34 °F caught **14 of 14** on held-out marginal nights |
| **Model choice** | ICON **2.33 °F** MAE vs GFS 2.82 — bigger win than any correction |
| **Temperature correction** | skill **0.06 ± 0.08** — real but modest, mostly a summer effect |

**What it cannot do:** wind (the anemometer reads about half), snow (an unheated
gauge misses it), air quality (PurpleAir needs a winter first), hourly shower
timing (convective cells miss a point sensor).

## Detailed findings

Kept out of this file so it stays readable. Load one when it is relevant.

| Document | Covers |
|---|---|
| [docs/findings-models.md](docs/findings-models.md) | Why ICON over GFS, the lead-1/lead-2 discontinuity, cross-validated correction skill, regime conditioning |
| [docs/findings-rain.md](docs/findings-rain.md) | The rain-chance target, snow blindness, the frozen three-feature model, why three features beat fifteen |
| [docs/findings-frost.md](docs/findings-frost.md) | Frost verification as a decision, why 34 °F, hit and false-alarm rates |
| [docs/findings-instruments.md](docs/findings-instruments.md) | Radiation shield, the 90-day gauge blockage, pyranometer obstructions, the anemometer, alignment caveats |

## Commands

| Command | Purpose |
|---|---|
| `refresh` | Update data and rebuild the page. Warns if the station has gone quiet. |
| `alerts` | Conditions worth acting on. |
| `dashboard` | Build `data/dashboard.html`. |
| `rain-forecast` | Rain chance for the days ahead. |
| `rain-health` | Check the gauge for blockages. |
| `crossval` | Compare correction methods across walk-forward folds. |
| `backtest` | Score corrections on held-out data. |
| `frost-skill` | Score the frost call on held-out nights. |
| `bias` | Where the public forecast is wrong here. |
| `status` | What data is stored. |

## Layout

```
src/microclimate/
  config.py      environment and location
  store.py       DuckDB schema, upserts, migrations
  align.py       5-min observations to hourly, paired with forecasts
  features.py    leakage-safe lagged observations
  alerts.py      the three alert rules
  history.py     out-of-sample verification
  dashboard.py   page assembly
  shield.py      radiation-shield correction (opt-in)
  sources/       ambient, openmeteo, purpleair clients
  analysis/      bias, backtest, frost, rain, rain_model
```

## Conventions

- Timestamps stored as UTC; local time derived at query time.
- Error sign is **forecast − actual**: positive means the forecast reads high.
- Wind direction averages circularly — a plain mean of 350° and 10° gives 180°.
- Forecast sources are stored side by side; `--source` selects one.
- Data quality boundaries are encoded in code, not remembered: the 2024 backfill
  floor, `rain.GAUGE_OUTAGES`, and the snow filter.

## Principles worth keeping

Learned the hard way over the course of building this.

1. **Check the baseline before modelling.** Switching from GFS to ICON beat every
   correction fitted on top of it, and cost one parameter.
2. **Single splits flatter.** Every result that looked good on one split shrank
   under walk-forward folds — often by a factor of three.
3. **Simpler kept winning.** Three features beat fifteen; logistic beat gradient
   boosting; a lookup table beat the regime model.
4. **Ground truth is the bottleneck.** A blocked gauge, a shaded pyranometer and a
   half-reading anemometer each mattered more than any modelling choice.
5. **Physical facts beat inference.** Several confident statistical conclusions
   were corrected by a single sentence about the actual site.
