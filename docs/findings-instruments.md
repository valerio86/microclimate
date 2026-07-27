# Findings: instrument quality

## Instrument corrections

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

## Open question: the anemometer reads about half

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
