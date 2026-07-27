# Findings: model selection and correction

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

## How much does correcting help, really?

`microclimate crossval` runs five expanding-window folds and seals off a final
holdout. It is a much harsher test than a single split, and the corrections do
not survive it well.

On ICON, across folds:

| Method | skill mean | skill std | worst fold | best fold |
|---|---|---|---|---|
| regime | +0.06 | 0.08 | +0.02 | +0.20 |
| adaptive | +0.06 | 0.08 | +0.01 | +0.20 |
| harmonic | +0.05 | 0.07 | −0.01 | +0.16 |
| buckets | +0.03 | 0.08 | −0.02 | +0.17 |
| constant | +0.01 | 0.01 | −0.00 | +0.02 |

**The spread exceeds the mean.** Nearly all the benefit comes from one fold —
summer 2025 — which matches the seasonal picture: a large correctable warm bias
in July and August, very little the rest of the year. It is mostly a *summer*
correction, and single-split numbers of 0.10-0.11 were the luck of which period
landed in the test set.

On GFS the same methods are actively harmful: regime averages **−0.06** with a
worst fold of **−0.58**, harmonic −0.11 and −0.76. Flexible corrections fitted to
a model whose bias is unstable transfer badly, and only the crude ones (buckets,
constant) stay near zero. The lesson is not that correction never works but that
**it is worth far less than choosing the right model, and it is not free** — on
the wrong baseline it destroys skill.

The adaptive lag features (`recent_error`, persistence gap, trend) added nothing
over the regime model. A negative result, but a real one: ICON's errors here are
not autocorrelated day-to-day in a way a linear term can exploit.


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

## Regime conditioning

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
