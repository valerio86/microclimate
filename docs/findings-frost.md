# Findings: frost

## The frost call, verified

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
