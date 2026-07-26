"""Radiation-shield heating of the temperature sensor.

The WS-2902's thermometer sits in a passive, naturally-ventilated shield. In
sunlight the shield itself warms and the sensor reads high; ventilation carries
the heat away, so the error grows with solar radiation and shrinks with wind.

Both signatures are present in this station's data. Comparing against the
forecast, the apparent daytime warmth goes from +0.5 °F under cloud to −2.2 °F
in strong sun at low wind, easing to −1.5 °F when it is breezy.

**What we cannot do is take that at face value.** Fitting the whole
solar-correlated residual gives k ≈ 8 °F at 1000 W/m² with almost no wind
dependence (u0 ≈ 20 mph). Published errors for passive shields are ≤ 2-4 °F at
full sun and depend strongly on ventilation, so a fit that large is mostly
picking up something else — most likely a real land-surface difference, since a
mown open field heats more than a grid cell that averages in forest and water.
Forecast comparison alone cannot separate the instrument from the site: both
scale with sunshine.

**A first estimate from cloud shadows was wrong.** Filtering out everything
slower than 90 minutes and regressing 5-minute wiggles in temperature on those
in solar radiation gives a peak at a 5-minute lag worth 2.9 °F per 1000 W/m².
That looked like an instrument response — too fast, it seemed, for a field's
thermal inertia.

**A tree shadow refutes it.** A tree ~15 m southwest of the station clips the
sensor for a single 5-minute sample around midday, at a fixed bearing, on every
clear day — 178 such transits are on record. This is the control the cloud
analysis lacked: **a cloud shades the whole field and cools the air itself,
while a tree shades only the sensor** and leaves the air blowing past it
unchanged. Compositing all 178 transits, solar drops 592 → 378 W/m² and
temperature does *not* fall (+0.28 ± 0.06 °F, the wrong sign). A 2.9 °F/1000
W/m² shield would have produced a ~0.4 °F drop.

So the cloud-derived figure was mostly **real air cooling**, not instrument
error. Bounding the shield from the tree transits instead puts it below roughly
**0.75 °F per 1000 W/m²**, and plausibly near zero. `DEFAULT_K_F` is that upper
bound, so the correction stays deliberately small; the honest reading is that
this station's shield error has not been shown to exist at a detectable level.

Caveat on power: a 5-minute perturbation only partly equilibrates the shield, so
this bounds the *fast* component. A shield with a time constant much longer than
5 minutes could hide — but that would also contradict the 5-minute lag the cloud
regression found.

The ventilation term is the part still unpinned. Splitting the same regression
by wind gives 1.5-1.6 °F per 1000 W/m² across every wind band — no dependence at
all, which contradicts the physics and almost certainly reflects the
under-reading anemometer rather than reality. `DEFAULT_U0_MPH` therefore remains
a literature value. Re-fit it once the wind sensor is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Error at 1000 W/m², degrees F. This is an *upper bound* from the tree-transit
# experiment (see module docstring), not a point estimate — the measured effect
# is consistent with zero. Kept non-zero so the correction is available if a
# reference sensor later shows a real error, but sized so applying it cannot do
# much harm.
DEFAULT_K_F = 0.75

# Ventilation scale in mph: the wind at which the error halves. Passive shields
# lose most of their error by ~10 mph.
DEFAULT_U0_MPH = 5.0

# The anemometer reads roughly this fraction of true wind (see README). Applied
# so the ventilation term reflects actual airflow rather than the sensor's
# under-report. Set to 1.0 once the anemometer is repaired or replaced.
DEFAULT_WIND_SCALE = 0.47


def radiation_error(
    solar_wm2: np.ndarray | pd.Series,
    wind_mph: np.ndarray | pd.Series,
    k_f: float = DEFAULT_K_F,
    u0_mph: float = DEFAULT_U0_MPH,
    wind_scale: float = DEFAULT_WIND_SCALE,
) -> np.ndarray:
    """Estimated °F by which the shield reads high.

        error = k * (solar / 1000) / (1 + wind / u0)

    Always >= 0: the shield can warm the sensor, never cool it below ambient.
    """
    solar = np.asarray(pd.to_numeric(solar_wm2, errors="coerce"), dtype=float)
    wind = np.asarray(pd.to_numeric(wind_mph, errors="coerce"), dtype=float)

    solar = np.nan_to_num(solar, nan=0.0).clip(min=0.0)
    # Missing wind is treated as calm, the worst case, so the estimate is not
    # accidentally optimistic where ventilation is unknown.
    wind = np.nan_to_num(wind, nan=0.0).clip(min=0.0)
    true_wind = wind / wind_scale if wind_scale else wind

    return k_f * (solar / 1000.0) / (1.0 + true_wind / u0_mph)


def correct_observations(
    station: pd.DataFrame,
    k_f: float = DEFAULT_K_F,
    u0_mph: float = DEFAULT_U0_MPH,
    wind_scale: float = DEFAULT_WIND_SCALE,
) -> pd.DataFrame:
    """Return a copy with `temp_f` reduced by the estimated shield error.

    Adds `shield_error_f` so the adjustment stays visible and reversible rather
    than silently folded into the observations.
    """
    if station.empty or "solar_wm2" not in station.columns:
        return station

    out = station.copy()
    error = radiation_error(
        out["solar_wm2"], out.get("wind_mph", 0.0), k_f, u0_mph, wind_scale
    )
    out["shield_error_f"] = error
    out["temp_f"] = out["temp_f"] - error
    return out
