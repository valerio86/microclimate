# Findings: rain chance

## Rain chance — the strongest result here

Daily probability of measurable rain, scored with the Brier score against
climatology. Walk-forward folds on ICON at lead 1:

| Fold | Test starts | Days | Base rate | Brier | Skill |
|---|---|---|---|---|---|
| 0 | 2024-08-29 | 68 | 29% | 0.181 | +0.13 |
| 1 | 2025-02-25 | 33 | 55% | 0.117 | +0.53 |
| 2 | 2025-05-08 | 29 | 48% | 0.194 | +0.22 |
| 3 | 2025-09-11 | 57 | 33% | 0.139 | +0.37 |

**Mean skill 0.31, standard deviation 0.15, every fold positive.** Compare the
temperature corrections at 0.06 ± 0.08 with one fold carrying the average: here
the mean is twice the spread and the worst fold is still clearly useful. On a
single split it scores 0.44; the cross-validated 0.31 is the number to believe.

Calibration is close to honest — predicted 0.09 against observed 0.09, 0.33
against 0.33, 0.84 against 0.83. The middle band (0.51 predicted, 0.73 observed,
n=22) is under-confident, though at that sample size it is about two standard
errors out and may be nothing.

All of this comes from a deliberately dumb baseline: bucket the forecast total,
look up how often it actually rained in that bucket. No model. That the simple
thing works this well is itself the finding — the value is in knowing what ICON's
numbers *mean at this location*, not in a clever function.

Costs: 445 usable days out of ~940. Winter goes to snow blindness, summer 2025 to
the gauge blockage, and partial days are dropped rather than half-counted.

## The learned model: three features beat fifteen

Scored once on the sealed holdout (109 days, 2026-03-08 to 2026-07-25, never
touched during selection):

| Method | Brier | Skill |
|---|---|---|
| climatology | 0.251 | −0.01 |
| amount lookup (baseline) | 0.136 | +0.45 |
| **logistic, 3 features** | **0.120** | **+0.52** |
| logistic, all 15 features | 0.134 | +0.46 |

The three features are **how many of the three models forecast rain, their mean
total, and their spread** — nothing else. Model agreement is very nearly the
whole signal, and `model_spread_in` enters negative: when the models disagree,
rain is less likely.

Two of my predictions died here. **Convective share** was supposed to be the key
discriminator for whether a shower lands on a point sensor; removing it changes
the score not at all. And **gradient boosting** was supposed to beat logistic
regression; on 336 training days it does not (+0.33 against +0.39 across folds).

Calibration on the holdout is good at the extremes (0.75 predicted / 0.74
observed, 0.90 / 0.95) and **under-confident in the middle**: 0.44 predicted
against 0.65 observed. When this model says 45%, treat it as nearer 60%.

Note the holdout scores higher than cross-validation (+0.52 against +0.41 mean).
The holdout is one favourable period; **+0.41 ± 0.15 is the number to plan with**.
