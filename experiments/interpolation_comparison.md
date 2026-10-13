# 2D IV versus total-variance interpolation

## Finding

On the frozen Bybit snapshot captured at 2026-09-02T12:58:21.965864Z,
linear total-variance interpolation reconstructs held-out Combined IV more
accurately overall than direct linear IV interpolation. This is evidence for
this snapshot, not a universal accuracy or no-arbitrage guarantee.

## Reproduce

From the repository root:

```sh
.venv/bin/python experiments/compare_2d_interpolation.py
```

This produces `interpolation_comparison.json` (environment, snapshot hash,
aggregate metrics, per-source metrics and per-split metrics) and
`interpolation_holdout_predictions.csv` (individual held-out predictions).
No production code or dashboard settings are changed. This uses a Python
script because the project environment has no notebook execution dependencies.

## Method

- Existing frozen raw snapshot -> existing parser -> existing Combined IV
  builder at continuously compounded r=0; shared spot and frozen tau unchanged.
- Core K/S 0.8–1.2: 211 finite points across 10 expiries, 400 grid cells.
- Extended K/S 0.7–1.3: 253 finite points across 10 expiries, 540 grid cells.
- Only original finite Combined IV points are training observations. No 1D
  interpolation or held-out values enter training.
- Direct method: piecewise-linear IV on (K/S, tau).
- Variance method: piecewise-linear IV²×tau on the same coordinates, followed
  by sqrt(w/tau) at the test point.
- A shared training-only Delaunay triangulation fixes geometry and coverage for
  each paired comparison. Outside its convex hull, both methods return NaN.
- Primary geometry independently scales both axes to training min/max ranges.
  A sensitivity run uses unscaled coordinates; neither is tuned on test error.
- Random holdout: ceil(20% of observations) per expiry, 50 seeds beginning at
  20261013. Repeated draws are not independent market samples.
- Leave-one-expiry-out: remove each entire expiry in turn. Boundary expiries
  cannot be scored without extrapolation; all attempts remain in coverage.
- MAE and RMSE compare predicted versus original solved IV in percentage
  points: 0.01 IV = 1 percentage point. Pooled scores weight each scored
  prediction equally; JSON also retains per-expiry/split scores.

## Results: training-range-scaled coordinates

| Range / holdout | Scored / attempted | Direct IV MAE | Total variance MAE | Direct IV RMSE | Total variance RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core / random | 2,233 / 2,350 | 2.376 | 0.834 | 6.368 | 1.586 |
| Core / whole expiry | 164 / 211 | 2.245 | 0.853 | 5.532 | 1.195 |
| Extended / random | 2,613 / 2,750 | 2.416 | 0.775 | 8.950 | 1.660 |
| Extended / whole expiry | 195 / 253 | 4.114 | 0.944 | 12.550 | 1.461 |

Total variance has lower random-holdout MAE in all 50 seeds for both ranges.
For whole-expiry holdout it wins in 5/8 scorable Core expiries and 6/8 Extended
expiries, not every expiry. In Core, the near-ATM-average source subset of
whole-expiry holdout favors direct IV (MAE 0.816 versus 0.951).

Native-coordinate sensitivity gives the same overall direction:

| Range / holdout | Direct IV MAE | Total variance MAE |
| --- | ---: | ---: |
| Core / random | 1.599 | 0.695 |
| Core / whole expiry | 2.169 | 0.856 |
| Extended / random | 1.908 | 0.684 |
| Extended / whole expiry | 3.741 | 0.930 |

Geometry materially affects errors. These results do not settle an optimal
coordinate scaling policy. The strongest Core gains occur in short-dated put
wings; they are not uniform improvements throughout the surface.

## Actual missing-cell coverage (all observed points used)

| Range | Originally missing | Filled by existing 1D | Filled by 2D | Remaining after 2D |
| --- | ---: | ---: | ---: | ---: |
| Core | 189 | 150 | 163 | 26 |
| Extended | 287 | 192 | 231 | 56 |

Both 2D methods fill identical cells; changing the interpolated quantity changes
values, not coverage. This counts existing grid cells, not a denser display
mesh. Dense rendering alone is not extra market evidence.

## Checks and limits

- Synthetic affine-IV and affine-total-variance fields are recovered by their
  respective methods; outside-hull test points remain NaN.
- Training/test identities are disjoint; both methods have identical finite
  prediction masks. Input coordinate identities are unique and tau is positive.
- Core MAEs independently recomputed from the saved CSV match JSON summaries.
- These targets are model-solved market-mid Combined IV, not true latent IV.
  Quote noise, short-dated solver sensitivity, and side fallbacks remain.
- Observed-point holdouts do not mimic all systematically missing quotes.
- One snapshot cannot establish performance across market regimes; repeated
  seeds must not be interpreted as independent statistical confidence.
- Neither method enforces calendar/butterfly no-arbitrage or removes roughness
  already present in observed data.

Recommendation: total variance is the stronger candidate for the proposed 2D
display interpolation on current evidence. Agree on coordinate geometry and
display behavior before production integration; leave Smile and ATM based on
their existing observed-data contracts.
