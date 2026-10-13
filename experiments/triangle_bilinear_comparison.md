# Triangle versus bilinear: geometry-only experiment

## Finding

Neither geometry is a clear accuracy winner on shared supported targets in the
frozen Bybit snapshot. Strict existing-grid bilinear covers substantially fewer
targets because it requires finite observed values at all four local corners.
Keep triangle interpolation for the current sparse-data display; this result
does not establish that bilinear is generally less accurate or unsuitable for
complete grids.

## Reproduce

```sh
.venv/bin/python experiments/compare_triangle_bilinear.py
```

The script saves `triangle_bilinear_comparison.json` and individual predictions
in `triangle_bilinear_predictions.csv`. It reuses `predict_pair` from the prior
experiment and the existing production `solver.interpolate_iv()` formula.
Dashboard and production code are unchanged by this experiment.

## Controls and holdouts

- Same snapshot captured 2026-09-02T12:58:21.965864Z, same existing Combined IV
  builder at r=0, same spot-based K/S and frozen numerical tau.
- Both methods interpolate total variance w=IV²×tau and recover sqrt(w/tau).
- Triangle uses Delaunay with each coordinate scaled to the observed training
  min/max range, matching the current dashboard.
- Bilinear uses the retained Cartesian expiry/moneyness grid. Corner NaNs stay
  NaN. There is no preliminary fill, adaptive rectangle expansion, or smoothing.
- In 50 seeded axis holdouts (seeds 20261013–20261062), independently remove
  ceil(20%) of interior expiry nodes and ceil(20%) of interior strike nodes.
  Retain boundary axes. Training consists only of original observed points on
  the remaining cross-product; held-out original observations are test targets.
- Also leave out each whole expiry in turn, including boundary expiries that
  cannot be reconstructed without extrapolation.
- Compare MAE/RMSE only on identical test targets supported by both methods.
  Report each method's coverage separately. Counts pooled across seeds are
  repeated test instances, not unique observations or independent samples.
- This axis-holdout protocol differs from the previous random-point experiment:
  its aggregate scores must not be directly compared with those older scores.

## Genuine two-axis targets

These are points whose expiry AND strike axes were both removed. Edge targets
along one retained axis are excluded so they do not dilute geometry differences.
Errors are IV percentage points (0.01 IV = 1 percentage point).

| Range | Attempted | Triangle supported | Bilinear supported / shared | Triangle shared MAE | Bilinear shared MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core 0.8–1.2 | 408 | 376 | 117 | 0.737857 | 0.722484 |
| Extended 0.7–1.3 | 555 | 497 | 116 | 0.667604 | 0.669689 |

Core shared RMSE: triangle 1.056000, bilinear 0.997190.
Extended shared RMSE: triangle 0.868074, bilinear 0.863329.
Core per-split shared MAE favors bilinear 23 times and triangle 20 times;
seven splits have no shared two-axis target. Extended favors bilinear 20 times
and triangle 21 times; nine splits have no shared two-axis target.

## All axis-holdout targets

| Range | Attempted | Triangle supported | Bilinear supported / shared | Triangle shared MAE | Bilinear shared MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core | 3,841 | 3,621 | 1,113 | 0.516409 | 0.514793 |
| Extended | 4,727 | 4,359 | 1,188 | 0.501602 | 0.501806 |

Targets along a retained axis have matching shared predictions in this dataset.
For whole-expiry holdout, shared MAE is equal to numerical precision: Core
0.541674 on 46 shared targets (triangle covers 164/211, bilinear 46/211);
Extended 0.550893 on 45 shared targets (triangle 195/253, bilinear 45/253).

## Existing missing grid nodes

With all original observations as training support, triangle fills 163/189
originally missing Core cells and 231/287 Extended cells. Applying the strict
old bilinear function directly to those missing knots fills zero: a target's
own missing knot is one of the required corners. This is a limitation of using
the existing fixed-grid function as a hole-filler, not a theorem that bilinear
cannot be used in a missing-data pipeline. A pre-filled or regridded surface
would be a different method and needs its own test.

The existing function also propagates NaN from zero-weight corners at cell
edges. Consequently its reported edge-target coverage is stricter than an
implementation that requires only nonzero-weight corners. The genuine two-axis
subset avoids this particular implementation limitation.

## Verification and limits

- Synthetic affine total variance is recovered by both methods; outside-grid
  targets stay missing. Bilinear exactly recovers a synthetic x×tau cross term,
  while the triangular planar patch does not, verifying that the implementations
  actually differ on two-axis targets.
- Training and test identities are disjoint. All bilinear-supported points
  are also within triangle support in the executed experiments.
- All four aggregate shared MAEs independently recalculated from saved CSV
  match the JSON summaries.
- One snapshot of solved Combined IV is not latent-volatility ground truth.
  Holdouts of observed quotes do not establish accuracy for systematically
  missing quotes. Supported subsets are biased toward better-covered regions.
- No confidence interval or universal accuracy claim is justified by these
  overlapping splits. No visual smoothness or no-arbitrage claim is tested.
