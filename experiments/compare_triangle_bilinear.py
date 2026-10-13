"""Compare geometry only: both methods interpolate total implied variance.

Run: .venv/bin/python experiments/compare_triangle_bilinear.py
Use original finite Combined IV, existing bilinear implementation, and the
dashboard's scaled Delaunay geometry. Production code is not changed.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import QhullError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compare_2d_interpolation import predict_pair
from iv_surface.fetcher import load_chain_snapshot, parse_chain_snapshot
from iv_surface.solver import interpolate_iv
from iv_surface.surface_builder import build_combined_surface_from_chain
from iv_surface.surface_transforms import surface_to_long_frame


def bilinear_variance(train, test, expiries, strikes):
    """Strict four observed corners on the retained Cartesian grid.

    Axis nodes can be removed by a holdout, but missing corner values are never
    filled from other observations. This is the existing interpolate_iv formula
    applied to total variance rather than IV.
    """
    values = train.assign(variance=train.iv**2 * train.tau).pivot(
        index="tau", columns="moneyness", values="variance"
    ).reindex(index=expiries, columns=strikes).to_numpy()
    predictions = []
    for row in test.itertuples():
        try:
            variance = interpolate_iv(values, expiries, strikes, row.tau, row.moneyness)
        except ValueError:
            variance = np.nan
        predictions.append(np.sqrt(variance / row.tau))
    return np.array(predictions)


def errors(frame, field):
    delta = 100 * (frame[field] - frame.actual)
    return {
        "mae_pp": float(delta.abs().mean()) if len(frame) else None,
        "rmse_pp": float(np.sqrt(np.mean(delta**2))) if len(frame) else None,
        "median_absolute_error_pp": float(delta.abs().median()) if len(frame) else None,
        "p95_absolute_error_pp": float(delta.abs().quantile(.95)) if len(frame) else None,
    }


def summarize(rows):
    triangle = rows.triangle.notna()
    bilinear = rows.bilinear.notna()
    shared = rows[triangle & bilinear]
    assert not (bilinear & ~triangle).any(), "A supported rectangle must be inside the hull"
    split = shared.assign(
        triangle_error=(shared.triangle - shared.actual).abs(),
        bilinear_error=(shared.bilinear - shared.actual).abs(),
    ).groupby("split")[["triangle_error", "bilinear_error"]].mean()
    return {
        "attempted": len(rows), "triangle_covered": int(triangle.sum()),
        "bilinear_covered": int(bilinear.sum()), "shared_covered": len(shared),
        "shared_errors": {"triangle": errors(shared, "triangle"),
                          "bilinear": errors(shared, "bilinear")},
        "own_coverage_errors_not_comparable": {
            "triangle": errors(rows[triangle], "triangle"),
            "bilinear": errors(rows[bilinear], "bilinear"),
        },
        "scored_splits": len(split),
        "bilinear_mae_wins": int((split.bilinear_error < split.triangle_error - 1e-12).sum()),
        "triangle_mae_wins": int((split.triangle_error < split.bilinear_error - 1e-12).sum()),
        "mae_ties": int(np.isclose(split.bilinear_error, split.triangle_error, atol=1e-12, rtol=0).sum()),
    }


def run_split(observed, taus, xs, label):
    train_mask = observed.tau.isin(taus) & observed.moneyness.isin(xs)
    train, test = observed[train_mask], observed[~train_mask]
    assert not train.index.isin(test.index).any()
    try:
        triangle = predict_pair(train, test, True)[:, 1]
    except QhullError:
        triangle = np.full(len(test), np.nan)
    bilinear = bilinear_variance(train, test, taus, xs)
    rows = test[["tau", "moneyness", "source"]].copy()
    target_kind = np.where(
        ~test.tau.isin(taus) & ~test.moneyness.isin(xs), "both_axes_removed",
        np.where(~test.tau.isin(taus), "expiry_only_removed", "strike_only_removed"),
    )
    return rows.assign(actual=test.iv, triangle=triangle, bilinear=bilinear,
                       split=label, target_kind=target_kind)


def sanity_checks():
    train = pd.DataFrame({"tau": [.1, .1, .5, .5], "moneyness": [.8, 1.2, .8, 1.2]})
    test = pd.DataFrame({"tau": [.3, .3], "moneyness": [1., 2.]})
    # Both recover affine total variance; both leave outside targets unsupported.
    train["iv"] = np.sqrt((.02 + .01 * train.moneyness + .03 * train.tau) / train.tau)
    triangle = predict_pair(train, test, True)[:, 1]
    bilinear = bilinear_variance(train, test, [.1, .5], [.8, 1.2])
    assert np.isclose(triangle[0], np.sqrt(.039 / .3))
    np.testing.assert_allclose(triangle, bilinear, equal_nan=True)
    # Bilinear exactly reproduces a cross term; a triangle's planar patch cannot.
    train["iv"] = np.sqrt((.02 + .1 * train.moneyness * train.tau) / train.tau)
    bilinear = bilinear_variance(train, test.iloc[:1], [.1, .5], [.8, 1.2])
    assert np.isclose(bilinear[0], np.sqrt(.05 / .3))
    assert not np.isclose(predict_pair(train, test.iloc[:1], True)[0, 1], bilinear[0])


def main():
    sanity_checks()
    snapshot_path = ROOT / "data/bybit_btc_options_snapshot.json"
    snapshot = load_chain_snapshot(snapshot_path)
    chain = parse_chain_snapshot(snapshot)
    result = build_combined_surface_from_chain(chain, r=0)
    frame = surface_to_long_frame(result.iv_surface, result.expiries, result.strikes,
                                  result.spot_price, source_grid=result.source_grid)
    evidence = {
        "captured_at_utc": snapshot["captured_at_utc"],
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "rate": 0, "methods": {
            "triangle": "Scaled Delaunay linear total variance; training-only axis min/max scaling.",
            "bilinear": "Existing solver.interpolate_iv on total variance grid; strictly observed corners.",
        },
        "protocol": "50 seeded axis holdouts remove ceil(20%) of interior expiry and strike nodes independently; endpoints retained. Also leave-one-expiry-out, including unsupported boundary expiries.",
        "caveats": [
            "Accuracy comparison uses only points supported by both methods.",
            "Axis holdouts are different from the previous random-point holdouts; do not compare their errors directly.",
            "Missing four-corner rectangles remain missing; no preliminary 1D fill or adaptive enlarged rectangles.",
            "Bilinear requires four finite corners even for a target lying on a cell edge, matching existing function behavior.",
            "One snapshot of solved Combined IV, not latent volatility or independent market regimes.",
            "Repeated splits reuse targets and are not independent statistical evidence.",
            "Neither method enforces no-arbitrage or guarantees a smooth gradient.",
        ], "ranges": {},
    }
    all_rows = []
    for name, lower, upper in [("core", .8, 1.2), ("extended", .7, 1.3)]:
        grid = frame[frame.moneyness.between(lower, upper)]
        observed = grid[grid.iv.notna()]
        taus, xs = np.sort(grid.tau.unique()), np.sort(grid.moneyness.unique())
        random_rows = []
        for seed in range(20261013, 20261063):
            rng = np.random.default_rng(seed)
            held_t = rng.choice(taus[1:-1], max(1, int(np.ceil(.2 * len(taus[1:-1])))), replace=False)
            held_x = rng.choice(xs[1:-1], max(1, int(np.ceil(.2 * len(xs[1:-1])))), replace=False)
            random_rows.append(run_split(observed, taus[~np.isin(taus, held_t)],
                                         xs[~np.isin(xs, held_x)], str(seed)))
        expiry_rows = [run_split(observed, taus[taus != tau], xs, str(tau)) for tau in taus]
        section = {"observed": len(observed), "grid_cells": len(grid), "experiments": {}}
        for mode, parts in [("axis_holdout", random_rows), ("leave_one_expiry_out", expiry_rows)]:
            rows = pd.concat(parts, ignore_index=True)
            section["experiments"][mode] = {
                "pooled": summarize(rows),
                "splits": [{"split": label, **summarize(group)} for label, group in rows.groupby("split")],
                "by_source": {str(label): summarize(group) for label, group in rows.groupby("source")},
                "by_target_kind": {str(label): summarize(group) for label, group in rows.groupby("target_kind")},
            }
            all_rows.append(rows.assign(range=name, experiment=mode))
            print(name, mode, json.dumps(summarize(rows), indent=2))
        missing = grid[grid.iv.isna()]
        triangle = predict_pair(observed, missing, True)[:, 1]
        bilinear = bilinear_variance(observed, missing, taus, xs)
        section["actual_missing_coverage"] = {
            "missing": len(missing), "triangle_filled": int(np.isfinite(triangle).sum()),
            "strict_existing_grid_bilinear_filled": int(np.isfinite(bilinear).sum()),
        }
        evidence["ranges"][name] = section
    output = Path(__file__).resolve().parent
    (output / "triangle_bilinear_comparison.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    pd.concat(all_rows, ignore_index=True).to_csv(output / "triangle_bilinear_predictions.csv", index=False)


if __name__ == "__main__":
    main()
