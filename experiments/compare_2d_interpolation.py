"""Offline holdout experiment; run with .venv/bin/python experiments/compare_2d_interpolation.py.

Compares linear IV and linear total variance using identical training points,
triangulations, and test points. Outputs JSON evidence beside this script.
Does not alter the dashboard, snapshot, or production transforms.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iv_surface.fetcher import load_chain_snapshot, parse_chain_snapshot
from iv_surface.surface_builder import build_combined_surface_from_chain
from iv_surface.surface_transforms import interpolate_smiles_1d, surface_to_long_frame


def predict_pair(train, test, rescale):
    """Coordinates are K/S and tau; any scaling is fitted on training only."""
    points = train[["moneyness", "tau"]].to_numpy()
    targets = test[["moneyness", "tau"]].to_numpy()
    shift = points.min(axis=0) if rescale else np.zeros(2)
    scale = np.ptp(points, axis=0) if rescale else np.ones(2)
    if np.any(scale == 0) or len(points) < 3:
        return np.full((len(test), 2), np.nan)
    mesh = Delaunay((points - shift) / scale)
    values = train.iv.to_numpy()
    variance = values**2 * train.tau.to_numpy()
    query = (targets - shift) / scale
    direct = LinearNDInterpolator(mesh, values)(query)
    total = LinearNDInterpolator(mesh, variance)(query)
    recovered = np.sqrt(total / test.tau.to_numpy())
    assert np.array_equal(np.isfinite(direct), np.isfinite(recovered))
    return np.column_stack([direct, recovered])


def summarize(rows):
    scored = rows[rows.iv_prediction.notna()]
    output = {"attempted": len(rows), "scored": len(scored)}
    output["coverage_pct"] = 100 * len(scored) / len(rows) if len(rows) else None
    for method in ["iv", "variance"]:
        error = 100 * (scored[f"{method}_prediction"] - scored.actual)
        output[method] = {
            "mae_iv_percentage_points": float(error.abs().mean()) if len(error) else None,
            "rmse_iv_percentage_points": float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "median_absolute_error_pp": float(error.abs().median()) if len(error) else None,
            "p95_absolute_error_pp": float(error.abs().quantile(.95)) if len(error) else None,
        }
    by_split = scored.assign(
        iv_error=(scored.iv_prediction - scored.actual).abs(),
        variance_error=(scored.variance_prediction - scored.actual).abs(),
    ).groupby("split")[["iv_error", "variance_error"]].mean()
    output["scored_splits"] = len(by_split)
    output["variance_mae_wins"] = int((by_split.variance_error < by_split.iv_error).sum())
    return output


def evaluate(observed, mode, rescale):
    splits = []
    if mode == "random_20pct":
        for seed in range(20261013, 20261063):
            rng = np.random.default_rng(seed)
            held = np.concatenate([
                rng.choice(group.index, max(1, int(np.ceil(.2 * len(group)))), replace=False)
                for _, group in observed.groupby("tau")
            ])
            splits.append((str(seed), held))
    else:
        splits = [(str(tau), group.index.to_numpy()) for tau, group in observed.groupby("tau")]
    rows = []
    split_results = []
    for label, held in splits:
        test = observed.loc[held]
        train = observed.drop(index=held)
        assert not train.index.isin(test.index).any()
        prediction = predict_pair(train, test, rescale)
        part = test[["tau", "moneyness", "source"]].copy()
        part["actual"] = test.iv
        part["iv_prediction"] = prediction[:, 0]
        part["variance_prediction"] = prediction[:, 1]
        part["split"] = label
        rows.append(part)
        split_results.append({"split": label, **summarize(part)})
    all_rows = pd.concat(rows, ignore_index=True)
    return {
        "pooled": summarize(all_rows),
        "by_source": {str(key): summarize(group) for key, group in all_rows.groupby("source")},
        "splits": split_results,
    }, all_rows


def sanity_checks():
    train = pd.DataFrame({"moneyness": [.8, 1.2, .8, 1.2], "tau": [.1, .1, .5, .5]})
    test = pd.DataFrame({"moneyness": [1., 2.], "tau": [.3, .3]})
    for rescale in [False, True]:
        train["iv"] = .2 + .1 * train.moneyness + .2 * train.tau
        predictions = predict_pair(train, test, rescale)
        assert np.isclose(predictions[0, 0], .36)
        assert np.isnan(predictions[1]).all()
        train["iv"] = np.sqrt((.02 + .01 * train.moneyness + .03 * train.tau) / train.tau)
        predictions = predict_pair(train, test, rescale)
        assert np.isclose(predictions[0, 1], np.sqrt(.039 / .3))


def main():
    sanity_checks()
    path = ROOT / "data/bybit_btc_options_snapshot.json"
    snapshot = load_chain_snapshot(path)
    chain = parse_chain_snapshot(snapshot)
    result = build_combined_surface_from_chain(chain, r=0)
    frame = surface_to_long_frame(result.iv_surface, result.expiries, result.strikes,
                                  result.spot_price, source_grid=result.source_grid)
    assert not frame.duplicated(["tau", "moneyness"]).any()
    assert frame.tau.gt(0).all()
    observed_all = frame[np.isfinite(frame.iv)]
    evidence = {
        "snapshot": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "captured_at_utc": snapshot["captured_at_utc"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
        "rate": 0, "spot_price": result.spot_price,
        "grid_cells": len(frame), "finite_combined_cells": len(observed_all),
        "method": "Shared Delaunay mesh; linear IV versus linear IV**2*tau, recovered with sqrt(w/tau).",
        "caveats": [
            "One frozen snapshot, not independent market dates or ground-truth latent volatility.",
            "Targets are solved Combined IV, which may contain quote/model noise and side fallbacks.",
            "Random split results reuse observations; 50 seeds are not 50 independent samples.",
            "Errors conditional on shared convex-hull coverage; no extrapolation.",
            "Coordinate scaling affects triangulation; both scaled and native-unit results retained.",
            "Holdout observations differ from systematically missing quotes; results do not establish missing-cell accuracy.",
            "Neither method enforces arbitrage constraints or guarantees visual smoothness.",
        ], "ranges": {},
    }
    predictions = []
    for name, lower, upper in [("core", .8, 1.2), ("extended", .7, 1.3)]:
        grid = frame[frame.moneyness.between(lower, upper)].copy()
        observed = grid[np.isfinite(grid.iv)].copy()
        section = {"grid_cells": len(grid), "observed": len(observed),
                   "expiries": observed.tau.nunique(), "experiments": {}}
        for rescale in [True, False]:
            geometry = "unit_range_axes" if rescale else "native_axes"
            for mode in ["random_20pct", "leave_one_expiry_out"]:
                summary, rows = evaluate(observed, mode, rescale)
                section["experiments"][f"{geometry}/{mode}"] = summary
                predictions.append(rows.assign(range=name, geometry=geometry, experiment=mode))
        one_d = interpolate_smiles_1d(grid, min_moneyness=lower, max_moneyness=upper)
        missing = grid[~np.isfinite(grid.iv)]
        pair = predict_pair(observed, missing, True)
        section["missing_coverage"] = {
            "original_missing": len(missing),
            "filled_1d": int((np.isfinite(one_d.iv) & ~np.isfinite(one_d.observed_iv)).sum()),
            "filled_2d": int(np.isfinite(pair[:, 0]).sum()),
        }
        evidence["ranges"][name] = section
        print(name, json.dumps({k: v["pooled"] for k, v in section["experiments"].items()}, indent=2))
        print("missing coverage", section["missing_coverage"])
    output = Path(__file__).resolve().parent
    (output / "interpolation_comparison.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    pd.concat(predictions, ignore_index=True).to_csv(output / "interpolation_holdout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
