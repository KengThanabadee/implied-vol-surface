import numpy as np
import pandas as pd


_LONG_FRAME_COLUMNS = {"tau", "strike", "iv", "moneyness"}
_ATM_TOLERANCE = 1e-12
_SURFACE_VALUE_TYPES = {
    "observed": "Observed",
    "interpolated": "Interpolated",
    "missing": "Missing",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_expiry_datetimes(
    expiry_datetimes, expiry_count: int
) -> pd.Series | None:
    if expiry_datetimes is None:
        return None

    try:
        values = list(expiry_datetimes)
    except TypeError as exc:
        raise ValueError(
            "expiry_datetimes must contain len(expiries) values"
        ) from exc
    if len(values) != expiry_count:
        raise ValueError("expiry_datetimes must contain len(expiries) values")

    try:
        normalized = pd.to_datetime(
            values, utc=True, errors="raise", format="mixed"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "expiry_datetimes must contain valid datetime values"
        ) from exc
    return pd.Series(normalized, dtype="datetime64[ns, UTC]")


def surface_to_long_frame(
    iv_surface, expiries, strikes, spot_price, expiry_datetimes=None,
    source_grid=None,
) -> pd.DataFrame:
    """Convert an expiry-by-strike IV grid to chart-ready long-form data.

    Optional source_grid labels must align with iv_surface and are preserved
    unchanged in a source column, including labels for missing IV cells.
    """
    try:
        surface = np.asarray(iv_surface, dtype=float)
        expiry_values = np.asarray(list(expiries), dtype=float)
        strike_values = np.asarray(list(strikes), dtype=float)
        spot_value = float(spot_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("surface inputs must be numeric") from exc

    expected_shape = (len(expiry_values), len(strike_values))
    if surface.ndim != 2 or surface.shape != expected_shape:
        raise ValueError(
            "iv_surface shape must equal (len(expiries), len(strikes))"
        )
    if not np.isfinite(spot_value) or spot_value <= 0:
        raise ValueError("spot_price must be finite and positive")

    expiry_datetime_values = _normalize_expiry_datetimes(
        expiry_datetimes, len(expiry_values)
    )
    tau = np.repeat(expiry_values, len(strike_values))
    strike = np.tile(strike_values, len(expiry_values))
    data = {"tau": tau}
    if expiry_datetime_values is not None:
        data["expiry_datetime"] = expiry_datetime_values.repeat(
            len(strike_values)
        ).reset_index(drop=True)
    data.update(
        {
            "strike": strike,
            "iv": surface.reshape(-1),
            "spot_price": spot_value,
            "moneyness": strike / spot_value,
        }
    )
    if source_grid is not None:
        sources = np.asarray(source_grid, dtype=object)
        if sources.shape != surface.shape:
            raise ValueError("source_grid shape must equal iv_surface shape")
        data["source"] = sources.reshape(-1)
    return pd.DataFrame(data)


def interpolate_smiles_1d(
    long_frame: pd.DataFrame,
    min_moneyness: float = 0.8,
    max_moneyness: float = 1.2,
) -> pd.DataFrame:
    """Linearly fill interior IV gaps along moneyness within each expiry.

    The returned frame is restricted to the inclusive moneyness range. Original
    finite values are retained as Observed, missing cells between the first and
    last finite value of an expiry are Interpolated, and cells outside that
    observed support remain Missing. No values are filled across expiries.
    """
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    try:
        lower = float(min_moneyness)
        upper = float(max_moneyness)
    except (TypeError, ValueError) as exc:
        raise ValueError("moneyness bounds must be finite with min < max") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("moneyness bounds must be finite with min < max")

    duplicates = long_frame.duplicated(subset=["tau", "moneyness"], keep=False)
    if duplicates.any():
        raise ValueError("long_frame must have unique tau and moneyness cells")

    frame = long_frame[
        long_frame["moneyness"].between(lower, upper, inclusive="both")
    ].copy()
    # Use unique internal row labels without changing the caller's index.
    original_index = frame.index
    frame = frame.reset_index(drop=True)
    frame["observed_iv"] = pd.to_numeric(frame["iv"], errors="coerce")
    frame["surface_value_type"] = _SURFACE_VALUE_TYPES["missing"]

    for _, expiry_frame in frame.groupby("tau", sort=False, dropna=False):
        ordered = expiry_frame.sort_values("moneyness")
        x = ordered["moneyness"].to_numpy(dtype=float)
        observed = ordered["observed_iv"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(observed)

        frame.loc[ordered.index[finite], "surface_value_type"] = (
            _SURFACE_VALUE_TYPES["observed"]
        )
        if finite.sum() < 2:
            continue

        interior = (
            ~finite
            & np.isfinite(x)
            & (x >= x[finite].min())
            & (x <= x[finite].max())
        )
        if not interior.any():
            continue

        target_index = ordered.index[interior]
        frame.loc[target_index, "iv"] = np.interp(
            x[interior], x[finite], observed[finite]
        )
        frame.loc[target_index, "surface_value_type"] = (
            _SURFACE_VALUE_TYPES["interpolated"]
        )

    frame.index = original_index
    return frame


def atm_term_structure_frame(
    long_frame: pd.DataFrame, max_moneyness_distance: float = 0.02
) -> pd.DataFrame:
    """Select a bounded, transparent ATM proxy independently for each expiry."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    try:
        distance_bound = float(max_moneyness_distance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_moneyness_distance must be finite and non-negative"
        ) from exc
    if not np.isfinite(distance_bound) or distance_bound < 0:
        raise ValueError("max_moneyness_distance must be finite and non-negative")

    has_expiry_metadata = "expiry_datetime" in long_frame.columns
    rows = []
    for tau, expiry_frame in long_frame.groupby("tau", sort=True, dropna=False):
        metadata = (
            {"expiry_datetime": expiry_frame["expiry_datetime"].iloc[0]}
            if has_expiry_metadata
            else {}
        )
        iv = expiry_frame["iv"].to_numpy(dtype=float)
        moneyness = expiry_frame["moneyness"].to_numpy(dtype=float)
        finite = np.isfinite(iv) & np.isfinite(moneyness)

        if not finite.any():
            rows.append(
                {
                    "tau": tau,
                    **metadata,
                    "strike": np.nan,
                    "iv": np.nan,
                    "moneyness": np.nan,
                    "atm_distance": np.nan,
                    "selection_status": "no_finite_iv",
                }
            )
            continue

        candidates = expiry_frame.loc[finite].copy()
        candidates["atm_distance"] = np.abs(candidates["moneyness"] - 1.0)
        minimum_distance = float(candidates["atm_distance"].min())
        within_bound = minimum_distance < distance_bound or np.isclose(
            minimum_distance,
            distance_bound,
            atol=_ATM_TOLERANCE,
            rtol=0,
        )

        if not within_bound:
            rows.append(
                {
                    "tau": tau,
                    **metadata,
                    "strike": np.nan,
                    "iv": np.nan,
                    "moneyness": np.nan,
                    "atm_distance": minimum_distance,
                    "selection_status": "outside_bound",
                }
            )
            continue

        tied = candidates[
            np.isclose(
                candidates["atm_distance"],
                minimum_distance,
                atol=_ATM_TOLERANCE,
                rtol=0,
            )
        ]
        rows.append(
            {
                "tau": tau,
                **metadata,
                "strike": float(tied["strike"].mean()),
                "iv": float(tied["iv"].mean()),
                "moneyness": float(tied["moneyness"].mean()),
                "atm_distance": minimum_distance,
                "selection_status": (
                    "symmetric_average" if len(tied) > 1 else "nearest"
                ),
            }
        )

    columns = ["tau"]
    if has_expiry_metadata:
        columns.append("expiry_datetime")
    columns.extend(
        [
            "strike",
            "iv",
            "moneyness",
            "atm_distance",
            "selection_status",
        ]
    )
    return pd.DataFrame(rows, columns=columns)
