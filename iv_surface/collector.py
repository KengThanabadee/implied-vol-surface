from dataclasses import dataclass

import numpy as np
import pandas as pd

from iv_surface.solver import build_surface


_REQUIRED_COLUMNS = {
    "flag",
    "tau",
    "strike",
    "mid_price",
    "quote_source",
    "underlying_price",
}
_VALID_FLAGS = {"call", "put"}


@dataclass(frozen=True)
class SurfaceInputs:
    prices: np.ndarray
    S: float
    expiries: list[float]
    strikes: list[float]


@dataclass(frozen=True)
class SurfaceResult:
    surface: np.ndarray
    prices: np.ndarray
    S: float
    expiries: list[float]
    strikes: list[float]


def _validate_chain(chain: pd.DataFrame, flag: str) -> None:
    if not isinstance(chain, pd.DataFrame):
        raise TypeError("chain must be a pandas DataFrame")
    if flag not in _VALID_FLAGS:
        raise ValueError("flag must be 'call' or 'put'")

    missing = sorted(_REQUIRED_COLUMNS - set(chain.columns))
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")


def prepare_surface_inputs(chain: pd.DataFrame, flag: str = "call") -> SurfaceInputs:
    _validate_chain(chain, flag)

    data = chain.copy()
    for column in ["tau", "strike", "mid_price", "underlying_price"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    usable = data[
        (data["flag"] == flag)
        & (data["quote_source"] == "mid")
        & np.isfinite(data["mid_price"])
        & np.isfinite(data["tau"])
        & np.isfinite(data["strike"])
        & np.isfinite(data["underlying_price"])
        & (data["mid_price"] > 0)
        & (data["tau"] > 0)
        & (data["strike"] > 0)
        & (data["underlying_price"] > 0)
    ].copy()

    if usable.empty:
        raise ValueError(f"chain has no usable {flag} rows with valid mid prices")

    duplicates = usable.duplicated(subset=["tau", "strike"], keep=False)
    if duplicates.any():
        raise ValueError("chain contains duplicate rows for the same tau and strike")

    expiries = sorted(usable["tau"].unique().tolist())
    strikes = sorted(usable["strike"].unique().tolist())

    price_grid = usable.pivot(index="tau", columns="strike", values="mid_price")
    prices = price_grid.reindex(index=expiries, columns=strikes).to_numpy(dtype=float)
    S = float(usable["underlying_price"].median())

    return SurfaceInputs(prices=prices, S=S, expiries=expiries, strikes=strikes)


def build_surface_from_chain(
    chain: pd.DataFrame, flag: str = "call", r: float = 0
) -> SurfaceResult:
    inputs = prepare_surface_inputs(chain, flag=flag)
    surface = build_surface(inputs.prices, inputs.S, inputs.expiries, inputs.strikes, r, flag)

    return SurfaceResult(
        surface=surface,
        prices=inputs.prices,
        S=inputs.S,
        expiries=inputs.expiries,
        strikes=inputs.strikes,
    )
