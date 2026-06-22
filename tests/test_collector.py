import numpy as np
import pandas as pd
import pytest

from iv_surface.collector import (
    SurfaceInputs,
    SurfaceResult,
    build_surface_from_chain,
    prepare_surface_inputs,
)
from iv_surface.solver import bs_price


def _row(
    flag="call",
    tau=0.25,
    strike=100,
    mid_price=10,
    quote_source="mid",
    underlying_price=100,
):
    return {
        "flag": flag,
        "tau": tau,
        "strike": strike,
        "mid_price": mid_price,
        "quote_source": quote_source,
        "underlying_price": underlying_price,
    }


def test_prepare_surface_inputs_filters_sorts_and_keeps_missing_cells():
    chain = pd.DataFrame(
        [
            _row(tau=0.5, strike=100, mid_price=11),
            _row(tau=0.25, strike=90, mid_price=10),
            _row(tau=0.25, strike=100, mid_price=12, underlying_price=100),
            _row(flag="put", tau=0.25, strike=90, mid_price=99),
            _row(tau=0.5, strike=110, mid_price=np.nan),
            _row(tau=0.5, strike=120, mid_price=20, quote_source="none"),
        ]
    )

    result = prepare_surface_inputs(chain, flag="call")

    assert isinstance(result, SurfaceInputs)
    assert result.spot_price == 100
    assert result.expiries == [0.25, 0.5]
    assert result.strikes == [90, 100]
    assert result.option_price_grid.shape == (2, 2)
    assert result.option_price_grid[0, 0] == 10
    assert result.option_price_grid[0, 1] == 12
    assert np.isnan(result.option_price_grid[1, 0])
    assert result.option_price_grid[1, 1] == 11


def test_prepare_surface_inputs_warns_when_underlying_prices_differ():
    chain = pd.DataFrame(
        [
            _row(tau=0.25, strike=90, mid_price=10, underlying_price=99),
            _row(tau=0.25, strike=100, mid_price=12, underlying_price=100),
            _row(tau=0.5, strike=100, mid_price=11, underlying_price=101),
        ]
    )

    with pytest.warns(UserWarning, match="different underlying_price"):
        result = prepare_surface_inputs(chain, flag="call")

    assert result.spot_price == 100


def test_prepare_surface_inputs_uses_selected_put_flag():
    chain = pd.DataFrame(
        [
            _row(flag="call", tau=0.25, strike=100, mid_price=10),
            _row(flag="put", tau=0.25, strike=100, mid_price=8),
        ]
    )

    result = prepare_surface_inputs(chain, flag="put")

    assert result.option_price_grid.tolist() == [[8.0]]


def test_prepare_surface_inputs_rejects_invalid_flag():
    chain = pd.DataFrame([_row()])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain, flag="straddle")


def test_prepare_surface_inputs_rejects_missing_columns():
    chain = pd.DataFrame([_row()]).drop(columns=["quote_source"])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_prepare_surface_inputs_rejects_no_usable_rows():
    chain = pd.DataFrame([_row(mid_price=np.nan), _row(quote_source="none")])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_prepare_surface_inputs_rejects_duplicate_tau_strike():
    chain = pd.DataFrame(
        [
            _row(tau=0.25, strike=100, mid_price=10),
            _row(tau=0.25, strike=100, mid_price=11),
        ]
    )

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_build_surface_from_chain_solves_iv_surface_from_mid_prices():
    spot_price = 100
    r = 0.01
    sigma = 0.2
    expiries = [0.25, 0.5]
    strikes = [90, 100]
    rows = []

    for T in expiries:
        for K in strikes:
            rows.append(
                _row(
                    tau=T,
                    strike=K,
                    mid_price=bs_price(spot_price, K, T, r, sigma, "call"),
                    underlying_price=spot_price,
                )
            )

    result = build_surface_from_chain(pd.DataFrame(rows), flag="call", r=r)

    assert isinstance(result, SurfaceResult)
    assert result.spot_price == spot_price
    assert result.expiries == expiries
    assert result.strikes == strikes
    assert np.allclose(result.iv_surface, sigma, rtol=0.0, atol=1e-4)
