from collections import Counter

import numpy as np
import pandas as pd

from iv_surface.solver import bs_price
from scripts.check_bybit_chain import (
    _classify_solve_error,
    _compare_median_spot_to_row_spot,
    _finite_count,
    _solve_iv_grid_diagnostics,
)


def test_finite_count_counts_non_nan_cells():
    values = np.array([[1.0, np.nan], [2.0, np.inf]])

    assert _finite_count(values) == 2


def test_solve_iv_grid_diagnostics_counts_missing_and_bound_failures():
    spot_price = 100
    expiries = [0.25]
    strikes = [100, 110, 120]
    valid_price = bs_price(spot_price, 100, 0.25, 0.0, 0.2, "call")
    option_price_grid = np.array([[valid_price, np.nan, 200.0]])

    iv_surface, failures = _solve_iv_grid_diagnostics(
        option_price_grid, spot_price, expiries, strikes, r=0.0, flag="call"
    )

    assert np.isfinite(iv_surface[0, 0])
    assert np.isnan(iv_surface[0, 1])
    assert np.isnan(iv_surface[0, 2])
    assert failures == Counter(
        {"missing_option_price": 1, "outside_no_arbitrage_bounds": 1}
    )


def test_compare_median_spot_to_row_spot_detects_row_spot_fix():
    selected_spot_price = 200.0
    row_spot_that_differs_from_selected = 100.0
    row_spot_matching_selected = 200.0

    rows = pd.DataFrame(
        [
            {
                # This price is below intrinsic value under selected_spot_price.
                "mid_price": bs_price(
                    row_spot_that_differs_from_selected,
                    100.0,
                    0.25,
                    0.0,
                    0.2,
                    "call",
                ),
                "strike": 100.0,
                "tau": 0.25,
                "underlying_price": row_spot_that_differs_from_selected,
            },
            {
                "mid_price": bs_price(
                    row_spot_matching_selected,
                    100.0,
                    0.25,
                    0.0,
                    0.2,
                    "call",
                ),
                "strike": 100.0,
                "tau": 0.25,
                "underlying_price": row_spot_matching_selected,
            },
        ]
    )

    counts = _compare_median_spot_to_row_spot(
        rows, selected_spot_price=selected_spot_price, r=0.0, flag="call"
    )

    assert counts["total_usable_rows"] == 2
    assert counts["median_spot_solved"] == 1
    assert counts["row_spot_solved"] == 2
    assert counts["row_spot_fixes_median_failure"] == 1
    assert counts["median_spot_fixes_row_failure"] == 0
    assert counts["failed_both"] == 0


def test_classify_solve_error_groups_expected_messages():
    assert (
        _classify_solve_error("option_price 200 is outside no-arbitrage bounds")
        == "outside_no_arbitrage_bounds"
    )
    assert (
        _classify_solve_error("option_price must be finite, got nan")
        == "non_finite_input"
    )
    assert (
        _classify_solve_error("T must be > 0 for implied volatility")
        == "non_positive_tau"
    )
    assert _classify_solve_error("different problem") == "other_value_error"
