import argparse
import sys
import warnings

import numpy as np
import requests

from iv_surface.collector import _filter_usable_chain_rows, prepare_surface_inputs
from iv_surface.fetcher import DEFAULT_BYBIT_BASE_URL, DEFAULT_BYBIT_TIMEOUT, fetch_chain
from iv_surface.solver import build_surface


def _nan_ratio(values):
    if values.size == 0:
        return np.nan
    return float(np.isnan(values).mean())


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _print_underlying_price_summary(usable_rows, selected_spot_price):
    underlying_prices = usable_rows["underlying_price"].dropna()
    if underlying_prices.empty:
        return

    min_price = float(underlying_prices.min())
    median_price = float(underlying_prices.median())
    max_price = float(underlying_prices.max())
    spread = max_price - min_price
    print(
        "underlying_price: "
        f"selected_spot_price={selected_spot_price:.8g}, "
        f"median={median_price:.8g}, min={min_price:.8g}, "
        f"max={max_price:.8g}, spread={spread:.8g}"
    )
    if underlying_prices.nunique() > 1:
        print("warning: underlying_price differs across usable rows; using median")
    if not np.isclose(selected_spot_price, median_price, rtol=0.0, atol=1e-4):
        print("warning: selected_spot_price differs from usable underlying_price median")


def _print_flag_summary(chain, flag, r):
    flag_rows = chain[chain["flag"] == flag]
    usable_rows = _filter_usable_chain_rows(chain, flag)

    print(f"\n[{flag}]")
    print(f"rows: {len(flag_rows)}")
    print(f"usable_chain_rows: {len(usable_rows)}")

    if usable_rows.empty:
        print("surface_inputs: no usable mid quotes")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="usable rows have different underlying_price values.*",
            category=UserWarning,
        )
        inputs = prepare_surface_inputs(chain, flag=flag)

    iv_surface = build_surface(
        inputs.option_price_grid,
        inputs.spot_price,
        inputs.expiries,
        inputs.strikes,
        r,
        flag,
    )

    print(f"spot_price: {inputs.spot_price:.8g}")
    _print_underlying_price_summary(usable_rows, inputs.spot_price)
    print(f"expiries_count: {len(inputs.expiries)}")
    print(f"strikes_count: {len(inputs.strikes)}")
    print(f"option_price_grid_shape: {inputs.option_price_grid.shape}")
    print(f"option_price_grid_nan_ratio: {_nan_ratio(inputs.option_price_grid):.2%}")
    print(f"iv_surface_nan_ratio: {_nan_ratio(iv_surface):.2%}")

    if inputs.expiries:
        print(f"expiry_range_tau: {inputs.expiries[0]:.6g} -> {inputs.expiries[-1]:.6g}")
    if inputs.strikes:
        print(f"strike_range: {inputs.strikes[0]:.8g} -> {inputs.strikes[-1]:.8g}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual live Bybit sanity check for option-chain to IV surface flow."
    )
    parser.add_argument("--underlying", default="BTC", help="Bybit option base coin")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BYBIT_BASE_URL,
        help="Bybit REST API base URL",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_BYBIT_TIMEOUT,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument("--r", type=float, default=0.0, help="Risk-free rate")
    args = parser.parse_args()

    try:
        chain = fetch_chain(args.underlying, base_url=args.base_url, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"Bybit request failed: {exc}", file=sys.stderr)
        return 1

    print(f"underlying: {args.underlying}")
    print(f"base_url: {args.base_url}")
    print(f"timeout: {args.timeout:g}")
    print(f"rows: {len(chain)}")
    if chain.empty:
        return 0

    print("flag_counts:")
    print(chain["flag"].value_counts(dropna=False).to_string())
    usable_chain_rows = sum(
        len(_filter_usable_chain_rows(chain, flag)) for flag in ["call", "put"]
    )
    print(f"usable_chain_rows: {usable_chain_rows}")

    for flag in ["call", "put"]:
        _print_flag_summary(chain, flag=flag, r=args.r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
