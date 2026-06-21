import argparse
import sys

import numpy as np
import requests

from iv_surface.collector import build_surface_from_chain, prepare_surface_inputs
from iv_surface.fetcher import fetch_chain


def _nan_ratio(values):
    if values.size == 0:
        return np.nan
    return float(np.isnan(values).mean())


def _print_flag_summary(chain, flag, r):
    flag_rows = chain[chain["flag"] == flag]
    usable_rows = flag_rows[
        (flag_rows["quote_source"] == "mid") & np.isfinite(flag_rows["mid_price"])
    ]

    print(f"\n[{flag}]")
    print(f"rows: {len(flag_rows)}")
    print(f"usable_mid_rows: {len(usable_rows)}")

    if usable_rows.empty:
        print("surface_inputs: no usable mid quotes")
        return

    inputs = prepare_surface_inputs(chain, flag=flag)
    result = build_surface_from_chain(chain, flag=flag, r=r)

    print(f"spot_price: {inputs.spot_price:.8g}")
    print(f"expiries_count: {len(inputs.expiries)}")
    print(f"strikes_count: {len(inputs.strikes)}")
    print(f"option_price_grid_shape: {inputs.option_price_grid.shape}")
    print(f"option_price_grid_nan_ratio: {_nan_ratio(inputs.option_price_grid):.2%}")
    print(f"iv_surface_nan_ratio: {_nan_ratio(result.iv_surface):.2%}")

    if inputs.expiries:
        print(f"expiry_range_tau: {inputs.expiries[0]:.6g} -> {inputs.expiries[-1]:.6g}")
    if inputs.strikes:
        print(f"strike_range: {inputs.strikes[0]:.8g} -> {inputs.strikes[-1]:.8g}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual live Bybit sanity check for option-chain to IV surface flow."
    )
    parser.add_argument("--underlying", default="BTC", help="Bybit option base coin")
    parser.add_argument("--r", type=float, default=0.0, help="Risk-free rate")
    args = parser.parse_args()

    try:
        chain = fetch_chain(args.underlying)
    except requests.RequestException as exc:
        print(f"Bybit request failed: {exc}", file=sys.stderr)
        return 1

    print(f"underlying: {args.underlying}")
    print(f"rows: {len(chain)}")
    if chain.empty:
        return 0

    print("flag_counts:")
    print(chain["flag"].value_counts(dropna=False).to_string())
    print(f"usable_mid_rows: {int((chain['quote_source'] == 'mid').sum())}")

    for flag in ["call", "put"]:
        _print_flag_summary(chain, flag=flag, r=args.r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
