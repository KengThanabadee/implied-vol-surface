# iv-surface

Implied volatility solver and surface interpolation library, built from scratch.

Part of a personal quant library ecosystem feeding into a live delta/gamma hedging bot for BTC options on Bybit.

## What it does

Given an option market price, **`solve_iv`** inverts the Black-Scholes formula to find the implied volatility using Newton-Raphson with bisection fallback.

**`build_surface`** runs the solver across a grid of strikes and expiries, returning a 2D array of implied volatilities.

**`interpolate_iv`** queries IV at any arbitrary (K, T) point using bilinear interpolation — bridging the gap between discrete market quotes and continuous hedging needs.

## Installation

```bash
pip install -e .
```

## Usage

```python
from iv_surface.solver import bs_price, solve_iv, build_surface, interpolate_iv
import numpy as np

S = 100
strikes = [90, 100, 110]
expiries = [0.1, 0.25, 0.5]
r = 0

# Build price grid from market quotes
prices = np.array([
    [bs_price(S, K, T, r, sigma=0.2, flag="call") for K in strikes]
    for T in expiries
])

# Solve IV surface
surface = build_surface(prices, S, strikes, expiries, r)

# Query IV at arbitrary point
iv = interpolate_iv(surface, strikes, expiries, K=97, T=0.18)
```

## API

### `solve_iv(price, S, K, T, r, flag)`
Returns implied volatility for a single option. Raises `ValueError` if price is outside no-arbitrage bounds or non-finite.

### `build_surface(prices, S, strikes, expiries, r, flag)`
Returns a 2D numpy array of shape `(n_expiries, n_strikes)`. Cells where IV cannot be solved are filled with `NaN`.

### `interpolate_iv(surface, strikes, expiries, K, T)`
Returns interpolated IV at target `(K, T)`. Raises `ValueError` if target is outside the grid.

## Running tests

```bash
pytest
```
