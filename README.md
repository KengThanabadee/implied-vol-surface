# iv-surface

A small personal quant library for learning and building implied-volatility tooling from first principles.

`iv-surface` provides Black-Scholes pricing helpers, implied-volatility root finding, simple surface construction, bilinear interpolation, and basic Bybit option-chain fetching.

The implied-volatility root-finding logic is implemented without `scipy.optimize`.

## Status

This is an early local package in a personal quant-library ecosystem. It is useful for experiments and for feeding downstream projects, but it is not production trading infrastructure and is not financial advice.

## What It Does

- Prices European calls and puts with Black-Scholes.
- Solves implied volatility with Newton-Raphson and bisection fallback.
- Builds a 2D IV surface from a grid of option prices.
- Interpolates IV inside an expiry/strike grid with bilinear interpolation.
- Fetches and parses Bybit option-chain ticker data into a DataFrame.

## Core Data Shape

Surface arrays use expiry rows and strike columns:

```text
surface.shape == (len(expiries), len(strikes))
surface[i, j] == IV at expiries[i], strikes[j]
```

Example layout:

```text
              K=90   K=100   K=110
T=0.10        0.18    0.20    0.22
T=0.25        0.19    0.21    0.23
T=0.50        0.20    0.22    0.24
```

Accordingly, the surface APIs take `expiries` before `strikes`, and interpolation uses `target_T` and `target_K`.

## Installation

Install locally in editable mode:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Example

```python
import numpy as np

from iv_surface.solver import bs_price, build_surface, interpolate_iv

S = 100
r = 0.03
sigma = 0.20

expiries = [0.10, 0.25, 0.50]
strikes = [90, 100, 110]

prices = np.array([
    [bs_price(S, K, T, r, sigma, "call") for K in strikes]
    for T in expiries
])

surface = build_surface(prices, S, expiries, strikes, r, flag="call")
iv = interpolate_iv(surface, expiries, strikes, target_T=0.18, target_K=97)

print(surface)
print(iv)
```

## API

### `bs_price(S, K, T, r, sigma, flag, t=0)`

Returns the Black-Scholes price for a European call or put.

### `solve_iv(price, S, K, T, r, flag, sigma_low=1e-6, sigma_high=10.0)`

Returns implied volatility for a single option price. Raises `ValueError` for invalid inputs, expired options, or prices outside the bracket implied by the sigma bounds.

### `build_surface(prices, S, expiries, strikes, r=0, flag="call")`

Returns a 2D NumPy array of implied volatilities with shape `(n_expiries, n_strikes)`. Cells that cannot be solved are filled with `NaN`.

### `interpolate_iv(surface, expiries, strikes, target_T, target_K)`

Returns bilinearly interpolated IV at the target `(T, K)`. Raises `ValueError` if the target point is outside the grid or if the grid is not valid.

### `fetch_chain(underlying="BTC")`

Fetches Bybit option tickers for an underlying and returns a tidy pandas DataFrame with parsed symbol fields, mark price, underlying price, and time to expiry.

## Assumptions And Limitations

- Uses Black-Scholes European option assumptions.
- Time to expiry is measured in years.
- No dividend yield parameter yet.
- IV solving requires `T > 0`; implied volatility is undefined at expiry.
- `build_surface` fills failed IV solves with `NaN`.
- Interpolation requires at least two strictly increasing expiries and strikes.
- Interpolation is in-grid only; no extrapolation.
- The surface is a raw interpolated grid, not an arbitrage-free fitted surface.

## Tests

```bash
pytest
```
