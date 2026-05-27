import numpy as np
import pytest
from iv_surface.solver import bs_price, solve_iv, build_surface, interpolate_iv

def test_solve_iv_round_trip():
    S = 100
    K = 100
    T = 1
    r = 0.03
    sigma = 0.2
    flag = "call"

    price = bs_price(S, K, T, r, sigma, flag)
    implied_vol = solve_iv(price, S, K, T, r, flag)
    assert abs(implied_vol - sigma) < 1e-4

def test_solve_iv_bad_price():
    with pytest.raises(ValueError):
        solve_iv(101, 100, 100, 1, 0.03, "call")

def test_build_surface_round_trip():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    sigma = 0.2
    S = 100
    r = 0

    prices = np.array([
        [bs_price(S, K, T, r, sigma, "call") for K in strikes]
        for T in expiries
    ])

    surface = build_surface(prices, S, strikes, expiries, r)

    assert np.allclose(sigma, surface, atol=1e-4)

def test_interpolate_iv_flat_surface():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    sigma = 0.2
    S = 100
    r = 0

    prices = np.array([
        [bs_price(S, K, T, r, sigma, "call") for K in strikes]
        for T in expiries
    ])
    surface = build_surface(prices, S, strikes, expiries, r)

    # query a point strictly inside the grid
    iv = interpolate_iv(surface, strikes, expiries, K=97, T=0.18)
    assert abs(iv - sigma) < 1e-3

def test_interpolate_iv_out_of_bounds():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.full((3, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(surface, strikes, expiries, K=85, T=0.25)

    with pytest.raises(ValueError):
        interpolate_iv(surface, strikes, expiries, K=100, T=0.05)
