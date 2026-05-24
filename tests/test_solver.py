import numpy as np
import pytest
from iv_surface.solver import bs_price, solve_iv, build_surface

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
