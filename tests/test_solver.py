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

def test_solve_iv_rejects_non_finite_price():
    with pytest.raises(ValueError):
        solve_iv(np.inf, 100, 100, 1, 0.03, "call")

    with pytest.raises(ValueError):
        solve_iv(-np.inf, 100, 100, 1, 0.03, "call")

    with pytest.raises(ValueError):
        solve_iv(np.nan, 100, 100, 1, 0.03, "call")

def test_solve_iv_returns_lower_bracket_endpoint():
    S = 100
    K = 100
    T = 1
    r = 0.03
    sigma_low = 1e-6
    flag = "call"

    price = bs_price(S, K, T, r, sigma_low, flag)
    implied_vol = solve_iv(price, S, K, T, r, flag)

    assert abs(implied_vol - sigma_low) < 1e-4

def test_solve_iv_returns_upper_bracket_endpoint():
    S = 100
    K = 100
    T = 1
    r = 0.03
    sigma_high = 10.0
    flag = "call"

    price = bs_price(S, K, T, r, sigma_high, flag)
    implied_vol = solve_iv(price, S, K, T, r, flag)

    assert abs(implied_vol - sigma_high) < 1e-4

def test_solve_iv_rejects_expired_option():
    with pytest.raises(ValueError):
        solve_iv(0, 100, 100, 0, 0.03, "call")

def test_solve_iv_rejects_non_finite_inputs():
    cases = [
        {"price": 10, "S": np.nan, "K": 100, "T": 1, "r": 0.03, "flag": "call"},
        {"price": 10, "S": 100, "K": np.nan, "T": 1, "r": 0.03, "flag": "call"},
        {"price": 10, "S": 100, "K": 100, "T": np.nan, "r": 0.03, "flag": "call"},
        {"price": 10, "S": 100, "K": 100, "T": 1, "r": np.nan, "flag": "call"},
    ]

    for kwargs in cases:
        with pytest.raises(ValueError):
            solve_iv(**kwargs)

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

def test_build_surface_wrong_price_shape():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    prices = np.ones((2, 3))

    with pytest.raises(ValueError):
        build_surface(prices, S=100, strikes=strikes, expiries=expiries)

def test_interpolate_iv_flat_surface_from_prices():
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
    assert abs(iv - sigma) < 1e-4

def test_interpolate_iv_non_flat_surface():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(surface, strikes, expiries, K=95, T=0.175)

    assert abs(iv - 0.195) < 1e-4

def test_interpolate_iv_out_of_bounds():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.full((3, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(surface, strikes, expiries, K=85, T=0.25)

    with pytest.raises(ValueError):
        interpolate_iv(surface, strikes, expiries, K=100, T=0.05)

def test_interpolate_iv_wrong_surface_shape():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.full((2, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(surface, strikes, expiries, K=100, T=0.25)

def test_interpolate_iv_rejects_singleton_grid_dimension():
    with pytest.raises(ValueError):
        interpolate_iv(np.array([[0.2, 0.3]]), [100, 110], [1], K=105, T=1)

    with pytest.raises(ValueError):
        interpolate_iv(np.array([[0.2], [0.3]]), [100], [1, 2], K=100, T=1.5)

def test_interpolate_iv_rejects_unsorted_or_duplicate_grid():
    surface = np.full((3, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(surface, [90, 110, 100], [0.1, 0.25, 0.5], K=100, T=0.25)

    with pytest.raises(ValueError):
        interpolate_iv(surface, [90, 100, 110], [0.1, 0.25, 0.25], K=100, T=0.25)

def test_interpolate_iv_exact_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])
    iv = interpolate_iv(surface, strikes, expiries, K=100, T=0.25)

    assert abs(iv - 0.21) < 1e-4

def test_interpolate_iv_upper_boundary_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(surface, strikes, expiries, K=110, T=0.5)

    assert abs(iv - 0.24) < 1e-4

def test_interpolate_iv_lower_boundary_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(surface, strikes, expiries, K=90, T=0.1)

    assert abs(iv - 0.18) < 1e-4
