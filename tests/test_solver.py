import pytest
from iv_surface.solver import bs_price, solve_iv

def test_round_trip():
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