from datetime import datetime, timezone

import numpy as np
import pytest

from iv_surface.fetcher import _to_float, compute_tau, parse_symbol


def test_to_float_returns_nan_for_bad_values():
    assert _to_float("123.45") == 123.45
    assert np.isnan(_to_float(None))
    assert np.isnan(_to_float("not-a-number"))
    assert np.isnan(_to_float(np.inf))


def test_parse_symbol_call_and_put():
    call = parse_symbol("BTC-27JUN25-100000-C")
    put = parse_symbol("BTC-27JUN25-100000-P")

    assert call["underlying"] == "BTC"
    assert call["expiry_dt"] == datetime(2025, 6, 27, 8, tzinfo=timezone.utc)
    assert call["strike"] == 100000.0
    assert call["flag"] == "call"
    assert put["flag"] == "put"


def test_parse_symbol_rejects_invalid_flag():
    with pytest.raises(ValueError):
        parse_symbol("BTC-27JUN25-100000-X")


def test_compute_tau_clamps_expired_time_to_zero():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert compute_tau(now, now) == 0.0
    assert compute_tau(datetime(2025, 12, 31, tzinfo=timezone.utc), now) == 0.0


def test_compute_tau_one_year():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expiry = datetime(2027, 1, 1, 6, tzinfo=timezone.utc)

    assert abs(compute_tau(expiry, now) - 1.0) < 1e-12
