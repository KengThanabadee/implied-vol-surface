from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iv_surface.deribit import load_deribit_snapshot, parse_deribit_snapshot
from iv_surface.fetcher import compute_tau
from iv_surface.solver import bs_price
from iv_surface.surface_builder import build_combined_surface_from_chain


FIXTURE = Path(__file__).resolve().parents[1] / "data/deribit_btc_options_snapshot.json"


def snapshot():
    return load_deribit_snapshot(FIXTURE)


def test_real_snapshot_reconciles_and_keeps_frozen_metadata(monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise AssertionError("Snapshot parsing must be offline")

    monkeypatch.setattr(requests.sessions.Session, "request", fail)
    raw = snapshot()
    chain = parse_deribit_snapshot(raw)
    assert len(raw["requests"][0]["response"]["result"]) == len(chain) == 936
    assert len(raw["requests"][1]["response"]["result"]) == 936
    assert chain.tau.nunique() == 12
    assert chain.symbol.nunique() == 936
    assert chain.mid_price.isna().sum() == 36
    captured = datetime.fromisoformat(raw["captured_at_utc"].replace("Z", "+00:00"))
    assert (chain.tau == chain.expiry_dt.map(lambda dt: compute_tau(dt, captured))).all()
    assert all(dt.tzinfo == timezone.utc for dt in chain.expiry_dt)
    assert (chain.index_price == 82227.73).all()
    np.testing.assert_allclose(chain.bid_price, chain.bid_price_btc * chain.index_price)
    np.testing.assert_allclose(chain.ask_price, chain.ask_price_btc * chain.index_price)
    np.testing.assert_allclose(chain.mid_price, chain.mid_price_btc * chain.index_price)
    assert not chain.duplicated(["flag", "tau", "strike"]).any()
    assert raw == snapshot()  # The adapter never rewrites raw data.


def _one_contract():
    raw = snapshot()
    instrument = deepcopy(raw["requests"][0]["response"]["result"][0])
    book = next(deepcopy(b) for b in raw["requests"][1]["response"]["result"]
                if b["instrument_name"] == instrument["instrument_name"])
    raw["requests"][0]["response"]["result"] = [instrument]
    raw["requests"][1]["response"]["result"] = [book]
    return raw, instrument, book


@pytest.mark.parametrize("flag", ["call", "put"])
def test_usd_conversion_recovers_known_iv_with_own_solver(flag):
    raw, instrument, book = _one_contract()
    spot, strike, sigma, rate = 100000, 100000, .45, .03
    raw["requests"][2]["response"]["result"]["index_price"] = spot
    instrument["strike"] = strike
    instrument["option_type"] = flag
    instrument["expiration_timestamp"] = datetime(2027, 3, 26, 8, tzinfo=timezone.utc).timestamp() * 1000
    captured = datetime.fromisoformat(raw["captured_at_utc"].replace("Z", "+00:00"))
    tau = compute_tau(datetime(2027, 3, 26, 8, tzinfo=timezone.utc), captured)
    price = bs_price(spot, strike, tau, rate, sigma, flag)
    book.update(bid_price=price / spot * .99, ask_price=price / spot * 1.01,
                mark_price=999, mark_iv=99, underlying_price=spot * 1.2)
    chain = parse_deribit_snapshot(raw)
    assert chain.iloc[0].mid_price == pytest.approx(price)
    assert chain.iloc[0].mark_iv == .99
    result = build_combined_surface_from_chain(chain, r=rate)
    assert result.spot_price == spot
    assert result.iv_surface[0, 0] == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("bid,ask", [(None,.01), (.01,None), (0,.01), (.01,0), (.02,.01), (-.01,.01), ("bad",.01)])
def test_invalid_quotes_stay_missing_despite_mark_and_exchange_mid(bid, ask):
    raw, _, book = _one_contract()
    book.update(bid_price=bid, ask_price=ask, mark_price=.01, mid_price=.01)
    chain = parse_deribit_snapshot(raw)
    assert np.isnan(chain.iloc[0].mid_price)
    assert chain.iloc[0].quote_source == "none"
    assert np.isfinite(chain.iloc[0].mark_price)


def test_absent_book_retains_instrument_as_missing():
    raw, _, _ = _one_contract()
    raw["requests"][1]["response"]["result"] = []
    chain = parse_deribit_snapshot(raw)
    assert len(chain) == 1
    assert np.isnan(chain.iloc[0].mid_price)
    assert chain.iloc[0].quote_source == "none"


@pytest.mark.parametrize("case", ["schema", "time", "rpc", "index", "currency", "contract", "duplicate", "identity", "join", "expiry", "flag", "book_currency"])
def test_rejects_invalid_snapshot_and_ambiguous_contracts(case):
    raw, instrument, book = _one_contract()
    if case == "schema": raw["schema_version"] = -1
    elif case == "time": raw["captured_at_utc"] = "2020-01-01T00:00:00Z"
    elif case == "rpc": raw["requests"][1]["response"]["error"] = {"code": 1}
    elif case == "index": raw["requests"][2]["response"]["result"]["index_price"] = 0
    elif case == "currency": instrument["quote_currency"] = "USDC"
    elif case == "contract": instrument["contract_size"] = 2
    elif case == "duplicate": raw["requests"][0]["response"]["result"].append(instrument)
    elif case == "identity":
        other = deepcopy(instrument); other["instrument_name"] += "-OTHER"
        raw["requests"][0]["response"]["result"].append(other)
    elif case == "join": book["instrument_name"] = "unknown"
    elif case == "expiry": instrument["expiration_timestamp"] = None
    elif case == "flag": instrument["option_type"] = "other"
    elif case == "book_currency": book["quote_currency"] = "USD"
    with pytest.raises(ValueError): parse_deribit_snapshot(raw)
