"""Adapt frozen Deribit BTC inverse-option snapshots to the USD chain model."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iv_surface.fetcher import (
    _compute_mid_price,
    _parse_utc_timestamp,
    _to_float,
    compute_tau,
)


_REQUESTS = {
    "/api/v2/public/get_instruments": {
        "currency": "BTC", "kind": "option", "expired": "false",
    },
    "/api/v2/public/get_book_summary_by_currency": {
        "currency": "BTC", "kind": "option",
    },
    "/api/v2/public/get_index_price": {"index_name": "btc_usd"},
}


def _snapshot_results(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("Deribit snapshot must be an object")
    if (snapshot.get("schema_version") != 1
            or snapshot.get("exchange") != "deribit"
            or snapshot.get("currency") != "BTC"):
        raise ValueError("Expected a version 1 Deribit BTC snapshot")
    started = _parse_utc_timestamp(snapshot.get("capture_started_at_utc"))
    captured = _parse_utc_timestamp(snapshot.get("captured_at_utc"))
    if started > captured:
        raise ValueError("Snapshot capture times are out of order")
    if not isinstance(snapshot.get("base_url"), str) or not snapshot["base_url"]:
        raise ValueError("Snapshot needs its source base_url")
    requests = snapshot.get("requests")
    if not isinstance(requests, list) or len(requests) != len(_REQUESTS):
        raise ValueError("Snapshot needs instrument, book and index responses")
    results = {}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Snapshot request must be an object")
        endpoint = request.get("endpoint")
        if not isinstance(endpoint, str) or endpoint not in _REQUESTS or endpoint in results:
            raise ValueError("Unexpected or duplicate Deribit snapshot endpoint")
        if request.get("params") != _REQUESTS[endpoint]:
            raise ValueError("Unexpected Deribit request parameters")
        received = _parse_utc_timestamp(request.get("received_at_utc"))
        if not started <= received <= captured:
            raise ValueError("Response timestamp is outside the capture interval")
        response = request.get("response")
        if not isinstance(response, dict) or "error" in response or "result" not in response:
            raise ValueError("Invalid Deribit JSON-RPC response")
        results[endpoint] = response["result"]
    return captured, results


def _by_name(items, label):
    if not isinstance(items, list):
        raise ValueError(f"Deribit {label} result must be a list")
    named = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Deribit {label} entry must be an object")
        name = item.get("instrument_name")
        if not isinstance(name, str) or not name or name in named:
            raise ValueError(f"Missing or duplicate instrument in Deribit {label}")
        named[name] = item
    return named


def parse_deribit_snapshot(snapshot: dict) -> pd.DataFrame:
    """Return USD premiums and spot, retaining raw BTC quotes and frozen maturity.

    The existing spot Black-Scholes model is used downstream. Deribit's
    expiry-specific underlying_price and mark_iv remain comparison fields.
    """
    captured, results = _snapshot_results(snapshot)
    instruments = _by_name(results["/api/v2/public/get_instruments"], "instruments")
    books = _by_name(results["/api/v2/public/get_book_summary_by_currency"], "books")
    if not instruments:
        raise ValueError("Deribit snapshot has no option instruments")
    if books.keys() - instruments.keys():
        raise ValueError("Deribit book has instruments absent from metadata")
    index = results["/api/v2/public/get_index_price"]
    if not isinstance(index, dict):
        raise ValueError("Deribit index result must be an object")
    spot = _to_float(index.get("index_price"))
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Deribit BTC/USD index must be finite and positive")

    rows = []
    for name, instrument in instruments.items():
        required = {
            "kind": "option", "base_currency": "BTC", "quote_currency": "BTC",
            "settlement_currency": "BTC", "counter_currency": "USD",
            "instrument_type": "reversed", "price_index": "btc_usd",
            "contract_size": 1,
        }
        if any(instrument.get(key) != value for key, value in required.items()):
            raise ValueError(f"Unsupported Deribit contract: {name}")
        flag = instrument.get("option_type")
        strike = _to_float(instrument.get("strike"))
        expiry_ms = _to_float(instrument.get("expiration_timestamp"))
        if flag not in {"call", "put"} or not np.isfinite(strike) or strike <= 0:
            raise ValueError(f"Invalid Deribit option identity: {name}")
        if not np.isfinite(expiry_ms) or expiry_ms <= 0:
            raise ValueError(f"Invalid Deribit expiration timestamp: {name}")
        try:
            expiry = datetime.fromtimestamp(expiry_ms / 1000, timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"Invalid Deribit expiration timestamp: {name}") from exc
        book = books.get(name, {})
        if book and (book.get("base_currency") != "BTC" or book.get("quote_currency") != "BTC"):
            raise ValueError(f"Unexpected Deribit book currency: {name}")
        bid = _to_float(book.get("bid_price"))
        ask = _to_float(book.get("ask_price"))
        mid = _compute_mid_price(bid, ask)
        mark = _to_float(book.get("mark_price"))
        rows.append({
            "symbol": name, "underlying": "BTC", "flag": flag, "strike": strike,
            "expiry_dt": expiry, "tau": compute_tau(expiry, captured),
            "bid_price": bid * spot, "ask_price": ask * spot,
            "mid_price": mid * spot,
            "quote_source": "mid" if np.isfinite(mid) else "none",
            "bid_price_btc": bid, "ask_price_btc": ask, "mid_price_btc": mid,
            "mark_price_btc": mark, "mark_price": mark * spot,
            "bid_iv": np.nan, "ask_iv": np.nan,
            "mark_iv": _to_float(book.get("mark_iv")) / 100,
            "index_price": spot,
            "underlying_price": _to_float(book.get("underlying_price")),
        })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["flag", "tau", "strike"]).any():
        raise ValueError("Duplicate Deribit flag/tau/strike identity")
    return frame.sort_values(["tau", "strike", "flag"]).reset_index(drop=True)


def load_deribit_snapshot(path) -> dict:
    """Read and validate a Deribit snapshot without a network request."""
    with Path(path).open(encoding="utf-8") as source:
        snapshot = json.load(source)
    _snapshot_results(snapshot)
    return snapshot
