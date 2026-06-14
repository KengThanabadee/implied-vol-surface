from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

_BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"
_EXPIRY_FMT = "%d%b%y"


def _to_float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_symbol(symbol: str) -> dict:
    """Parse a Bybit option symbol like BTC-27JUN25-100000-C into components."""
    parts = symbol.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected symbol format: {symbol!r}")
    underlying, expiry_str, strike_str, flag_char = parts
    expiry_dt = datetime.strptime(expiry_str, _EXPIRY_FMT).replace(
        hour=8, tzinfo=timezone.utc
    )
    strike = float(strike_str)
    if flag_char.upper() == "C":
        flag = "call"
    elif flag_char.upper() == "P":
        flag = "put"
    else:
        raise ValueError(f"Unexpected option flag: {flag_char!r}")
    return {"underlying": underlying, "expiry_dt": expiry_dt, "strike": strike, "flag": flag}


def compute_tau(expiry_dt: datetime, now: datetime) -> float:
    """Time to expiry in years."""
    diff = expiry_dt - now
    seconds = diff.total_seconds()
    return max(seconds / (365.25 * 24 * 3600), 0.0)


def fetch_chain(underlying: str = "BTC") -> pd.DataFrame:
    """Fetch live option chain from Bybit and return a tidy DataFrame."""
    params = {"category": "option", "baseCoin": underlying}
    resp = requests.get(_BYBIT_TICKERS_URL, params=params, timeout=10)
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"JSON decode failed: {exc}") from exc

    result = data.get("result", {})
    tickers = result.get("list") if isinstance(result, dict) else None
    if not isinstance(tickers, list):
        raise RuntimeError(f"Unexpected response structure: {data}")

    now = datetime.now(timezone.utc)
    rows = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            continue

        mark_price = _to_float(ticker.get("markPrice"))
        underlying_price = _to_float(ticker.get("underlyingPrice"))
        tau = compute_tau(parsed["expiry_dt"], now)

        rows.append(
            {
                "symbol": symbol,
                "underlying": parsed["underlying"],
                "expiry_dt": parsed["expiry_dt"],
                "strike": parsed["strike"],
                "flag": parsed["flag"],
                "mark_price": mark_price,
                "underlying_price": underlying_price,
                "tau": tau,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["expiry_dt", "strike"]).reset_index(drop=True)
    return df
