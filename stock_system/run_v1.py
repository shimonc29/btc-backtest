from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd

import backtest_v1


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=max&interval=1d&includeAdjustedClose=true&events=history"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {payload.get('chart', {}).get('error')}")
    result = result[0]
    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not ts:
        raise RuntimeError(f"Yahoo returned empty timestamps for {symbol}")
    df = pd.DataFrame({
        "Date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "Open": quote.get("open", []),
        "High": quote.get("high", []),
        "Low": quote.get("low", []),
        "Close": quote.get("close", []),
        "Volume": quote.get("volume", []),
    }).set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# Keep V1 strategy/backtest logic frozen; replace only the unavailable data transport.
backtest_v1.fetch_stooq = fetch_yahoo

if __name__ == "__main__":
    backtest_v1.main()
