"""Run the frozen V3 paper trader using OKX public BTC-USDT 4H candles.

This adapter exists because Binance market-data requests from GitHub-hosted runners
can return HTTP 451. It does NOT place any orders and does not require API keys.
"""
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

import paper_trader as pt

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_INST_ID = "BTC-USDT"
OKX_BAR = "4H"
TARGET_BARS = 600
PAGE_LIMIT = 100
FOUR_HOURS_MS = 4 * 60 * 60 * 1000


def _request_page(after=None):
    params = {
        "instId": OKX_INST_ID,
        "bar": OKX_BAR,
        "limit": PAGE_LIMIT,
    }
    if after is not None:
        params["after"] = str(after)
    req = Request(
        f"{OKX_URL}?{urlencode(params)}",
        headers={"User-Agent": "btc-v3-paper-trader/1.1"},
    )
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("code") != "0" or not isinstance(payload.get("data"), list):
        raise RuntimeError(f"Unexpected OKX response: {str(payload)[:500]}")
    return payload["data"]


def fetch_okx_klines():
    rows = []
    seen = set()
    after = None

    while len(rows) < TARGET_BARS:
        page = _request_page(after=after)
        if not page:
            break

        oldest_ts = None
        new_count = 0
        for item in page:
            # OKX schema: ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm
            if len(item) < 9:
                continue
            ts = int(item[0])
            oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
            if ts in seen:
                continue
            seen.add(ts)
            rows.append({
                "open_time": ts,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": ts + FOUR_HOURS_MS - 1,
                "confirm": str(item[8]),
            })
            new_count += 1

        if oldest_ts is None or new_count == 0:
            break
        after = oldest_ts
        time.sleep(0.15)

    if len(rows) < 250:
        raise RuntimeError(f"Not enough OKX 4H candles: got {len(rows)}")

    df = pd.DataFrame(rows).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


# Keep strategy logic frozen; only replace the market-data source.
pt.SYMBOL = OKX_INST_ID
pt.fetch_klines = fetch_okx_klines

if __name__ == "__main__":
    pt.main()
