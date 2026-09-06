from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

import backtest_v1
import strategy_v2

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]


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


# Reuse the already-tested portfolio/risk engine, but swap in V2 signals and Yahoo transport.
backtest_v1.fetch_stooq = fetch_yahoo
backtest_v1.add_indicators = strategy_v2.add_indicators
backtest_v1.DEFAULTS = strategy_v2.DEFAULTS


def main() -> None:
    out = backtest_v1.run_backtest(symbols=SYMBOLS, params=strategy_v2.DEFAULTS)
    Path("stock_system/results").mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in out.items() if k not in ("trade_log", "equity_curve")}
    slim["strategy"] = "V2 trend pullback"
    slim["symbols"] = SYMBOLS
    Path("stock_system/results/v2_summary.json").write_text(json.dumps(slim, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(out["trade_log"]).to_csv("stock_system/results/v2_trades.csv", index=False)
    pd.DataFrame(out["equity_curve"]).to_csv("stock_system/results/v2_equity.csv", index=False)
    print(json.dumps(slim, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
