from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from strategy_v3 import DEFAULTS, add_features, target_weights

INITIAL_CAPITAL_USD = 1500.0
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLI", "XLY"]


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


def run_backtest(params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    raw = {s: fetch_yahoo(s) for s in UNIVERSE}
    features = {s: add_features(df, p) for s, df in raw.items()}
    spy = raw["SPY"]

    # Start only once every symbol has enough history for a fair cross-sectional comparison.
    common_start = max(df.index.min() for df in raw.values())
    start_date = max(common_start, pd.Timestamp("2001-01-01"))
    end_date = min(df.index.max() for df in raw.values())
    dates = spy.loc[(spy.index >= start_date) & (spy.index <= end_date)].index
    if len(dates) < 300:
        raise RuntimeError("Insufficient common data for V3")

    cash = INITIAL_CAPITAL_USD
    shares = {s: 0.0 for s in UNIVERSE}
    last_close = {s: float(raw[s].loc[:dates[0]].iloc[-1]["Close"]) for s in UNIVERSE}
    equity_curve = []
    rebalance_log = []
    turnover_events = 0

    def equity_at(date: pd.Timestamp) -> float:
        eq = cash
        for s in UNIVERSE:
            hist = raw[s].loc[:date]
            if not hist.empty:
                px = float(hist.iloc[-1]["Close"])
                last_close[s] = px
            else:
                px = last_close[s]
            eq += shares[s] * px
        return eq

    previous_month = None
    pending_weights: dict[str, float] | None = None
    pending_signal_date: pd.Timestamp | None = None

    for i, date in enumerate(dates):
        month = (date.year, date.month)

        # Execute pending rebalance at today's open, so the signal only uses prior completed closes.
        if pending_weights is not None:
            eq_before = equity_at(date)
            target_dollars = {s: eq_before * pending_weights.get(s, 0.0) for s in UNIVERSE}

            # Sell first.
            for s in UNIVERSE:
                if date not in raw[s].index:
                    continue
                open_px = float(raw[s].loc[date, "Open"])
                current_value = shares[s] * open_px
                target_value = target_dollars[s]
                if current_value > target_value + 1e-9:
                    sell_value = current_value - target_value
                    qty = min(shares[s], sell_value / open_px)
                    if qty > 1e-9:
                        fill = open_px * (1 - p["slippage_rate"])
                        gross = qty * fill
                        fee = gross * p["fee_rate"]
                        cash += gross - fee
                        shares[s] -= qty
                        turnover_events += 1

            # Then buy using available cash.
            for s in UNIVERSE:
                if date not in raw[s].index:
                    continue
                open_px = float(raw[s].loc[date, "Open"])
                current_value = shares[s] * open_px
                desired = max(0.0, target_dollars[s] - current_value)
                if desired <= 1e-9 or cash <= 0:
                    continue
                fill = open_px * (1 + p["slippage_rate"])
                max_spend = cash / (1 + p["fee_rate"])
                spend = min(desired, max_spend)
                qty = spend / fill
                if qty > 1e-9:
                    cost = qty * fill
                    fee = cost * p["fee_rate"]
                    cash -= cost + fee
                    shares[s] += qty
                    turnover_events += 1

            eq_after = equity_at(date)
            rebalance_log.append({
                "signal_date": pending_signal_date.isoformat() if pending_signal_date is not None else None,
                "execution_date": date.isoformat(),
                "targets": pending_weights,
                "equity_before": eq_before,
                "equity_after": eq_after,
            })
            pending_weights = None
            pending_signal_date = None

        eq = equity_at(date)
        equity_curve.append({"date": date.isoformat(), "equity": eq})

        # Month-end signal: if the next trading day is a new month, compute targets from today's close.
        next_date = dates[i + 1] if i + 1 < len(dates) else None
        is_month_end = next_date is None or (next_date.year, next_date.month) != month
        if is_month_end and next_date is not None:
            pending_weights = target_weights(features, spy, date, p)
            pending_signal_date = date

        previous_month = month

    final_equity = equity_at(dates[-1])
    curve = pd.DataFrame(equity_curve)
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date")
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    daily_ret = curve["equity"].pct_change().dropna()

    total_return = final_equity / INITIAL_CAPITAL_USD - 1.0
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    cagr = (final_equity / INITIAL_CAPITAL_USD) ** (1 / years) - 1.0
    max_dd = abs(float(curve["drawdown"].min())) if not curve.empty else 0.0
    sharpe = 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    # Monthly win rate from portfolio equity, not individual fills.
    monthly = curve["equity"].resample("ME").last().pct_change().dropna()
    positive_month_pct = float((monthly > 0).mean() * 100) if len(monthly) else 0.0

    # Benchmark SPY buy-and-hold over same test window.
    spy_start = float(spy.loc[dates[0], "Close"])
    spy_end = float(spy.loc[dates[-1], "Close"])
    spy_return = spy_end / spy_start - 1.0
    spy_cagr = (spy_end / spy_start) ** (1 / years) - 1.0

    summary = {
        "strategy": "V3 adaptive relative-strength rotation",
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "final_equity_usd": final_equity,
        "return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "positive_month_pct": positive_month_pct,
        "rebalances": len(rebalance_log),
        "turnover_events": turnover_events,
        "data_start": dates[0].isoformat(),
        "data_end": dates[-1].isoformat(),
        "symbols": UNIVERSE,
        "params": p,
        "benchmark_spy_return_pct": spy_return * 100,
        "benchmark_spy_cagr_pct": spy_cagr * 100,
    }
    return {"summary": summary, "equity": curve.reset_index(), "rebalances": rebalance_log}


def main() -> None:
    result = run_backtest()
    outdir = Path("stock_system/results")
    outdir.mkdir(parents=True, exist_ok=True)
    outdir.joinpath("v3_summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    result["equity"].to_csv(outdir / "v3_equity.csv", index=False)
    pd.DataFrame(result["rebalances"]).to_csv(outdir / "v3_rebalances.csv", index=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
