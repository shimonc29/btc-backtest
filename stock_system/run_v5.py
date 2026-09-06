from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from strategy_v5 import DEFAULTS, add_features, target_weights

INITIAL_CAPITAL_USD = 1500.0
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLI", "XLY"]


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=max&interval=1d&includeAdjustedClose=true&events=history"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    result = result[0]
    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
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
    spy_features = features["SPY"]

    common_start = max(df.index.min() for df in raw.values())
    start_date = max(common_start, pd.Timestamp("2001-01-01"))
    end_date = min(df.index.max() for df in raw.values())
    dates = raw["SPY"].loc[(raw["SPY"].index >= start_date) & (raw["SPY"].index <= end_date)].index
    if len(dates) < 300:
        raise RuntimeError("Insufficient common data for V5")

    cash = INITIAL_CAPITAL_USD
    shares = {s: 0.0 for s in UNIVERSE}
    equity_curve = []
    rebalance_log = []
    regime_counts: dict[str, int] = {}
    stress_count = 0
    turnover_events = 0
    pending_weights: dict[str, float] | None = None
    pending_meta: dict | None = None
    peak_equity = INITIAL_CAPITAL_USD

    def equity_at(date: pd.Timestamp) -> float:
        eq = cash
        for s in UNIVERSE:
            hist = raw[s].loc[:date]
            if not hist.empty:
                eq += shares[s] * float(hist.iloc[-1]["Close"])
        return eq

    for i, date in enumerate(dates):
        if pending_weights is not None:
            eq_before = equity_at(date)
            target_dollars = {s: eq_before * pending_weights.get(s, 0.0) for s in UNIVERSE}

            # Sell first, then buy. Fractional shares are allowed in research.
            for s in UNIVERSE:
                if date not in raw[s].index:
                    continue
                open_px = float(raw[s].loc[date, "Open"])
                current = shares[s] * open_px
                target = target_dollars[s]
                if current > target + 1e-9:
                    qty = min(shares[s], (current - target) / open_px)
                    if qty > 1e-9:
                        fill = open_px * (1 - p["slippage_rate"])
                        gross = qty * fill
                        fee = gross * p["fee_rate"]
                        cash += gross - fee
                        shares[s] -= qty
                        turnover_events += 1

            for s in UNIVERSE:
                if date not in raw[s].index:
                    continue
                open_px = float(raw[s].loc[date, "Open"])
                current = shares[s] * open_px
                desired = max(0.0, target_dollars[s] - current)
                if desired <= 1e-9 or cash <= 0:
                    continue
                fill = open_px * (1 + p["slippage_rate"])
                spend = min(desired, cash / (1 + p["fee_rate"]))
                qty = spend / fill
                if qty > 1e-9:
                    cost = qty * fill
                    fee = cost * p["fee_rate"]
                    cash -= cost + fee
                    shares[s] += qty
                    turnover_events += 1

            meta = pending_meta or {}
            rebalance_log.append({
                "signal_date": meta.get("signal_date"),
                "execution_date": date.isoformat(),
                "regime": meta.get("regime"),
                "breadth": meta.get("breadth"),
                "stress": meta.get("stress"),
                "drawdown_brake": meta.get("drawdown_brake"),
                "targets": pending_weights,
                "equity_before": eq_before,
                "equity_after": equity_at(date),
            })
            regime = meta.get("regime") or "UNKNOWN"
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if meta.get("stress"):
                stress_count += 1
            pending_weights = None
            pending_meta = None

        eq = equity_at(date)
        peak_equity = max(peak_equity, eq)
        current_dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
        equity_curve.append({"date": date.isoformat(), "equity": eq, "drawdown": current_dd})

        next_date = dates[i + 1] if i + 1 < len(dates) else None
        is_month_end = next_date is None or (next_date.year, next_date.month) != (date.year, date.month)
        if is_month_end and next_date is not None:
            weights, regime, br, stress = target_weights(features, spy_features, date, p)
            brake = "NONE"
            if current_dd >= p["dd_brake_2"]:
                weights = {}
                brake = "CASH"
            elif current_dd >= p["dd_brake_1"]:
                weights = {s: w * p["dd_brake_1_multiplier"] for s, w in weights.items()}
                brake = "HALF"
            pending_weights = weights
            pending_meta = {
                "signal_date": date.isoformat(),
                "regime": regime,
                "breadth": br,
                "stress": stress,
                "drawdown_brake": brake,
            }

    final_equity = equity_at(dates[-1])
    curve = pd.DataFrame(equity_curve)
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date")
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    daily_ret = curve["equity"].pct_change().dropna()

    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    total_return = final_equity / INITIAL_CAPITAL_USD - 1.0
    cagr = (final_equity / INITIAL_CAPITAL_USD) ** (1 / years) - 1.0
    max_dd = abs(float(curve["drawdown"].min()))
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    monthly = curve["equity"].resample("ME").last().pct_change().dropna()
    positive_month_pct = float((monthly > 0).mean() * 100) if len(monthly) else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    spy = raw["SPY"]
    spy_start = float(spy.loc[dates[0], "Close"])
    spy_end = float(spy.loc[dates[-1], "Close"])
    spy_return = spy_end / spy_start - 1.0
    spy_cagr = (spy_end / spy_start) ** (1 / years) - 1.0
    spy_window = spy.loc[dates[0]:dates[-1], "Close"].copy()
    spy_curve = spy_window / spy_start
    spy_dd = abs(float((spy_curve / spy_curve.cummax() - 1.0).min()))

    summary = {
        "strategy": "V5 participation-first adaptive ensemble",
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "final_equity_usd": final_equity,
        "return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "calmar": calmar,
        "positive_month_pct": positive_month_pct,
        "rebalances": len(rebalance_log),
        "turnover_events": turnover_events,
        "regime_counts": regime_counts,
        "stress_rebalances": stress_count,
        "data_start": dates[0].isoformat(),
        "data_end": dates[-1].isoformat(),
        "symbols": UNIVERSE,
        "params": p,
        "benchmark_spy_return_pct": spy_return * 100,
        "benchmark_spy_cagr_pct": spy_cagr * 100,
        "benchmark_spy_max_drawdown_pct": spy_dd * 100,
    }
    return {"summary": summary, "equity": curve.reset_index(), "rebalances": rebalance_log}


def main() -> None:
    result = run_backtest()
    outdir = Path("stock_system/results")
    outdir.mkdir(parents=True, exist_ok=True)
    outdir.joinpath("v5_summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    result["equity"].to_csv(outdir / "v5_equity.csv", index=False)
    pd.DataFrame(result["rebalances"]).to_csv(outdir / "v5_rebalances.csv", index=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
