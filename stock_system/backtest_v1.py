from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from urllib.request import Request, urlopen
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_v1 import DEFAULTS, add_indicators

INITIAL_CAPITAL_USD = 1500.0  # roughly the starting scale of ~5,000 ILS; research account is USD-based
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
SYMBOLS = ["SPY", "QQQ"]


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    stop: float
    risk_per_share: float
    entry_fee: float
    highest_close: float
    entry_date: pd.Timestamp


def fetch_stooq(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        txt = r.read().decode("utf-8")
    df = pd.read_csv(StringIO(txt))
    if df.empty or "Date" not in df:
        raise RuntimeError(f"No data returned for {symbol}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        raise RuntimeError(f"Bad OHLCV schema for {symbol}")
    return df[need].dropna()


def run_backtest(symbols: list[str] | None = None, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    symbols = symbols or SYMBOLS
    data = {s: add_indicators(fetch_stooq(s), p) for s in symbols}
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    cash = INITIAL_CAPITAL_USD
    positions: dict[str, Position] = {}
    trades = []
    equity_curve = []
    peak_equity = INITIAL_CAPITAL_USD
    killed = False

    def mark_equity(date: pd.Timestamp) -> float:
        eq = cash
        for s, pos in positions.items():
            df = data[s]
            if date in df.index:
                eq += pos.qty * float(df.loc[date, "Close"])
            else:
                prior = df.loc[:date]
                if not prior.empty:
                    eq += pos.qty * float(prior.iloc[-1]["Close"])
        return eq

    for i, date in enumerate(all_dates):
        # 1) Manage exits using today's bar. Stops are assumed to trigger intraday.
        for s in list(positions):
            if date not in data[s].index:
                continue
            row = data[s].loc[date]
            pos = positions[s]
            exit_reason = None
            exit_price = None

            # trailing stop only uses information from previously completed closes
            trail_candidate = pos.highest_close - p["trail_atr"] * float(row["atr"]) if pd.notna(row["atr"]) else pos.stop
            effective_stop = max(pos.stop, trail_candidate)

            if float(row["Low"]) <= effective_stop:
                exit_reason = "STOP"
                exit_price = effective_stop * (1 - SLIPPAGE_RATE)
            elif float(row["Close"]) < float(row["ema_fast"]):
                exit_reason = "EMA50_EXIT"
                exit_price = float(row["Close"]) * (1 - SLIPPAGE_RATE)

            if exit_reason:
                gross = pos.qty * exit_price
                exit_fee = gross * FEE_RATE
                cash += gross - exit_fee
                pnl = (exit_price - pos.entry_price) * pos.qty - pos.entry_fee - exit_fee
                initial_risk = pos.risk_per_share * pos.qty
                trades.append({
                    "symbol": s,
                    "entry_date": pos.entry_date.isoformat(),
                    "exit_date": date.isoformat(),
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "r_multiple": pnl / initial_risk if initial_risk > 0 else 0.0,
                    "reason": exit_reason,
                })
                del positions[s]
            else:
                pos.highest_close = max(pos.highest_close, float(row["Close"]))

        eq = mark_equity(date)
        peak_equity = max(peak_equity, eq)
        drawdown = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
        if drawdown >= p["kill_switch_drawdown"]:
            killed = True

        # 2) Entries: signal on today's close, fill on next available day's open.
        if not killed and len(positions) < p["max_positions"]:
            candidates = []
            for s, df in data.items():
                if s in positions or date not in df.index:
                    continue
                row = df.loc[date]
                if bool(row.get("entry_signal", False)) and pd.notna(row["atr"]):
                    candidates.append((s, float(row["atr"]), float(row["Close"])))

            for s, atr, signal_close in candidates:
                if len(positions) >= p["max_positions"]:
                    break
                df = data[s]
                future = df.loc[df.index > date]
                if future.empty:
                    continue
                next_date = future.index[0]
                if next_date not in all_dates:
                    continue
                entry = float(future.iloc[0]["Open"]) * (1 + SLIPPAGE_RATE)
                stop_distance = p["stop_atr"] * atr
                if stop_distance <= 0:
                    continue
                risk_budget = eq * p["risk_pct"]
                qty_by_risk = risk_budget / stop_distance  # fractional shares supported in research
                qty_by_cash = cash / (entry * (1 + FEE_RATE))
                qty = max(0.0, min(qty_by_risk, qty_by_cash))
                if qty < 0.001:
                    continue
                cost = qty * entry
                entry_fee = cost * FEE_RATE
                cash -= cost + entry_fee
                positions[s] = Position(
                    symbol=s,
                    qty=qty,
                    entry_price=entry,
                    stop=entry - stop_distance,
                    risk_per_share=stop_distance,
                    entry_fee=entry_fee,
                    highest_close=entry,
                    entry_date=next_date,
                )

        eq = mark_equity(date)
        peak_equity = max(peak_equity, eq)
        dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
        equity_curve.append({"date": date.isoformat(), "equity": eq, "drawdown": dd})

    # Liquidate remaining positions at final close for research accounting.
    final_date = all_dates[-1]
    for s in list(positions):
        df = data[s]
        last = df.loc[:final_date].iloc[-1]
        pos = positions[s]
        exit_price = float(last["Close"]) * (1 - SLIPPAGE_RATE)
        gross = pos.qty * exit_price
        exit_fee = gross * FEE_RATE
        cash += gross - exit_fee
        pnl = (exit_price - pos.entry_price) * pos.qty - pos.entry_fee - exit_fee
        initial_risk = pos.risk_per_share * pos.qty
        trades.append({
            "symbol": s, "entry_date": pos.entry_date.isoformat(), "exit_date": final_date.isoformat(),
            "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": exit_price,
            "pnl": pnl, "r_multiple": pnl / initial_risk if initial_risk > 0 else 0.0,
            "reason": "END_OF_TEST",
        })
        del positions[s]

    final_equity = cash
    returns = final_equity / INITIAL_CAPITAL_USD - 1
    start = pd.Timestamp(all_dates[0])
    end = pd.Timestamp(all_dates[-1])
    years = max((end - start).days / 365.25, 1 / 365.25)
    cagr = (final_equity / INITIAL_CAPITAL_USD) ** (1 / years) - 1
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    max_dd = max((x["drawdown"] for x in equity_curve), default=0.0)

    return {
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "final_equity_usd": final_equity,
        "return_pct": returns * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "profit_factor": pf,
        "trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
        "avg_r": float(np.mean([t["r_multiple"] for t in trades])) if trades else 0.0,
        "kill_switch_triggered": killed,
        "params": p,
        "trade_log": trades,
        "equity_curve": equity_curve,
        "data_start": start.isoformat(),
        "data_end": end.isoformat(),
    }


def main() -> None:
    out = run_backtest()
    Path("stock_system/results").mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in out.items() if k not in ("trade_log", "equity_curve")}
    Path("stock_system/results/v1_summary.json").write_text(json.dumps(slim, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(out["trade_log"]).to_csv("stock_system/results/v1_trades.csv", index=False)
    pd.DataFrame(out["equity_curve"]).to_csv("stock_system/results/v1_equity.csv", index=False)
    print(json.dumps(slim, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
