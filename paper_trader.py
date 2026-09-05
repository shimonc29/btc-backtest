import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "4h"
API_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 500

INITIAL_EQUITY = 2000.0
RISK_PCT = 0.005
BREAKOUT_LOOKBACK = 20
INITIAL_STOP_ATR = 2.0
TRAIL_ATR = 3.0
TIME_STOP_BARS = 30
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0002

PAPER_DIR = Path("paper")
STATE_PATH = PAPER_DIR / "state.json"
STATUS_PATH = PAPER_DIR / "status.json"
EVENTS_PATH = PAPER_DIR / "events.csv"
TRADES_PATH = PAPER_DIR / "trades.csv"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def fetch_klines():
    query = urlencode({"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT})
    req = Request(f"{API_URL}?{query}", headers={"User-Agent": "btc-v3-paper-trader/1.0"})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if not isinstance(payload, list) or len(payload) < 250:
        raise RuntimeError(f"Unexpected Binance kline response: {str(payload)[:300]}")

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(payload, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("Int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df.reset_index(drop=True)


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def prepare(closed):
    df = closed.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"] = atr(df, 14)
    df["atr_pct"] = df["atr14"] / df["close"]
    df["ema200_prev12"] = df["ema200"].shift(12)
    df["breakout"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["atr_q40"] = df["atr_pct"].rolling(100).quantile(0.40)
    df["atr_q90"] = df["atr_pct"].rolling(100).quantile(0.90)
    return df


def default_state():
    return {
        "version": 1,
        "strategy": "BTC Momentum Breakout 4H V3",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "initial_equity": INITIAL_EQUITY,
        "equity": INITIAL_EQUITY,
        "position": None,
        "last_processed_open_time": None,
        "closed_trades": 0,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def load_state():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return default_state(), True
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f), False


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def append_csv(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def ts_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def signal_flags(row):
    needed = [row["atr14"], row["breakout"], row["atr_q40"], row["atr_q90"], row["ema200_prev12"]]
    valid = all(np.isfinite(x) for x in needed)
    if not valid:
        return {"valid": False, "trend_ok": False, "breakout_ok": False, "volatility_ok": False, "entry_signal": False}
    trend_ok = bool(
        row["close"] > row["ema200"]
        and row["ema50"] > row["ema200"]
        and row["ema200"] > row["ema200_prev12"]
    )
    breakout_ok = bool(row["close"] > row["breakout"])
    volatility_ok = bool(row["atr_q40"] <= row["atr_pct"] <= row["atr_q90"])
    return {
        "valid": True,
        "trend_ok": trend_ok,
        "breakout_ok": breakout_ok,
        "volatility_ok": volatility_ok,
        "entry_signal": bool(trend_ok and breakout_ok and volatility_ok),
    }


def record_event(kind, bar, price=None, note="", equity=None):
    append_csv(
        EVENTS_PATH,
        {
            "event_time_utc": utc_now_iso(),
            "bar_open_utc": ts_iso(int(bar["open_time"])),
            "type": kind,
            "price": "" if price is None else f"{price:.8f}",
            "equity": "" if equity is None else f"{equity:.8f}",
            "note": note,
        },
        ["event_time_utc", "bar_open_utc", "type", "price", "equity", "note"],
    )


def close_position(state, pos, bar, exit_price, outcome):
    exit_notional = pos["qty"] * exit_price
    exit_fee = exit_notional * FEE_RATE
    gross_pnl = pos["qty"] * (exit_price - pos["entry_price"])
    net_pnl = gross_pnl - pos["entry_fee"] - exit_fee
    r_multiple = net_pnl / pos["risk_dollars"] if pos["risk_dollars"] > 0 else 0.0
    state["equity"] += net_pnl
    state["closed_trades"] = int(state.get("closed_trades", 0)) + 1

    trade = {
        "entry_time_utc": ts_iso(pos["entry_bar_open_time"]),
        "exit_time_utc": ts_iso(int(bar["open_time"])),
        "entry_price": f"{pos['entry_price']:.8f}",
        "exit_price": f"{exit_price:.8f}",
        "qty": f"{pos['qty']:.10f}",
        "outcome": outcome,
        "bars_held": pos["bars_held"],
        "gross_pnl": f"{gross_pnl:.8f}",
        "fees": f"{(pos['entry_fee'] + exit_fee):.8f}",
        "net_pnl": f"{net_pnl:.8f}",
        "r_multiple": f"{r_multiple:.8f}",
        "equity_after": f"{state['equity']:.8f}",
    }
    append_csv(
        TRADES_PATH,
        trade,
        ["entry_time_utc", "exit_time_utc", "entry_price", "exit_price", "qty", "outcome", "bars_held", "gross_pnl", "fees", "net_pnl", "r_multiple", "equity_after"],
    )
    record_event("EXIT", bar, exit_price, outcome, state["equity"])
    state["position"] = None


def open_position(state, signal_row, next_bar):
    entry = float(next_bar["open"]) * (1 + SLIPPAGE_RATE)
    stop_distance = INITIAL_STOP_ATR * float(signal_row["atr14"])
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        return False

    risk_dollars = float(state["equity"]) * RISK_PCT
    qty = risk_dollars / stop_distance
    notional = qty * entry
    if notional > state["equity"]:
        qty = state["equity"] / entry
        notional = qty * entry
        risk_dollars = qty * stop_distance

    initial_stop = entry - stop_distance
    entry_fee = notional * FEE_RATE
    state["position"] = {
        "entry_signal_bar_open_time": int(signal_row["open_time"]),
        "entry_bar_open_time": int(next_bar["open_time"]),
        "entry_price": entry,
        "qty": qty,
        "notional": notional,
        "risk_dollars": risk_dollars,
        "entry_fee": entry_fee,
        "initial_stop": initial_stop,
        "trail_stop": initial_stop,
        "highest_close": entry,
        "bars_held": 0,
    }
    record_event("ENTRY", next_bar, entry, f"signal_bar={ts_iso(int(signal_row['open_time']))}", state["equity"])
    return True


def main():
    all_bars = fetch_klines()
    now_ms = int(time.time() * 1000)
    closed = all_bars[all_bars["close_time"] < now_ms].copy().reset_index(drop=True)
    if len(closed) < 250:
        raise RuntimeError("Not enough closed 4H candles from Binance")
    prepared = prepare(closed)

    state, is_new = load_state()
    if is_new:
        # Start live-forward from the latest closed candle, without retroactive trades.
        state["last_processed_open_time"] = int(prepared.iloc[-2]["open_time"])
        record_event("START", prepared.iloc[-1], None, "Paper trading initialized at $2,000", state["equity"])

    last_processed = state.get("last_processed_open_time")
    new_idx = prepared.index[prepared["open_time"] > int(last_processed)].tolist() if last_processed is not None else [prepared.index[-1]]

    processed_count = 0
    last_flags = None
    last_row = prepared.iloc[-1]

    for idx in new_idx:
        row = prepared.loc[idx]
        processed_count += 1
        exited_this_bar = False
        pos = state.get("position")

        if pos is not None and int(row["open_time"]) >= int(pos["entry_bar_open_time"]):
            # Match V3 backtest: no trail update on the entry bar; later bars use previous closed bar only.
            if int(row["open_time"]) > int(pos["entry_bar_open_time"]) and idx > 0:
                prev = prepared.loc[idx - 1]
                pos["highest_close"] = max(float(pos["highest_close"]), float(prev["close"]))
                if np.isfinite(prev["atr14"]):
                    candidate = pos["highest_close"] - TRAIL_ATR * float(prev["atr14"])
                    pos["trail_stop"] = max(float(pos["trail_stop"]), candidate)

            pos["bars_held"] = int(pos.get("bars_held", 0)) + 1

            if float(row["low"]) <= float(pos["trail_stop"]):
                exit_price = float(pos["trail_stop"]) * (1 - SLIPPAGE_RATE)
                outcome = "TRAIL_STOP" if float(pos["trail_stop"]) > float(pos["initial_stop"]) else "INITIAL_STOP"
                close_position(state, pos, row, exit_price, outcome)
                exited_this_bar = True
            elif pos["bars_held"] >= 2 and float(row["close"]) < float(row["ema20"]):
                exit_price = float(row["close"]) * (1 - SLIPPAGE_RATE)
                close_position(state, pos, row, exit_price, "EMA20_EXIT")
                exited_this_bar = True
            elif pos["bars_held"] >= TIME_STOP_BARS:
                exit_price = float(row["close"]) * (1 - SLIPPAGE_RATE)
                close_position(state, pos, row, exit_price, "TIME")
                exited_this_bar = True
            else:
                state["position"] = pos

        last_flags = signal_flags(row)

        if state.get("position") is None and not exited_this_bar and last_flags["entry_signal"]:
            # Entry is filled at the next 4H bar open, exactly as in the backtest.
            full_matches = all_bars.index[all_bars["open_time"] == int(row["open_time"])].tolist()
            if full_matches:
                full_i = full_matches[0]
                if full_i + 1 < len(all_bars):
                    next_bar = all_bars.iloc[full_i + 1]
                    open_position(state, row, next_bar)

        state["last_processed_open_time"] = int(row["open_time"])
        last_row = row

    if last_flags is None:
        last_flags = signal_flags(last_row)

    state["updated_at"] = utc_now_iso()
    save_json(STATE_PATH, state)

    position = state.get("position")
    status = {
        "updated_at_utc": utc_now_iso(),
        "strategy": "BTC Momentum Breakout 4H V3 (FROZEN)",
        "mode": "PAPER_ONLY_NO_REAL_ORDERS",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "parameters": {
            "risk_pct": RISK_PCT,
            "breakout_lookback": BREAKOUT_LOOKBACK,
            "initial_stop_atr": INITIAL_STOP_ATR,
            "trail_atr": TRAIL_ATR,
            "time_stop_bars": TIME_STOP_BARS,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
        },
        "latest_closed_bar": {
            "open_time_utc": ts_iso(int(last_row["open_time"])),
            "close": float(last_row["close"]),
            "ema20": float(last_row["ema20"]),
            "ema50": float(last_row["ema50"]),
            "ema200": float(last_row["ema200"]),
            "atr14": float(last_row["atr14"]),
            "breakout_level": float(last_row["breakout"]),
            "atr_pct": float(last_row["atr_pct"]),
        },
        "signal": last_flags,
        "equity": float(state["equity"]),
        "return_pct": (float(state["equity"]) / float(state["initial_equity"]) - 1) * 100,
        "closed_trades": int(state.get("closed_trades", 0)),
        "position": position,
        "processed_new_closed_bars": processed_count,
    }
    save_json(STATUS_PATH, status)

    print(json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
