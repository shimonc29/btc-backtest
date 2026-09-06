from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULTS = {
    "ema_fast": 50,
    "ema_slow": 200,
    "breakout_lookback": 20,
    "atr_period": 14,
    "stop_atr": 2.0,
    "trail_atr": 3.0,
    "risk_pct": 0.005,
    "max_positions": 3,
    "kill_switch_drawdown": 0.12,
}


def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    out = df.copy().sort_index()
    out["ema_fast"] = out["Close"].ewm(span=p["ema_fast"], adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=p["ema_slow"], adjust=False).mean()
    out["ema_slow_slope"] = out["ema_slow"] - out["ema_slow"].shift(20)

    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(p["atr_period"]).mean()
    out["breakout_level"] = out["High"].shift(1).rolling(p["breakout_lookback"]).max()

    out["trend_ok"] = (
        (out["Close"] > out["ema_slow"]) &
        (out["ema_fast"] > out["ema_slow"]) &
        (out["ema_slow_slope"] > 0)
    )
    out["breakout_ok"] = out["Close"] > out["breakout_level"]
    out["entry_signal"] = out["trend_ok"] & out["breakout_ok"] & out["atr"].notna()
    return out


def position_size(equity: float, entry: float, atr: float, params: dict | None = None) -> int:
    p = {**DEFAULTS, **(params or {})}
    stop_distance = p["stop_atr"] * atr
    if equity <= 0 or entry <= 0 or stop_distance <= 0:
        return 0
    risk_budget = equity * p["risk_pct"]
    by_risk = int(np.floor(risk_budget / stop_distance))
    by_cash = int(np.floor(equity / entry))
    return max(0, min(by_risk, by_cash))
