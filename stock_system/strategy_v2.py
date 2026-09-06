from __future__ import annotations

import pandas as pd

DEFAULTS = {
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "atr_period": 14,
    "rsi_period": 3,
    "rsi_entry_max": 35.0,
    "stop_atr": 1.5,
    "trail_atr": 2.5,
    "risk_pct": 0.005,
    "max_positions": 3,
    "kill_switch_drawdown": 0.12,
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0)


def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    out = df.copy().sort_index()
    out["ema_fast"] = out["Close"].ewm(span=p["ema_fast"], adjust=False).mean()
    out["ema_mid"] = out["Close"].ewm(span=p["ema_mid"], adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=p["ema_slow"], adjust=False).mean()
    out["ema_slow_slope"] = out["ema_slow"] - out["ema_slow"].shift(20)

    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(p["atr_period"]).mean()
    out["rsi"] = _rsi(out["Close"], p["rsi_period"])

    out["trend_ok"] = (
        (out["Close"] > out["ema_slow"]) &
        (out["ema_mid"] > out["ema_slow"]) &
        (out["ema_slow_slope"] > 0)
    )
    out["pullback_ok"] = (
        (out["Close"] < out["ema_fast"]) &
        (out["Close"] > out["ema_mid"]) &
        (out["rsi"] <= p["rsi_entry_max"])
    )
    out["entry_signal"] = out["trend_ok"] & out["pullback_ok"] & out["atr"].notna()
    return out
