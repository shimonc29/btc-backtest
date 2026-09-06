from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULTS = {
    "mom_fast": 63,      # ~3 months
    "mom_mid": 126,      # ~6 months
    "mom_slow": 252,     # ~12 months
    "sma_long": 200,
    "vol_lookback": 63,
    "market_slope_lookback": 20,
    "top_n": 2,
    "max_asset_weight": 0.65,
    "stress_vol_threshold": 0.30,
    "stress_exposure": 0.50,
    "normal_exposure": 1.00,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0002,
}


def add_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    out = df.copy().sort_index()
    close = out["Close"]
    ret = close.pct_change()

    out["sma_long"] = close.rolling(p["sma_long"]).mean()
    out["mom_fast"] = close / close.shift(p["mom_fast"]) - 1.0
    out["mom_mid"] = close / close.shift(p["mom_mid"]) - 1.0
    out["mom_slow"] = close / close.shift(p["mom_slow"]) - 1.0
    out["vol"] = ret.rolling(p["vol_lookback"]).std() * np.sqrt(252)

    # Multi-horizon relative-strength score, lightly penalized for volatility.
    raw = 0.50 * out["mom_fast"] + 0.30 * out["mom_mid"] + 0.20 * out["mom_slow"]
    out["score"] = raw / out["vol"].replace(0, np.nan)

    out["eligible"] = (
        (close > out["sma_long"]) &
        (out["mom_mid"] > 0) &
        out["score"].notna()
    )
    return out


def market_regime(spy: pd.DataFrame, date: pd.Timestamp, params: dict | None = None) -> tuple[bool, float]:
    p = {**DEFAULTS, **(params or {})}
    hist = spy.loc[:date]
    if len(hist) < max(p["sma_long"], p["market_slope_lookback"]) + 2:
        return False, 0.0

    row = hist.iloc[-1]
    sma = hist["Close"].rolling(p["sma_long"]).mean()
    current_sma = float(sma.iloc[-1])
    prior_sma = float(sma.iloc[-1 - p["market_slope_lookback"]])
    risk_on = float(row["Close"]) > current_sma and current_sma > prior_sma

    recent_vol = hist["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    exposure = p["stress_exposure"] if pd.notna(recent_vol) and recent_vol > p["stress_vol_threshold"] else p["normal_exposure"]
    return bool(risk_on), float(exposure if risk_on else 0.0)


def target_weights(feature_map: dict[str, pd.DataFrame], spy: pd.DataFrame, date: pd.Timestamp,
                   params: dict | None = None) -> dict[str, float]:
    p = {**DEFAULTS, **(params or {})
    }
    risk_on, total_exposure = market_regime(spy, date, p)
    if not risk_on or total_exposure <= 0:
        return {}

    candidates: list[tuple[str, float, float]] = []
    for symbol, df in feature_map.items():
        hist = df.loc[:date]
        if hist.empty:
            continue
        row = hist.iloc[-1]
        if bool(row.get("eligible", False)) and pd.notna(row.get("score")) and pd.notna(row.get("vol")):
            candidates.append((symbol, float(row["score"]), float(row["vol"])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[: p["top_n"]]
    if not selected:
        return {}

    inv_vol = np.array([1.0 / max(v, 1e-9) for _, _, v in selected], dtype=float)
    raw = inv_vol / inv_vol.sum()

    # Cap concentration, then renormalize the remainder. With top_n=2 this keeps both names meaningful.
    capped = np.minimum(raw, p["max_asset_weight"])
    if capped.sum() <= 0:
        return {}
    capped = capped / capped.sum() * total_exposure

    return {selected[i][0]: float(capped[i]) for i in range(len(selected))}
