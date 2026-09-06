from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "sma_fast": 50,
    "sma_slow": 200,
    "mom_fast": 63,
    "mom_mid": 126,
    "mom_slow": 252,
    "vol_lookback": 63,
    "top_n": 2,
    "bull_core_weight": 0.55,
    "bull_satellite_weight": 0.45,
    "normal_core_weight": 0.70,
    "normal_satellite_weight": 0.30,
    "caution_exposure": 0.50,
    "stress_exposure": 0.25,
    "stress_vol_threshold": 0.30,
    "max_asset_weight": 0.60,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0002,
}


def add_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    out = df.copy().sort_index()
    c = out["Close"]
    r = c.pct_change()
    out["sma_fast"] = c.rolling(p["sma_fast"]).mean()
    out["sma_slow"] = c.rolling(p["sma_slow"]).mean()
    out["mom_fast"] = c / c.shift(p["mom_fast"]) - 1.0
    out["mom_mid"] = c / c.shift(p["mom_mid"]) - 1.0
    out["mom_slow"] = c / c.shift(p["mom_slow"]) - 1.0
    out["vol"] = r.rolling(p["vol_lookback"]).std() * np.sqrt(252)
    raw = 0.45 * out["mom_fast"] + 0.35 * out["mom_mid"] + 0.20 * out["mom_slow"]
    out["score"] = raw / out["vol"].replace(0, np.nan)
    out["eligible"] = (
        (c > out["sma_slow"]) &
        (out["mom_mid"] > 0) &
        out["score"].notna()
    )
    return out


def classify_regime(spy_features: pd.DataFrame, date: pd.Timestamp, params: dict | None = None) -> str:
    p = {**DEFAULTS, **(params or {})}
    hist = spy_features.loc[:date]
    if hist.empty:
        return "BEAR"
    row = hist.iloc[-1]
    needed = ["sma_fast", "sma_slow", "mom_fast", "mom_mid", "vol"]
    if any(pd.isna(row.get(k)) for k in needed):
        return "BEAR"

    close = float(row["Close"])
    sma_fast = float(row["sma_fast"])
    sma_slow = float(row["sma_slow"])
    mom_fast = float(row["mom_fast"])
    mom_mid = float(row["mom_mid"])
    vol = float(row["vol"])

    if close > sma_slow and sma_fast > sma_slow and mom_fast > 0 and mom_mid > 0:
        return "BULL_STRESS" if vol > p["stress_vol_threshold"] else "STRONG_BULL"
    if close > sma_slow and mom_mid > 0:
        return "NORMAL_BULL"
    if close > sma_slow or mom_mid > 0:
        return "CAUTION"
    return "BEAR"


def _satellite_weights(feature_map: dict[str, pd.DataFrame], date: pd.Timestamp, budget: float,
                       params: dict | None = None) -> dict[str, float]:
    p = {**DEFAULTS, **(params or {})}
    candidates: list[tuple[str, float, float]] = []
    for symbol, df in feature_map.items():
        hist = df.loc[:date]
        if hist.empty:
            continue
        row = hist.iloc[-1]
        if bool(row.get("eligible", False)) and pd.notna(row.get("score")) and pd.notna(row.get("vol")):
            candidates.append((symbol, float(row["score"]), max(float(row["vol"]), 1e-9)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[: p["top_n"]]
    if not selected or budget <= 0:
        return {}
    inv_vol = np.array([1.0 / x[2] for x in selected], dtype=float)
    raw = inv_vol / inv_vol.sum()
    capped = np.minimum(raw, p["max_asset_weight"])
    capped = capped / capped.sum() * budget
    return {selected[i][0]: float(capped[i]) for i in range(len(selected))}


def target_weights(feature_map: dict[str, pd.DataFrame], spy_features: pd.DataFrame,
                   date: pd.Timestamp, params: dict | None = None) -> tuple[dict[str, float], str]:
    p = {**DEFAULTS, **(params or {})}
    regime = classify_regime(spy_features, date, p)

    if regime == "STRONG_BULL":
        core = p["bull_core_weight"]
        sat_budget = p["bull_satellite_weight"]
    elif regime == "BULL_STRESS":
        core = p["stress_exposure"]
        sat_budget = 0.0
    elif regime == "NORMAL_BULL":
        core = p["normal_core_weight"]
        sat_budget = p["normal_satellite_weight"]
    elif regime == "CAUTION":
        core = p["caution_exposure"]
        sat_budget = 0.0
    else:
        return {}, regime

    weights: dict[str, float] = {"SPY": float(core)} if core > 0 else {}
    sat_map = {k: v for k, v in feature_map.items() if k != "SPY"}
    for s, w in _satellite_weights(sat_map, date, sat_budget, p).items():
        weights[s] = weights.get(s, 0.0) + w

    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        weights = {s: w / total for s, w in weights.items()}
    return weights, regime
