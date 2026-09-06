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
    "stress_vol_lookback": 252,
    "stress_vol_quantile": 0.90,
    "top_n": 2,
    "breadth_strong": 0.60,
    "breadth_normal": 0.45,
    "strong_spy_weight": 0.45,
    "strong_qqq_weight": 0.25,
    "strong_satellite_weight": 0.30,
    "normal_spy_weight": 0.55,
    "normal_qqq_weight": 0.15,
    "normal_satellite_weight": 0.20,
    "normal_cash_weight": 0.10,
    "caution_spy_weight": 0.60,
    "recovery_spy_weight": 0.35,
    "stress_multiplier": 0.75,
    "max_asset_weight": 0.60,
    "dd_brake_1": 0.18,
    "dd_brake_2": 0.25,
    "dd_brake_1_multiplier": 0.50,
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
    raw = 0.50 * out["mom_fast"] + 0.30 * out["mom_mid"] + 0.20 * out["mom_slow"]
    out["score"] = raw / out["vol"].replace(0, np.nan)
    out["eligible"] = (
        (c > out["sma_slow"]) &
        (out["mom_mid"] > 0) &
        out["score"].notna()
    )
    return out


def breadth(feature_map: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    flags = []
    for df in feature_map.values():
        hist = df.loc[:date]
        if hist.empty:
            continue
        row = hist.iloc[-1]
        if pd.isna(row.get("sma_slow")) or pd.isna(row.get("mom_mid")):
            continue
        flags.append(bool(float(row["Close"]) > float(row["sma_slow"]) and float(row["mom_mid"]) > 0))
    return float(np.mean(flags)) if flags else 0.0


def stress_flag(spy_features: pd.DataFrame, date: pd.Timestamp, params: dict | None = None) -> bool:
    p = {**DEFAULTS, **(params or {})}
    hist = spy_features.loc[:date]
    if len(hist) < p["stress_vol_lookback"] + 5:
        return False
    vol = hist["vol"].dropna()
    if len(vol) < p["stress_vol_lookback"]:
        return False
    current = float(vol.iloc[-1])
    threshold = float(vol.iloc[-p["stress_vol_lookback"]:].quantile(p["stress_vol_quantile"]))
    return current > threshold


def classify_regime(feature_map: dict[str, pd.DataFrame], spy_features: pd.DataFrame,
                    date: pd.Timestamp, params: dict | None = None) -> tuple[str, float, bool]:
    p = {**DEFAULTS, **(params or {})}
    hist = spy_features.loc[:date]
    if hist.empty:
        return "BEAR", 0.0, False
    row = hist.iloc[-1]
    needed = ["sma_fast", "sma_slow", "mom_fast", "mom_mid"]
    if any(pd.isna(row.get(k)) for k in needed):
        return "BEAR", 0.0, False

    close = float(row["Close"])
    sma_fast = float(row["sma_fast"])
    sma_slow = float(row["sma_slow"])
    mom_fast = float(row["mom_fast"])
    mom_mid = float(row["mom_mid"])
    br = breadth(feature_map, date)
    stress = stress_flag(spy_features, date, p)

    if close > sma_slow and sma_fast > sma_slow and mom_fast > 0 and mom_mid > 0 and br >= p["breadth_strong"]:
        return "STRONG_BULL", br, stress
    if close > sma_slow and mom_mid > 0 and br >= p["breadth_normal"]:
        return "NORMAL_BULL", br, stress
    if close > sma_slow:
        return "CAUTION", br, stress
    if mom_fast > 0 and br >= p["breadth_normal"]:
        return "RECOVERY", br, stress
    return "BEAR", br, stress


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

    # Mix relative-strength rank and inverse volatility instead of using either one alone.
    scores = np.array([max(x[1], 1e-9) for x in selected], dtype=float)
    inv_vol = np.array([1.0 / x[2] for x in selected], dtype=float)
    blend = 0.60 * (scores / scores.sum()) + 0.40 * (inv_vol / inv_vol.sum())
    capped = np.minimum(blend, p["max_asset_weight"])
    capped = capped / capped.sum() * budget
    return {selected[i][0]: float(capped[i]) for i in range(len(selected))}


def target_weights(feature_map: dict[str, pd.DataFrame], spy_features: pd.DataFrame,
                   date: pd.Timestamp, params: dict | None = None) -> tuple[dict[str, float], str, float, bool]:
    p = {**DEFAULTS, **(params or {})}
    regime, br, stress = classify_regime(feature_map, spy_features, date, p)

    weights: dict[str, float] = {}
    if regime == "STRONG_BULL":
        weights["SPY"] = p["strong_spy_weight"]
        weights["QQQ"] = p["strong_qqq_weight"]
        sat_budget = p["strong_satellite_weight"]
    elif regime == "NORMAL_BULL":
        weights["SPY"] = p["normal_spy_weight"]
        weights["QQQ"] = p["normal_qqq_weight"]
        sat_budget = p["normal_satellite_weight"]
    elif regime == "CAUTION":
        weights["SPY"] = p["caution_spy_weight"]
        sat_budget = 0.0
    elif regime == "RECOVERY":
        weights["SPY"] = p["recovery_spy_weight"]
        sat_budget = 0.0
    else:
        return {}, regime, br, stress

    excluded = {"SPY", "QQQ"}
    sat_map = {k: v for k, v in feature_map.items() if k not in excluded}
    for s, w in _satellite_weights(sat_map, date, sat_budget, p).items():
        weights[s] = weights.get(s, 0.0) + w

    # Stress is relative to the market's own history, so it should be rare. Reduce, do not abandon, exposure.
    if stress:
        weights = {s: w * p["stress_multiplier"] for s, w in weights.items()}

    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        weights = {s: w / total for s, w in weights.items()}
    return weights, regime, br, stress
