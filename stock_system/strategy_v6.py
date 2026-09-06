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
    "breadth_strong": 0.60,
    "breadth_weak": 0.35,
    "top_n": 2,
    "bull_spy": 0.60,
    "bull_qqq": 0.20,
    "bull_sat": 0.20,
    "normal_spy": 0.75,
    "normal_qqq": 0.15,
    "normal_sat": 0.10,
    "caution_spy": 0.70,
    "bear_spy": 0.20,
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
    raw = 0.50 * out["mom_fast"] + 0.30 * out["mom_mid"] + 0.20 * out["mom_slow"]
    out["score"] = raw / out["vol"].replace(0, np.nan)
    out["eligible"] = (c > out["sma_slow"]) & (out["mom_mid"] > 0) & out["score"].notna()
    return out


def breadth(feature_map: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    vals = []
    for df in feature_map.values():
        h = df.loc[:date]
        if h.empty:
            continue
        r = h.iloc[-1]
        if pd.isna(r.get("sma_slow")) or pd.isna(r.get("mom_mid")):
            continue
        vals.append(bool(float(r["Close"]) > float(r["sma_slow"]) and float(r["mom_mid"]) > 0))
    return float(np.mean(vals)) if vals else 0.0


def classify(feature_map: dict[str, pd.DataFrame], spy_features: pd.DataFrame,
             date: pd.Timestamp, params: dict | None = None) -> tuple[str, float]:
    p = {**DEFAULTS, **(params or {})}
    h = spy_features.loc[:date]
    if h.empty:
        return "BEAR", 0.0
    r = h.iloc[-1]
    needed = ["sma_fast", "sma_slow", "mom_fast", "mom_mid"]
    if any(pd.isna(r.get(k)) for k in needed):
        return "BEAR", 0.0
    close = float(r["Close"])
    sf = float(r["sma_fast"])
    ss = float(r["sma_slow"])
    mf = float(r["mom_fast"])
    mm = float(r["mom_mid"])
    br = breadth(feature_map, date)

    if close > ss and sf > ss and mf > 0 and mm > 0 and br >= p["breadth_strong"]:
        return "STRONG_BULL", br
    if close > ss and mm > 0:
        return "NORMAL_BULL", br
    if close > ss or br >= p["breadth_weak"]:
        return "CAUTION", br
    return "BEAR", br


def _satellites(feature_map: dict[str, pd.DataFrame], date: pd.Timestamp, budget: float,
                params: dict | None = None) -> dict[str, float]:
    p = {**DEFAULTS, **(params or {})}
    cands: list[tuple[str, float, float]] = []
    for s, df in feature_map.items():
        h = df.loc[:date]
        if h.empty:
            continue
        r = h.iloc[-1]
        if bool(r.get("eligible", False)) and pd.notna(r.get("score")) and pd.notna(r.get("vol")):
            cands.append((s, float(r["score"]), max(float(r["vol"]), 1e-9)))
    cands.sort(key=lambda x: x[1], reverse=True)
    selected = cands[: p["top_n"]]
    if not selected or budget <= 0:
        return {}
    score = np.array([max(x[1], 1e-9) for x in selected], dtype=float)
    invvol = np.array([1.0 / x[2] for x in selected], dtype=float)
    blend = 0.7 * score / score.sum() + 0.3 * invvol / invvol.sum()
    blend = np.minimum(blend, p["max_asset_weight"])
    blend = blend / blend.sum() * budget
    return {selected[i][0]: float(blend[i]) for i in range(len(selected))}


def target_weights(feature_map: dict[str, pd.DataFrame], spy_features: pd.DataFrame,
                   date: pd.Timestamp, params: dict | None = None) -> tuple[dict[str, float], str, float]:
    p = {**DEFAULTS, **(params or {})}
    regime, br = classify(feature_map, spy_features, date, p)

    if regime == "STRONG_BULL":
        weights = {"SPY": p["bull_spy"], "QQQ": p["bull_qqq"]}
        sat_budget = p["bull_sat"]
    elif regime == "NORMAL_BULL":
        weights = {"SPY": p["normal_spy"], "QQQ": p["normal_qqq"]}
        sat_budget = p["normal_sat"]
    elif regime == "CAUTION":
        weights = {"SPY": p["caution_spy"]}
        sat_budget = 0.0
    else:
        weights = {"SPY": p["bear_spy"]}
        sat_budget = 0.0

    sat_map = {k: v for k, v in feature_map.items() if k not in {"SPY", "QQQ"}}
    for s, w in _satellites(sat_map, date, sat_budget, p).items():
        weights[s] = weights.get(s, 0.0) + w

    total = sum(weights.values())
    if total > 1.0:
        weights = {s: w / total for s, w in weights.items()}
    return weights, regime, br
