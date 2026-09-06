from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import run_v6
from strategy_v6 import DEFAULTS, add_features, target_weights

PAPER_DIR = Path("stock_system/paper_v6")
STATE_PATH = PAPER_DIR / "state.json"
SNAPSHOT_PATH = PAPER_DIR / "latest_snapshot.json"
TRADES_PATH = PAPER_DIR / "trades.csv"
INITIAL_PAPER_USD = 1500.0
# V3 is a deliberate clean-forward reset: no historical replay is allowed.
PAPER_VERSION = 3
TRADE_COLUMNS = ["signal_date", "execution_date", "symbol", "side", "qty", "fill", "fee", "regime", "bootstrap"]


def _fresh_state() -> dict[str, Any]:
    return {
        "engine": "V6 frozen paper clean-forward",
        "version": PAPER_VERSION,
        "initial_capital_usd": INITIAL_PAPER_USD,
        "cash": INITIAL_PAPER_USD,
        "shares": {s: 0.0 for s in run_v6.UNIVERSE},
        "pending": None,
        "last_processed_session": None,
        "last_monthly_signal": None,
        "started": None,
        "trade_count": 0,
        "notes": [
            "Paper only. No broker orders are sent.",
            "No historical trade replay: the paper account starts from the reset date and only moves forward.",
            "Signals use adjusted Yahoo closes; simulated fills use adjusted next-session open.",
            "The first allocation is a bootstrap signal and is tagged separately from normal month-end rebalances.",
        ],
    }


def _load_state() -> tuple[dict[str, Any], bool]:
    if not STATE_PATH.exists():
        return _fresh_state(), True
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if int(state.get("version", 0)) != PAPER_VERSION:
        return _fresh_state(), True
    return state, False


def _save_state(state: dict[str, Any]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _reset_trade_log() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADES_PATH, index=False)


def _append_trades(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if TRADES_PATH.exists():
        old = pd.read_csv(TRADES_PATH)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(TRADES_PATH, index=False)


def _equity_at_close(state: dict[str, Any], raw: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    eq = float(state["cash"])
    for s in run_v6.UNIVERSE:
        sh = float(state["shares"].get(s, 0.0))
        if sh == 0:
            continue
        h = raw[s].loc[:date]
        if not h.empty:
            eq += sh * float(h.iloc[-1]["Close"])
    return eq


def _execute_pending(state: dict[str, Any], raw: dict[str, pd.DataFrame], session: pd.Timestamp) -> list[dict[str, Any]]:
    pending = state.get("pending")
    if not pending:
        return []
    signal_date = pd.Timestamp(pending["signal_date"])
    if session <= signal_date:
        return []

    p = DEFAULTS
    previous_sessions = raw["SPY"].index[raw["SPY"].index < session]
    if len(previous_sessions) == 0:
        return []
    prev = previous_sessions[-1]
    eq_before = _equity_at_close(state, raw, prev)
    weights = {s: float(w) for s, w in pending["weights"].items()}
    assert sum(weights.values()) <= 1.0000001
    assert all(w >= -1e-12 for w in weights.values())

    target_dollars = {s: eq_before * weights.get(s, 0.0) for s in run_v6.UNIVERSE}
    rows: list[dict[str, Any]] = []

    for s in run_v6.UNIVERSE:
        if session not in raw[s].index:
            continue
        open_px = float(raw[s].loc[session, "Open"])
        current = float(state["shares"].get(s, 0.0)) * open_px
        target = target_dollars[s]
        if current <= target + 1e-8:
            continue
        qty = min(float(state["shares"].get(s, 0.0)), (current - target) / open_px)
        if qty <= 1e-10:
            continue
        fill = open_px * (1 - p["slippage_rate"])
        gross = qty * fill
        fee = gross * p["fee_rate"]
        state["cash"] = float(state["cash"]) + gross - fee
        state["shares"][s] = float(state["shares"].get(s, 0.0)) - qty
        rows.append({
            "signal_date": pending["signal_date"], "execution_date": session.date().isoformat(),
            "symbol": s, "side": "SELL", "qty": qty, "fill": fill, "fee": fee,
            "regime": pending["regime"], "bootstrap": pending.get("bootstrap", False),
        })

    for s in run_v6.UNIVERSE:
        if session not in raw[s].index:
            continue
        open_px = float(raw[s].loc[session, "Open"])
        current = float(state["shares"].get(s, 0.0)) * open_px
        desired = max(0.0, target_dollars[s] - current)
        if desired <= 1e-8 or float(state["cash"]) <= 0:
            continue
        fill = open_px * (1 + p["slippage_rate"])
        spend = min(desired, float(state["cash"]) / (1 + p["fee_rate"]))
        qty = spend / fill
        if qty <= 1e-10:
            continue
        cost = qty * fill
        fee = cost * p["fee_rate"]
        state["cash"] = float(state["cash"]) - cost - fee
        state["shares"][s] = float(state["shares"].get(s, 0.0)) + qty
        rows.append({
            "signal_date": pending["signal_date"], "execution_date": session.date().isoformat(),
            "symbol": s, "side": "BUY", "qty": qty, "fill": fill, "fee": fee,
            "regime": pending["regime"], "bootstrap": pending.get("bootstrap", False),
        })

    if float(state["cash"]) < -1e-6:
        raise RuntimeError(f"Paper cash went negative: {state['cash']}")
    state["trade_count"] = int(state.get("trade_count", 0)) + len(rows)
    state["pending"] = None
    return rows


def main() -> None:
    raw = {s: run_v6.fetch_yahoo(s) for s in run_v6.UNIVERSE}
    features = {s: add_features(df, DEFAULTS) for s, df in raw.items()}
    common_end = min(df.index.max() for df in raw.values())
    sessions = raw["SPY"].index[raw["SPY"].index <= common_end]
    latest = sessions[-1]

    state, reset_required = _load_state()
    if reset_required:
        _reset_trade_log()
        state["started"] = latest.date().isoformat()
        state["last_processed_session"] = latest.date().isoformat()
        weights, regime, br = target_weights(features, features["SPY"], latest, DEFAULTS)
        state["pending"] = {
            "signal_date": latest.date().isoformat(),
            "regime": regime,
            "breadth": br,
            "weights": weights,
            "bootstrap": True,
        }
        _save_state(state)
    else:
        last_processed = pd.Timestamp(state["last_processed_session"]) if state.get("last_processed_session") else latest
        new_sessions = [d for d in sessions if d > last_processed]
        all_trade_rows: list[dict[str, Any]] = []

        for session in new_sessions:
            all_trade_rows.extend(_execute_pending(state, raw, session))

            future = sessions[sessions > session]
            next_session = future[0] if len(future) else None
            is_month_end = next_session is not None and (next_session.year, next_session.month) != (session.year, session.month)
            month_key = f"{session.year:04d}-{session.month:02d}"
            if is_month_end and state.get("last_monthly_signal") != month_key:
                weights, regime, br = target_weights(features, features["SPY"], session, DEFAULTS)
                state["pending"] = {
                    "signal_date": session.date().isoformat(),
                    "regime": regime,
                    "breadth": br,
                    "weights": weights,
                    "bootstrap": False,
                }
                state["last_monthly_signal"] = month_key

            state["last_processed_session"] = session.date().isoformat()

        _append_trades(all_trade_rows)
        _save_state(state)

    latest_eq = _equity_at_close(state, raw, latest)
    current_weights = {}
    if latest_eq > 0:
        for s in run_v6.UNIVERSE:
            sh = float(state["shares"].get(s, 0.0))
            if sh:
                current_weights[s] = sh * float(raw[s].loc[latest, "Close"]) / latest_eq

    snapshot = {
        "as_of": latest.date().isoformat(),
        "paper_equity_usd": latest_eq,
        "cash_usd": float(state["cash"]),
        "shares": state["shares"],
        "current_weights": current_weights,
        "pending": state.get("pending"),
        "trade_count": int(state.get("trade_count", 0)),
        "last_processed_session": state.get("last_processed_session"),
        "status": "PAPER_ONLY_NO_BROKER_CLEAN_FORWARD",
        "version": PAPER_VERSION,
    }
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
