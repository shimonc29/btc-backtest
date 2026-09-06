from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PAPER_DIR = Path("stock_system/paper_v6")
SNAPSHOT_PATH = PAPER_DIR / "latest_snapshot.json"
TRADES_PATH = PAPER_DIR / "trades.csv"
ALERT_PATH = PAPER_DIR / "latest_alert.json"
SENT_PATH = PAPER_DIR / "last_sent_alert.json"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_alert(snapshot: dict) -> dict:
    pending = snapshot.get("pending")
    weights = pending.get("weights", {}) if pending else {}
    current = snapshot.get("current_weights", {}) or {}

    if pending:
        if pending.get("bootstrap"):
            action = "ENTER"
            title = "כניסה ראשונית לפי V6"
        else:
            action = "REBALANCE"
            title = "איזון תיק לפי V6"
        parts = [f"{s} {pct(float(w))}" for s, w in sorted(weights.items(), key=lambda x: -float(x[1]))]
        message = f"{title}: " + ", ".join(parts) + ". ביצוע מתוכנן בפתיחת יום המסחר הבא."
        signal_date = pending.get("signal_date")
        regime = pending.get("regime")
        alert_id = f"{action}:{signal_date}:{regime}:" + ";".join(f"{s}={float(w):.6f}" for s, w in sorted(weights.items()))
    elif current:
        action = "HOLD"
        title = "החזקה"
        parts = [f"{s} {pct(float(w))}" for s, w in sorted(current.items(), key=lambda x: -float(x[1]))]
        message = "אין שינוי נדרש כרגע. התיק הנוכחי: " + ", ".join(parts) + "."
        signal_date = snapshot.get("as_of")
        regime = None
        alert_id = f"HOLD:{snapshot.get('as_of')}"
    else:
        action = "WAIT"
        title = "ממתינים"
        message = "אין כרגע פוזיציה ואין הוראת V6 ממתינה."
        signal_date = snapshot.get("as_of")
        regime = None
        alert_id = f"WAIT:{snapshot.get('as_of')}"

    return {
        "alert_id": alert_id,
        "as_of": snapshot.get("as_of"),
        "action": action,
        "title": title,
        "message_he": message,
        "regime": regime,
        "signal_date": signal_date,
        "target_weights": weights,
        "current_weights": current,
        "paper_equity_usd": snapshot.get("paper_equity_usd"),
        "cash_usd": snapshot.get("cash_usd"),
        "trade_count": snapshot.get("trade_count"),
        "paper_only": True,
        "execution_note": "Signal only; no broker order is sent by this alert layer.",
    }


def maybe_send_telegram(alert: dict) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False

    prior = {}
    if SENT_PATH.exists():
        try:
            prior = json.loads(SENT_PATH.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
    if prior.get("alert_id") == alert["alert_id"]:
        return False

    text = "📊 V6 Stock System\n" + alert["message_he"]
    data = urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST")
    with urlopen(req, timeout=20) as r:
        if r.status >= 300:
            raise RuntimeError(f"Telegram send failed: HTTP {r.status}")
    SENT_PATH.write_text(json.dumps({"alert_id": alert["alert_id"]}, indent=2), encoding="utf-8")
    return True


def main() -> None:
    if not SNAPSHOT_PATH.exists():
        raise RuntimeError("Missing paper snapshot")
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    alert = build_alert(snapshot)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(json.dumps(alert, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    sent = maybe_send_telegram(alert)
    print(json.dumps({"alert": alert, "telegram_sent": sent}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
