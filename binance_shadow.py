"""Read-only Binance connector for BTC V3 shadow monitoring.

Security contract:
- Only HTTP GET requests are implemented.
- No order, transfer, withdrawal, margin, futures, or account-write endpoints exist.
- Credentials are read only from environment variables.
- Intended for local/VPS shadow use because GitHub-hosted runners may receive Binance HTTP 451.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.binance.com"
USER_AGENT = "btc-v3-binance-shadow/1.0"


@dataclass(frozen=True)
class ShadowConfig:
    api_key: str | None
    api_secret: str | None
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls) -> "ShadowConfig":
        return cls(
            api_key=(os.getenv("BINANCE_API_KEY") or "").strip() or None,
            api_secret=(os.getenv("BINANCE_API_SECRET") or "").strip() or None,
            base_url=(os.getenv("BINANCE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


class BinanceShadowConnector:
    """Minimal read-only Binance Spot REST client."""

    def __init__(self, config: ShadowConfig | None = None, timeout: int = 20):
        self.config = config or ShadowConfig.from_env()
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None, *, signed: bool = False) -> Any:
        params = dict(params or {})
        headers = {"User-Agent": USER_AGENT}

        if signed:
            if not self.config.has_credentials:
                raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required for signed read-only calls")
            params.setdefault("timestamp", int(time.time() * 1000))
            params.setdefault("recvWindow", 5000)
            query = urlencode(params)
            signature = hmac.new(
                self.config.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.config.api_key

        query = urlencode(params)
        url = f"{self.config.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # Public read-only endpoints
    def ping(self) -> dict[str, Any]:
        return self._get("/api/v3/ping")

    def server_time(self) -> dict[str, Any]:
        return self._get("/api/v3/time")

    def ticker_price(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        return self._get("/api/v3/ticker/price", {"symbol": symbol})

    def exchange_info(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        return self._get("/api/v3/exchangeInfo", {"symbol": symbol})

    # Signed GET-only account endpoints
    def account(self) -> dict[str, Any]:
        return self._get("/api/v3/account", {"omitZeroBalances": "true"}, signed=True)

    def open_orders(self, symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
        return self._get("/api/v3/openOrders", {"symbol": symbol}, signed=True)

    def balances(self) -> dict[str, float]:
        account = self.account()
        out: dict[str, float] = {}
        for item in account.get("balances", []):
            asset = str(item.get("asset", ""))
            free = float(item.get("free", 0) or 0)
            locked = float(item.get("locked", 0) or 0)
            if asset:
                out[asset] = free + locked
        return out


def self_test() -> dict[str, Any]:
    """Offline safety test: confirms the connector exposes only intended read methods."""
    public_methods = {
        name
        for name in dir(BinanceShadowConnector)
        if not name.startswith("_") and callable(getattr(BinanceShadowConnector, name))
    }
    allowed = {
        "ping",
        "server_time",
        "ticker_price",
        "exchange_info",
        "account",
        "open_orders",
        "balances",
    }
    unexpected = sorted(public_methods - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected public methods in shadow connector: {unexpected}")

    return {
        "mode": "READ_ONLY_SHADOW",
        "network_called": False,
        "http_methods_implemented": ["GET"],
        "public_methods": sorted(public_methods),
        "orders_enabled": False,
        "withdrawals_enabled": False,
        "transfers_enabled": False,
    }


def live_read_check() -> dict[str, Any]:
    connector = BinanceShadowConnector()
    result: dict[str, Any] = {
        "mode": "READ_ONLY_SHADOW",
        "ticker": connector.ticker_price("BTCUSDT"),
        "has_credentials": connector.config.has_credentials,
    }
    if connector.config.has_credentials:
        balances = connector.balances()
        result["balances"] = {k: balances[k] for k in ("BTC", "USDT") if k in balances}
        result["open_orders_count"] = len(connector.open_orders("BTCUSDT"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance READ-ONLY / SHADOW connector")
    parser.add_argument("--self-test", action="store_true", help="Run offline safety validation")
    parser.add_argument("--live-read-check", action="store_true", help="Perform read-only network calls")
    args = parser.parse_args()

    if args.live_read_check:
        print(json.dumps(live_read_check(), indent=2, sort_keys=True))
    else:
        print(json.dumps(self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
