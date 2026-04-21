"""
SignalDispatcher — formats Signal objects and pushes them via Telegram Bot API.

Uses plain requests (no python-telegram-bot dependency) to avoid an extra
heavy dependency.  Credentials are loaded from .env via config.py.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from trading_bot.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from trading_bot.inference.signal import Signal

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SEC = 15


class SignalDispatcher:
    """
    Sends trading signals to a Telegram chat.

    Parameters
    ----------
    bot_token:
        Telegram bot token.  Falls back to TELEGRAM_BOT_TOKEN from .env.
    chat_id:
        Target chat/user ID.  Falls back to TELEGRAM_CHAT_ID from .env.
    send_hold_signals:
        When True, low-confidence / HOLD signals are also dispatched.
        Default False — only BUY/SELL with ≥ MEDIUM confidence are sent.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        send_hold_signals: bool = True,  # TODO: Set back to False after live test
    ) -> None:
        self._token = bot_token or TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or TELEGRAM_CHAT_ID
        self._send_hold = send_hold_signals

        if not self._token:
            logger.warning("TELEGRAM_BOT_TOKEN not set — signals will not be delivered.")
        if not self._chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set — signals will not be delivered.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def dispatch(self, signal: Signal) -> bool:
        """
        Format *signal* and push it to Telegram.

        Parameters
        ----------
        signal:
            Completed Signal from EnsemblePredictor.

        Returns
        -------
        bool
            True if the message was accepted by Telegram (HTTP 200 + ok=true).
        """
        if not self._is_configured():
            logger.error("Telegram credentials missing — skipping dispatch.")
            return False

        if not signal.is_actionable and not self._send_hold:
            logger.info(
                "Signal for %s is LOW confidence — skipping Telegram dispatch.", signal.ticker
            )
            return False

        text = self._format(signal)
        return self._send(text)

    def dispatch_batch(self, signals: list[Signal]) -> dict[str, bool]:
        """Dispatch a list of signals; returns {ticker: success} mapping."""
        results: dict[str, bool] = {}
        for sig in signals:
            results[sig.ticker] = self.dispatch(sig)
        return results

    def send_text(self, message: str) -> bool:
        """Send a raw text message (e.g. error alerts, status pings)."""
        if not self._is_configured():
            return False
        return self._send(message)

    # ── Formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format(signal: Signal) -> str:
        """Delegate to Signal's own Markdown formatters."""
        if signal.is_actionable:
            return signal.to_telegram_message()
        return signal.to_hold_message()

    # ── HTTP layer ─────────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        """POST *text* to the Telegram sendMessage endpoint."""
        url = _TELEGRAM_API.format(token=self._token)
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                logger.info("Telegram message sent (message_id=%s).", data["result"]["message_id"])
                return True
            logger.error("Telegram API returned ok=false: %s", data)
            return False
        except requests.RequestException as exc:
            logger.error("Telegram dispatch failed: %s", exc)
            return False

    def _is_configured(self) -> bool:
        return bool(self._token and self._chat_id)
