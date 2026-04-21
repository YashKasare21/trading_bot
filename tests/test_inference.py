"""Tests for Phase 4 inference modules: risk, notifier, signal, predictor."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from trading_bot.inference.risk import RiskLevels, RiskManager
from trading_bot.inference.signal import Action, Confidence, Signal

_IST = ZoneInfo("Asia/Kolkata")

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_signal(action: Action = Action.BUY, confidence: Confidence = Confidence.MEDIUM) -> Signal:
    return Signal(
        ticker="RELIANCE.NS",
        action=action,
        confidence=confidence,
        generated_at=datetime.now(_IST),
        current_price=2500.0,
        entry_low=2477.5,
        entry_high=2522.5,
        stop_loss=2410.0,
        target=2657.5,
        risk_reward_ratio=2.0,
        sentiment_score=0.3,
        sentiment_label="Positive",
        market_regime=2,
        atr_14=45.0,
        votes={"SAC": "BUY", "PPO": "BUY", "RSI": "HOLD"},
        vote_count=2,
    )


# ── RiskManager ───────────────────────────────────────────────────────────────


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(sl_multiplier=1.5, tp_multiplier=3.0)

    def test_buy_levels_structure(self):
        levels = self.rm.calculate(close=1000.0, atr=20.0, action=Action.BUY)
        assert isinstance(levels, RiskLevels)
        assert levels.entry_low < levels.entry_price < levels.entry_high
        assert levels.stop_loss < levels.entry_low
        assert levels.target_price > levels.entry_high

    def test_sell_levels_structure(self):
        levels = self.rm.calculate(close=1000.0, atr=20.0, action=Action.SELL)
        assert levels.stop_loss > levels.entry_high
        assert levels.target_price < levels.entry_low

    def test_hold_returns_close_price(self):
        levels = self.rm.calculate(close=1234.5, atr=20.0, action=Action.HOLD)
        assert levels.entry_price == 1234.5
        assert levels.stop_loss == 1234.5
        assert levels.target_price == 1234.5
        assert levels.risk_reward_ratio == 0.0

    def test_risk_reward_ratio(self):
        levels = self.rm.calculate(close=1000.0, atr=20.0, action=Action.BUY)
        assert levels.risk_reward_ratio == pytest.approx(2.0, abs=0.01)

    def test_zero_atr_defaults_to_one(self):
        levels = self.rm.calculate(close=1000.0, atr=0.0, action=Action.BUY)
        assert levels.atr == 1.0

    def test_negative_atr_defaults_to_one(self):
        levels = self.rm.calculate(close=1000.0, atr=-5.0, action=Action.BUY)
        assert levels.atr == 1.0

    def test_buy_sl_distance(self):
        close, atr = 2500.0, 45.0
        levels = self.rm.calculate(close, atr, Action.BUY)
        expected_sl = (close - 0.5 * atr) - 1.5 * atr  # entry_low - 1.5*atr
        assert levels.stop_loss == pytest.approx(expected_sl, abs=0.01)

    def test_custom_multipliers(self):
        rm = RiskManager(sl_multiplier=2.0, tp_multiplier=4.0)
        levels = rm.calculate(1000.0, 10.0, Action.BUY)
        assert levels.risk_reward_ratio == pytest.approx(2.0, abs=0.01)

    def test_levels_are_rounded(self):
        levels = self.rm.calculate(close=1234.567, atr=13.333, action=Action.BUY)
        assert levels.entry_low == round(levels.entry_low, 2)
        assert levels.stop_loss == round(levels.stop_loss, 2)
        assert levels.target_price == round(levels.target_price, 2)


# ── Signal ────────────────────────────────────────────────────────────────────


class TestSignal:
    def test_is_actionable_medium(self):
        sig = _make_signal(Action.BUY, Confidence.MEDIUM)
        assert sig.is_actionable is True

    def test_is_actionable_high(self):
        sig = _make_signal(Action.BUY, Confidence.HIGH)
        assert sig.is_actionable is True

    def test_not_actionable_low(self):
        sig = _make_signal(Action.HOLD, Confidence.LOW)
        assert sig.is_actionable is False

    def test_to_dict_roundtrip(self):
        sig = _make_signal()
        restored = Signal.from_dict(sig.to_dict())
        assert restored.ticker == sig.ticker
        assert restored.action == sig.action
        assert restored.confidence == sig.confidence
        assert restored.current_price == sig.current_price

    def test_to_telegram_message_contains_key_fields(self):
        sig = _make_signal(Action.BUY, Confidence.MEDIUM)
        msg = sig.to_telegram_message()
        assert "RELIANCE.NS" in msg
        assert "BUY" in msg
        assert "Stop Loss" in msg or "SL" in msg or "Stop" in msg

    def test_to_hold_message_for_hold(self):
        sig = _make_signal(Action.HOLD, Confidence.LOW)
        msg = sig.to_hold_message()
        assert "NO SIGNAL" in msg or "HOLD" in msg

    def test_signal_id_is_stable(self):
        sig = _make_signal()
        assert sig.signal_id == sig.signal_id
        assert len(sig.signal_id) == 16


# ── SignalDispatcher ───────────────────────────────────────────────────────────


class TestSignalDispatcher:
    def test_skips_dispatch_when_credentials_missing(self):
        from trading_bot.inference.notifier import SignalDispatcher
        with patch("trading_bot.inference.notifier.TELEGRAM_BOT_TOKEN", ""), \
             patch("trading_bot.inference.notifier.TELEGRAM_CHAT_ID", ""):
            dispatcher = SignalDispatcher(bot_token=None, chat_id=None)
            sig = _make_signal(Action.BUY, Confidence.MEDIUM)
            result = dispatcher.dispatch(sig)
        assert result is False

    def test_skips_hold_signal_by_default(self):
        from trading_bot.inference.notifier import SignalDispatcher
        dispatcher = SignalDispatcher(bot_token="tok", chat_id="123", send_hold_signals=False)
        sig = _make_signal(Action.HOLD, Confidence.LOW)
        result = dispatcher.dispatch(sig)
        assert result is False

    def test_sends_hold_when_flag_set(self):
        from trading_bot.inference.notifier import SignalDispatcher
        dispatcher = SignalDispatcher(bot_token="tok", chat_id="123", send_hold_signals=True)
        sig = _make_signal(Action.HOLD, Confidence.LOW)

        with patch("trading_bot.inference.notifier.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"ok": True, "result": {"message_id": 1}}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            result = dispatcher.dispatch(sig)

        assert result is True

    def test_returns_false_on_request_exception(self):
        import requests as req_lib

        from trading_bot.inference.notifier import SignalDispatcher
        dispatcher = SignalDispatcher(bot_token="tok", chat_id="123")
        sig = _make_signal(Action.BUY, Confidence.MEDIUM)

        with patch("trading_bot.inference.notifier.requests.post") as mock_post:
            mock_post.side_effect = req_lib.RequestException("timeout")
            result = dispatcher.dispatch(sig)

        assert result is False

    def test_dispatch_batch_returns_per_ticker_results(self):
        from trading_bot.inference.notifier import SignalDispatcher
        with patch("trading_bot.inference.notifier.TELEGRAM_BOT_TOKEN", ""), \
             patch("trading_bot.inference.notifier.TELEGRAM_CHAT_ID", ""):
            dispatcher = SignalDispatcher(bot_token=None, chat_id=None)
            sigs = [_make_signal(), _make_signal()]
            results = dispatcher.dispatch_batch(sigs)
        assert "RELIANCE.NS" in results
        assert all(v is False for v in results.values())
