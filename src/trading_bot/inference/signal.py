"""
Signal dataclass — the core data object flowing through the entire inference pipeline.

Every downstream consumer (notifier, dashboard, accuracy_tracker) works with Signal.
All timestamps are timezone-aware (Asia/Kolkata / IST).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

_REGIME_LABELS = {0: "Bearish", 1: "Sideways", 2: "Bullish"}
_REGIME_EMOJI = {0: "🐻", 1: "➡️", 2: "🐂"}
_ACTION_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Confidence(str, Enum):
    HIGH = "HIGH"     # ≥ 3/3 ensemble agreement
    MEDIUM = "MEDIUM" # 2/3 ensemble agreement
    LOW = "LOW"       # 1/3 or less — signal NOT generated


@dataclass
class Signal:
    """Complete trading signal produced by the ensemble inference pipeline."""

    # Core signal
    ticker: str
    action: Action
    confidence: Confidence
    generated_at: datetime          # timezone-aware IST

    # Price levels (calculated from ATR-14)
    current_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    risk_reward_ratio: float

    # Supporting context
    sentiment_score: float          # -1.0 to 1.0
    sentiment_label: str            # "Positive" / "Neutral" / "Negative"
    market_regime: int              # 0=bear, 1=sideways, 2=bull (from HMM)
    atr_14: float                   # ATR(14) used for level calculation

    # Ensemble votes
    votes: dict                     # {"SAC": "BUY", "PPO": "BUY", "RSI": "HOLD"}
    vote_count: int                 # number of BUY or SELL votes (not HOLD)

    # Optional metadata
    model_run_name: str = ""
    notes: str = ""

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_actionable(self) -> bool:
        """True only when confidence is HIGH or MEDIUM (≥ 2/3 votes agree)."""
        return self.confidence in (Confidence.HIGH, Confidence.MEDIUM)

    @property
    def signal_id(self) -> str:
        """Stable unique ID: sha256 of ticker + ISO timestamp (first 16 chars)."""
        raw = f"{self.ticker}:{self.generated_at.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict.

        datetime is stored as ISO string; Enum values stored as their .value strings.
        """
        raw = asdict(self)
        raw["generated_at"] = self.generated_at.isoformat()
        raw["action"] = self.action.value
        raw["confidence"] = self.confidence.value
        return raw

    @classmethod
    def from_dict(cls, data: dict) -> "Signal":
        """Reconstruct a Signal from a dict produced by to_dict()."""
        d = dict(data)
        d["generated_at"] = datetime.fromisoformat(d["generated_at"])
        d["action"] = Action(d["action"])
        d["confidence"] = Confidence(d["confidence"])
        return cls(**d)

    def to_json(self) -> str:
        """Compact JSON string (for logging / SQLite storage)."""
        return json.dumps(self.to_dict())

    # ── Telegram formatting ────────────────────────────────────────────────────

    def to_telegram_message(self) -> str:
        """
        Full-detail signal message for BUY / SELL signals.

        Price levels formatted in Indian comma style (₹2,847.50).
        Includes percentage move from current price for target + stop.
        """
        emoji = _ACTION_EMOJI.get(self.action.value, "⚪")
        ts = self.generated_at.strftime("%H:%M IST, %d %b %Y")
        regime_label = _REGIME_LABELS.get(self.market_regime, "Unknown")

        entry_mid = (self.entry_low + self.entry_high) / 2
        if entry_mid > 0:
            target_pct = (self.target - entry_mid) / entry_mid * 100
            stop_pct = (self.stop_loss - entry_mid) / entry_mid * 100
        else:
            target_pct = stop_pct = 0.0

        vote_line = " | ".join(f"{k}={v}" for k, v in self.votes.items())
        votes_total = len(self.votes)

        lines = [
            f"{emoji} {self.action.value} SIGNAL — {self.ticker}",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Confidence: {self.confidence.value} ({self.vote_count}/{votes_total} votes)",
            f"Current Price: ₹{self.current_price:,.2f}",
            "",
            f"📍 Entry Zone:  ₹{self.entry_low:,.0f} – ₹{self.entry_high:,.0f}",
            f"🛑 Stop Loss:   ₹{self.stop_loss:,.0f}  ({stop_pct:+.1f}%)",
            f"🎯 Target:      ₹{self.target:,.0f}  ({target_pct:+.1f}%)",
            f"⚖️  Risk/Reward: 1 : {self.risk_reward_ratio:.1f}",
            "",
            f"📊 Regime: {regime_label} | Sentiment: {self.sentiment_label} ({self.sentiment_score:.2f})",
            f"🗳️  Votes: {vote_line}",
            "",
            f"⏰ Generated: {ts}",
            "⚠️ Not financial advice. Paper trade first.",
        ]
        return "\n".join(lines)

    def to_hold_message(self) -> str:
        """
        Shorter message for HOLD signals or low-confidence outputs.
        Still sent so you know the pipeline ran.
        """
        ts = self.generated_at.strftime("%H:%M IST, %d %b %Y")
        regime_label = _REGIME_LABELS.get(self.market_regime, "Unknown")
        vote_line = " | ".join(f"{k}={v}" for k, v in self.votes.items())
        votes_total = len(self.votes)

        lines = [
            f"⚪ NO SIGNAL — {self.ticker}",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Confidence: {self.confidence.value} ({self.vote_count}/{votes_total} votes)",
            f"Votes: {vote_line}",
            f"Sentiment: {self.sentiment_label} ({self.sentiment_score:.2f})",
            f"Regime: {regime_label}",
            "",
            f"⏰ {ts}",
        ]
        return "\n".join(lines)
