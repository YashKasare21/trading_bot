"""
SQLAlchemy-backed data store.

Resolves the database URL in this priority order:
1. DATABASE_URL environment variable (Supabase / any PostgreSQL)
2. db_path constructor argument (SQLite file path)
3. Default SQLite at CACHE_DIR/trading_bot.db

The postgres:// → postgresql:// rewrite handles Supabase connection strings
which use the legacy scheme that SQLAlchemy 1.4+ no longer accepts.
"""

import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from trading_bot.config import CACHE_DIR

load_dotenv()

if TYPE_CHECKING:
    from trading_bot.inference.signal import Signal

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class PriceData(_Base):
    """Stores daily OHLCV bars for a single ticker."""

    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class SentimentCache(_Base):
    """Caches Gemini sentiment scores keyed by ticker + date + headline hash."""

    __tablename__ = "sentiment_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    headline_hash = Column(String(64), nullable=False)
    sentiment_label = Column(String(16), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    raw_response = Column(String(512), nullable=True)


class SignalRecord(_Base):
    """Persists every inference signal for accuracy tracking and the dashboard."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(16), nullable=False, unique=True, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    action = Column(String(8), nullable=False)
    confidence = Column(String(8), nullable=False)
    generated_at = Column(String(32), nullable=False, index=True)
    current_price = Column(Float, nullable=False)
    entry_low = Column(Float, nullable=False)
    entry_high = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    risk_reward_ratio = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String(16), nullable=False)
    market_regime = Column(Integer, nullable=False)
    atr_14 = Column(Float, nullable=False)
    votes = Column(Text, nullable=False)   # JSON-encoded dict
    vote_count = Column(Integer, nullable=False)
    model_run_name = Column(String(64), nullable=False, default="")
    notes = Column(Text, nullable=False, default="")


class DataStore:
    """Database-backed data store (PostgreSQL via DATABASE_URL or SQLite fallback)."""

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialise the data store.

        Resolution order for the database URL:
        1. DATABASE_URL env var (Supabase / PostgreSQL) — takes priority over db_path.
        2. db_path argument — explicit SQLite file path.
        3. Default SQLite at CACHE_DIR/trading_bot.db.

        Args:
            db_path: SQLite file path used only when DATABASE_URL is not set.
        """
        db_url = os.getenv("DATABASE_URL", "")

        if db_url:
            # Supabase (and some other hosts) emit postgres:// which SQLAlchemy
            # 1.4+ rejects — rewrite to the canonical postgresql:// scheme.
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            self._engine = create_engine(db_url, echo=False)
            logger.info("DataStore connected to PostgreSQL via DATABASE_URL.")
        else:
            if db_path is None:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                db_path = CACHE_DIR / "trading_bot.db"
            self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
            logger.debug("DataStore initialised at SQLite: %s", db_path)

        _Base.metadata.create_all(self._engine)

    # ── Price Data ─────────────────────────────────────────────────────────────

    def save_price_data(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Upsert OHLCV rows for *ticker* from a DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
            ticker: NSE ticker symbol.
        """
        with Session(self._engine) as session:
            for ts, row in df.iterrows():
                bar_date = ts.date() if hasattr(ts, "date") else ts
                existing = session.execute(
                    select(PriceData).where(
                        PriceData.ticker == ticker,
                        PriceData.date == bar_date,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        PriceData(
                            date=bar_date,
                            ticker=ticker,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=float(row["Volume"]),
                        )
                    )
            session.commit()
        logger.info("Saved %d rows of price data for %s", len(df), ticker)

    def load_price_data(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """
        Load OHLCV rows for *ticker* between *start* and *end* (inclusive).

        Returns:
            DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
            Empty DataFrame if no data exists for the range.
        """
        with Session(self._engine) as session:
            rows = session.execute(
                select(PriceData).where(
                    PriceData.ticker == ticker,
                    PriceData.date >= start,
                    PriceData.date <= end,
                )
            ).scalars().all()

        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        records = [
            {
                "Date": r.date,
                "Open": r.open,
                "High": r.high,
                "Low": r.low,
                "Close": r.close,
                "Volume": r.volume,
            }
            for r in rows
        ]
        df = pd.DataFrame(records).set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_cached_dates(self, ticker: str) -> list[date]:
        """
        Return all dates that already have cached price data for *ticker*.

        Used to skip re-fetching already-stored data.
        """
        with Session(self._engine) as session:
            rows = session.execute(
                select(PriceData.date).where(PriceData.ticker == ticker)
            ).scalars().all()
        return sorted(rows)

    # ── Sentiment Cache ────────────────────────────────────────────────────────

    def save_sentiment(
        self,
        ticker: str,
        entry_date: date,
        headline: str,
        label: str,
        score: float,
        raw_response: str = "",
    ) -> None:
        """
        Persist a single sentiment result. Skips if the headline hash already exists.

        Args:
            ticker: NSE ticker symbol.
            entry_date: Publication date of the headline.
            headline: Raw headline text (will be hashed for deduplication).
            label: Sentiment label ('Positive', 'Neutral', 'Negative').
            score: Numeric score (1.0, 0.0, -1.0).
            raw_response: Raw text returned by the Gemini API.
        """
        headline_hash = hashlib.sha256(headline.encode()).hexdigest()
        with Session(self._engine) as session:
            existing = session.execute(
                select(SentimentCache).where(
                    SentimentCache.ticker == ticker,
                    SentimentCache.date == entry_date,
                    SentimentCache.headline_hash == headline_hash,
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    SentimentCache(
                        date=entry_date,
                        ticker=ticker,
                        headline_hash=headline_hash,
                        sentiment_label=label,
                        sentiment_score=score,
                        raw_response=raw_response[:512],
                    )
                )
                session.commit()

    def load_sentiment(self, ticker: str, entry_date: date) -> list[dict]:
        """
        Load all cached sentiment entries for *ticker* on *entry_date*.

        Returns:
            List of dicts with keys: headline_hash, sentiment_label, sentiment_score.
        """
        with Session(self._engine) as session:
            rows = session.execute(
                select(SentimentCache).where(
                    SentimentCache.ticker == ticker,
                    SentimentCache.date == entry_date,
                )
            ).scalars().all()

        return [
            {
                "headline_hash": r.headline_hash,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score,
            }
            for r in rows
        ]

    # ── Signal Store ───────────────────────────────────────────────────────────

    def save_signal(self, signal: "Signal") -> None:
        """
        Persist an inference Signal.  Skips if the signal_id already exists
        (idempotent — safe to call multiple times for the same signal).

        Args:
            signal: Signal dataclass from the inference pipeline.
        """
        d = signal.to_dict()
        with Session(self._engine) as session:
            existing = session.execute(
                select(SignalRecord).where(SignalRecord.signal_id == signal.signal_id)
            ).scalar_one_or_none()
            if existing is not None:
                return
            session.add(
                SignalRecord(
                    signal_id=signal.signal_id,
                    ticker=d["ticker"],
                    action=d["action"],
                    confidence=d["confidence"],
                    generated_at=d["generated_at"],
                    current_price=d["current_price"],
                    entry_low=d["entry_low"],
                    entry_high=d["entry_high"],
                    stop_loss=d["stop_loss"],
                    target=d["target"],
                    risk_reward_ratio=d["risk_reward_ratio"],
                    sentiment_score=d["sentiment_score"],
                    sentiment_label=d["sentiment_label"],
                    market_regime=d["market_regime"],
                    atr_14=d["atr_14"],
                    votes=json.dumps(d["votes"]),
                    vote_count=d["vote_count"],
                    model_run_name=d.get("model_run_name", ""),
                    notes=d.get("notes", ""),
                )
            )
            session.commit()
        logger.debug("Saved signal %s for %s", signal.signal_id, signal.ticker)

    def load_signals(
        self,
        ticker: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Load recent signals, optionally filtered by ticker.

        Args:
            ticker: If provided, filter to this ticker only.
            limit: Maximum rows returned, newest first.

        Returns:
            DataFrame with one row per signal.  Empty if no data.
        """
        with Session(self._engine) as session:
            stmt = select(SignalRecord).order_by(desc(SignalRecord.generated_at)).limit(limit)
            if ticker:
                stmt = stmt.where(SignalRecord.ticker == ticker)
            rows = session.execute(stmt).scalars().all()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            rec = {
                "signal_id": r.signal_id,
                "ticker": r.ticker,
                "action": r.action,
                "confidence": r.confidence,
                "generated_at": r.generated_at,
                "current_price": r.current_price,
                "entry_low": r.entry_low,
                "entry_high": r.entry_high,
                "stop_loss": r.stop_loss,
                "target": r.target,
                "risk_reward_ratio": r.risk_reward_ratio,
                "sentiment_score": r.sentiment_score,
                "sentiment_label": r.sentiment_label,
                "market_regime": r.market_regime,
                "atr_14": r.atr_14,
                "votes": json.loads(r.votes) if r.votes else {},
                "vote_count": r.vote_count,
            }
            records.append(rec)

        return pd.DataFrame(records)
