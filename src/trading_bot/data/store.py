"""
SQLite-backed data store using SQLAlchemy.

Persists OHLCV price data and Gemini sentiment cache so we avoid
redundant API calls across runs.
"""

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from trading_bot.config import CACHE_DIR

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


class DataStore:
    """Thin wrapper around an SQLite database for price and sentiment data."""

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialise the data store.

        Args:
            db_path: Path to the SQLite file. Defaults to CACHE_DIR/trading_bot.db.
        """
        if db_path is None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            db_path = CACHE_DIR / "trading_bot.db"
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        _Base.metadata.create_all(self._engine)
        logger.debug("DataStore initialised at %s", db_path)

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
