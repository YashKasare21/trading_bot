"""
Tests for DataStore (SQLite-backed price and sentiment cache).

All tests use a temporary in-memory/on-disk DB — no shared state.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_store():
    from trading_bot.data.store import DataStore

    tmpdir = tempfile.mkdtemp()
    return DataStore(db_path=Path(tmpdir) / "test.db")


def _make_ohlcv(n: int = 10, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed=99)
    close = 1000.0 + np.cumsum(rng.normal(0, 5, size=n))
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, size=n).astype(float),
        },
        index=dates,
    )


# ── Price data ────────────────────────────────────────────────────────────────


def test_save_and_load_price_data_roundtrip():
    """Rows saved to the store must be retrievable with matching Close values."""
    store = _make_store()
    df = _make_ohlcv(n=10)
    ticker = "RELIANCE.NS"

    store.save_price_data(df, ticker)

    start = df.index[0].date()
    end = df.index[-1].date()
    loaded = store.load_price_data(ticker, start, end)

    assert len(loaded) == len(df)
    np.testing.assert_allclose(loaded["Close"].values, df["Close"].values, rtol=1e-6)


def test_load_price_data_empty_range():
    """Loading a date range with no data must return an empty DataFrame."""
    store = _make_store()
    result = store.load_price_data("NOTEXIST.NS", date(2020, 1, 1), date(2020, 1, 31))
    assert result.empty
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_save_price_data_no_duplicates():
    """Saving the same data twice must not create duplicate rows."""
    store = _make_store()
    df = _make_ohlcv(n=5)
    ticker = "TCS.NS"

    store.save_price_data(df, ticker)
    store.save_price_data(df, ticker)  # second save — should be a no-op

    start = df.index[0].date()
    end = df.index[-1].date()
    loaded = store.load_price_data(ticker, start, end)
    assert len(loaded) == len(df)


def test_load_price_data_sorted_by_date():
    """Loaded DataFrame must be sorted ascending by date."""
    store = _make_store()
    df = _make_ohlcv(n=20)
    ticker = "INFY.NS"
    store.save_price_data(df, ticker)

    start = df.index[0].date()
    end = df.index[-1].date()
    loaded = store.load_price_data(ticker, start, end)
    assert loaded.index.is_monotonic_increasing


def test_get_cached_dates_returns_correct_dates():
    """get_cached_dates must return the exact dates saved."""
    store = _make_store()
    df = _make_ohlcv(n=5)
    ticker = "WIPRO.NS"
    store.save_price_data(df, ticker)

    cached = store.get_cached_dates(ticker)
    expected = [ts.date() for ts in df.index]
    assert sorted(cached) == sorted(expected)


def test_get_cached_dates_empty_for_unknown_ticker():
    """Unknown ticker must return an empty list from get_cached_dates."""
    store = _make_store()
    assert store.get_cached_dates("UNKNOWN.NS") == []


# ── Sentiment cache ───────────────────────────────────────────────────────────


def test_save_and_load_sentiment_roundtrip():
    """A saved sentiment entry must be retrievable with correct fields."""
    store = _make_store()
    ticker = "RELIANCE.NS"
    entry_date = date(2024, 3, 15)
    headline = "Reliance posts record quarterly profit"

    store.save_sentiment(ticker, entry_date, headline, "Positive", 1.0, "Positive")

    results = store.load_sentiment(ticker, entry_date)
    assert len(results) == 1
    assert results[0]["sentiment_label"] == "Positive"
    assert results[0]["sentiment_score"] == pytest.approx(1.0)


def test_save_sentiment_deduplicates_by_hash():
    """Saving the same headline twice must not create duplicate cache entries."""
    store = _make_store()
    ticker = "TCS.NS"
    entry_date = date(2024, 3, 15)
    headline = "TCS wins $500M deal"

    store.save_sentiment(ticker, entry_date, headline, "Positive", 1.0)
    store.save_sentiment(ticker, entry_date, headline, "Positive", 1.0)

    results = store.load_sentiment(ticker, entry_date)
    assert len(results) == 1


def test_load_sentiment_returns_empty_for_unknown():
    """No cache entries for unknown ticker/date must return empty list."""
    store = _make_store()
    results = store.load_sentiment("UNKNOWN.NS", date(2024, 1, 1))
    assert results == []


def test_load_sentiment_multiple_headlines():
    """Multiple headlines on the same date must all be returned."""
    store = _make_store()
    ticker = "HDFCBANK.NS"
    entry_date = date(2024, 4, 10)
    headlines = [
        ("HDFC reports strong Q4", "Positive", 1.0),
        ("Banking sector under pressure", "Negative", -1.0),
        ("Market holds steady", "Neutral", 0.0),
    ]
    for h, label, score in headlines:
        store.save_sentiment(ticker, entry_date, h, label, score)

    results = store.load_sentiment(ticker, entry_date)
    assert len(results) == 3
    scores = {r["sentiment_score"] for r in results}
    assert scores == {1.0, -1.0, 0.0}
