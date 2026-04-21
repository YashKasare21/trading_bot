"""
Tests for MarketDataFetcher.

yfinance.download is mocked throughout — no real network calls.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_store():
    from trading_bot.data.store import DataStore

    tmpdir = tempfile.mkdtemp()
    return DataStore(db_path=Path(tmpdir) / "test.db")


def _synthetic_yf_response(n: int = 20, start: str = "2023-01-02") -> pd.DataFrame:
    """Mimic a yfinance.download() return: flat columns, DatetimeIndex."""
    rng = np.random.default_rng(seed=77)
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


def _make_fetcher(store=None):
    from trading_bot.data.fetcher import MarketDataFetcher

    return MarketDataFetcher(store=store or _make_store())


# ── _clean() ──────────────────────────────────────────────────────────────────


def test_clean_flat_columns():
    """_clean must handle flat columns and return required OHLCV columns."""
    from trading_bot.data.fetcher import MarketDataFetcher

    raw = _synthetic_yf_response()
    result = MarketDataFetcher._clean(raw, "TEST.NS")
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result.dtypes["Close"] == np.float64


def test_clean_multiindex_columns():
    """_clean must flatten a MultiIndex (yfinance quirk for single-ticker calls)."""
    from trading_bot.data.fetcher import MarketDataFetcher

    raw = _synthetic_yf_response()
    # Wrap in MultiIndex
    raw.columns = pd.MultiIndex.from_tuples(
        [(c, "TEST.NS") for c in raw.columns]
    )
    result = MarketDataFetcher._clean(raw, "TEST.NS")
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_clean_drops_nan_rows():
    """_clean must drop rows that have any NaN."""
    from trading_bot.data.fetcher import MarketDataFetcher

    raw = _synthetic_yf_response(n=10)
    raw.iloc[3, raw.columns.get_loc("Close")] = float("nan")
    result = MarketDataFetcher._clean(raw, "TEST.NS")
    assert len(result) == 9


def test_clean_missing_required_column_returns_empty():
    """If a required column is absent after cleaning, return an empty DataFrame."""
    from trading_bot.data.fetcher import MarketDataFetcher

    raw = _synthetic_yf_response().drop(columns=["Volume"])
    result = MarketDataFetcher._clean(raw, "TEST.NS")
    assert result.empty


# ── fetch_ohlcv() — cache miss path ───────────────────────────────────────────


def test_fetch_ohlcv_downloads_on_cache_miss():
    """When no cached data exists, yfinance.download must be called."""
    store = _make_store()
    fetcher = _make_fetcher(store)
    raw = _synthetic_yf_response(n=20)

    with patch("yfinance.download", return_value=raw) as mock_dl:
        result = fetcher.fetch_ohlcv("RELIANCE.NS", date(2023, 1, 2), date(2023, 1, 31))
        mock_dl.assert_called_once()

    assert not result.empty
    assert "Close" in result.columns


def test_fetch_ohlcv_uses_cache_on_second_call():
    """Second call for the same range must NOT call yfinance.download again."""
    store = _make_store()
    fetcher = _make_fetcher(store)
    raw = _synthetic_yf_response(n=20)
    # Use the exact date range covered by the mock data to avoid partial-miss re-fetch
    fetch_start = raw.index[0].date()
    fetch_end = raw.index[-1].date()

    with patch("yfinance.download", return_value=raw):
        fetcher.fetch_ohlcv("TCS.NS", fetch_start, fetch_end)

    with patch("yfinance.download") as mock_dl:
        result = fetcher.fetch_ohlcv("TCS.NS", fetch_start, fetch_end)
        mock_dl.assert_not_called()

    assert not result.empty


def test_fetch_ohlcv_empty_yfinance_returns_empty_df():
    """If yfinance returns an empty DataFrame, fetch_ohlcv must return empty."""
    fetcher = _make_fetcher()

    with patch("yfinance.download", return_value=pd.DataFrame()):
        result = fetcher.fetch_ohlcv("GHOST.NS", date(2023, 1, 2), date(2023, 1, 10))

    assert result.empty


def test_fetch_ohlcv_use_cache_false_always_downloads():
    """use_cache=False must bypass the cache and call yfinance.download."""
    fetcher = _make_fetcher()
    raw = _synthetic_yf_response(n=10)

    with patch("yfinance.download", return_value=raw) as mock_dl:
        fetcher.fetch_ohlcv("INFY.NS", date(2023, 1, 2), date(2023, 1, 20), use_cache=False)
        fetcher.fetch_ohlcv("INFY.NS", date(2023, 1, 2), date(2023, 1, 20), use_cache=False)
        assert mock_dl.call_count == 2


# ── fetch_batch() ─────────────────────────────────────────────────────────────


def test_fetch_batch_returns_dict_of_dataframes():
    """fetch_batch must return a dict mapping each ticker to its DataFrame."""
    fetcher = _make_fetcher()
    raw = _synthetic_yf_response(n=20)

    with patch("yfinance.download", return_value=raw):
        result = fetcher.fetch_batch(
            ["RELIANCE.NS", "TCS.NS"],
            date(2023, 1, 2),
            date(2023, 1, 31),
        )

    assert isinstance(result, dict)
    assert "RELIANCE.NS" in result
    assert "TCS.NS" in result
    assert not result["RELIANCE.NS"].empty


def test_fetch_batch_omits_empty_tickers():
    """Tickers for which yfinance returns empty data must be excluded from result."""
    fetcher = _make_fetcher()

    def _side_effect(ticker, **kwargs):
        if "GHOST" in ticker:
            return pd.DataFrame()
        return _synthetic_yf_response(n=20)

    with patch("yfinance.download", side_effect=_side_effect):
        result = fetcher.fetch_batch(
            ["RELIANCE.NS", "GHOST.NS"],
            date(2023, 1, 2),
            date(2023, 1, 31),
        )

    assert "RELIANCE.NS" in result
    assert "GHOST.NS" not in result
