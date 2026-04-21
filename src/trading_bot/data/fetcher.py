"""
NSE-aware OHLCV data fetcher using yfinance.

Handles yfinance MultiIndex quirks, caches data in DataStore, and is
IST timezone-aware for end-of-day fetches.
"""

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from trading_bot.config import TIMEZONE
from trading_bot.data.store import DataStore

logger = logging.getLogger(__name__)

_IST = ZoneInfo(TIMEZONE)
_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


class MarketDataFetcher:
    """Fetches and caches OHLCV data from Yahoo Finance for NSE tickers."""

    def __init__(self, store: DataStore | None = None) -> None:
        """
        Args:
            store: DataStore instance. Creates a default one if not provided.
        """
        self._store = store or DataStore()

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for *ticker* between *start* and *end* (inclusive).

        Reads from the local SQLite cache when available and only downloads
        missing date ranges from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker (e.g. 'RELIANCE.NS', '^NSEI').
            start: First date to include.
            end: Last date to include.
            use_cache: If True, serve from cache and only fetch new data.

        Returns:
            DataFrame with DatetimeIndex and float64 columns [Open, High, Low, Close, Volume].
            Rows with any NaN are dropped.
        """
        if use_cache:
            cached = self._store.load_price_data(ticker, start, end)
            cached_dates = set(d for d in self._store.get_cached_dates(ticker))

            # Build set of trading-day dates already in cache for this range
            expected = pd.date_range(start=start, end=end, freq="B")
            missing = [d.date() for d in expected if d.date() not in cached_dates]

            if not missing:
                logger.info("Serving %s from cache (%d rows)", ticker, len(cached))
                return cached

            fetch_start = missing[0]
            fetch_end = missing[-1]
            logger.info(
                "Cache miss for %s: fetching %s → %s from Yahoo Finance",
                ticker, fetch_start, fetch_end,
            )
        else:
            fetch_start, fetch_end = start, end

        raw = yf.download(
            ticker,
            start=str(fetch_start),
            end=str(fetch_end),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return pd.DataFrame(columns=_REQUIRED_COLS)

        df = self._clean(raw, ticker)

        if use_cache and not df.empty:
            self._store.save_price_data(df, ticker)
            # Reload full range from cache to include previously-cached rows
            return self._store.load_price_data(ticker, start, end)

        return df

    def fetch_latest_eod(self, ticker: str) -> pd.DataFrame:
        """
        Fetch today's end-of-day bar for *ticker*.

        Intended to be called post 15:30 IST. Bypasses the cache so
        today's data is always fresh.

        Returns:
            Single-row DataFrame with today's OHLCV.
        """
        today = datetime.now(tz=_IST).date()
        raw = yf.download(
            ticker,
            start=str(today),
            end=str(today),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logger.warning("No EOD data available for %s on %s", ticker, today)
            return pd.DataFrame(columns=_REQUIRED_COLS)
        return self._clean(raw, ticker)

    def fetch_batch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Args:
            tickers: List of Yahoo Finance ticker symbols.
            start: First date.
            end: Last date.

        Returns:
            Mapping of ticker → DataFrame. Tickers with no data are omitted.
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start, end)
            if not df.empty:
                result[ticker] = df
        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Flatten MultiIndex columns, coerce to float64, drop NaNs."""
        # Flatten MultiIndex (yfinance quirk with multiple tickers or some versions)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise column capitalisation
        raw.columns = [str(c).capitalize() for c in raw.columns]

        missing = [c for c in _REQUIRED_COLS if c not in raw.columns]
        if missing:
            logger.error(
                "Columns missing for %s after cleaning: %s. Available: %s",
                ticker, missing, raw.columns.tolist(),
            )
            return pd.DataFrame(columns=_REQUIRED_COLS)

        df = raw[_REQUIRED_COLS].copy()

        # Convert each column individually (avoids look-ahead bias from bulk to_numeric)
        for col in _REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        logger.debug("Cleaned %s: %d rows", ticker, len(df))
        return df
