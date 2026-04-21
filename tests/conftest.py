"""
Shared pytest fixtures for the trading bot test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    200 rows of realistic OHLCV data with a DatetimeIndex.

    Uses a seeded random walk on Close so results are reproducible.
    """
    rng = np.random.default_rng(seed=42)
    n = 200

    # Geometric random walk for close prices around 1000 (INR scale)
    log_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = 1000.0 * np.exp(np.cumsum(log_returns))

    # Construct plausible OHLCV from Close
    noise = rng.uniform(0.995, 1.005, size=(n, 2))
    high = close * np.maximum(noise[:, 0], noise[:, 1])
    low = close * np.minimum(noise[:, 0], noise[:, 1])
    open_ = close * rng.uniform(0.997, 1.003, size=n)
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)

    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_featured_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature set: original OHLCV + lagged log returns.

    Deliberately lightweight — does NOT run the full FeaturePipeline so
    tests remain fast and free of heavy optional dependencies.
    """
    df = sample_ohlcv_df.copy()
    for lag in [1, 3, 5, 10, 20]:
        df[f"lag_{lag}d"] = np.log(df["Close"] / df["Close"].shift(lag))
    # Drop warm-up rows
    df = df.iloc[30:].copy()
    df = df.ffill().dropna()
    return df.reset_index(drop=True)


@pytest.fixture
def trading_env(sample_featured_df: pd.DataFrame):
    """StockTradingEnv initialised with sample_featured_df."""
    from trading_bot.env.trading_env import StockTradingEnv

    return StockTradingEnv(sample_featured_df, window_size=20, initial_balance=100_000.0)
