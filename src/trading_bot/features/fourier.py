"""
Fourier cycle features for the trading feature pipeline.

Captures dominant market cycles (weekly, biweekly, monthly, quarterly)
by extracting sin/cos components of the FFT of the Close price rolling window.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_fourier_features(
    df: pd.DataFrame,
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add Fourier sin/cos features derived from the Close price.

    For each period N, computes the FFT of a rolling window of length N over
    the Close column and extracts the dominant frequency's sin and cos components.

    This adds columns: fourier_sin_{N}, fourier_cos_{N} for each period.

    Args:
        df: DataFrame with at least a 'Close' column.
        periods: List of rolling window lengths to compute FFT for.
                 Defaults to [3, 6, 12, 21] (weekly, biweekly, monthly, quarterly).

    Returns:
        DataFrame with Fourier feature columns appended in-place.
    """
    if periods is None:
        periods = [3, 6, 12, 21]

    result = df.copy()
    close = result["Close"].values.astype(np.float64)
    n = len(close)

    for period in periods:
        sin_col = f"fourier_sin_{period}"
        cos_col = f"fourier_cos_{period}"
        sin_vals = np.zeros(n, dtype=np.float64)
        cos_vals = np.zeros(n, dtype=np.float64)

        for i in range(period - 1, n):
            window = close[i - period + 1 : i + 1]
            fft_vals = np.fft.fft(window)
            # Dominant frequency is at index 1 (skip DC component at index 0)
            dominant = fft_vals[1]
            sin_vals[i] = dominant.imag
            cos_vals[i] = dominant.real

        result[sin_col] = sin_vals
        result[cos_col] = cos_vals

    logger.debug("Added Fourier features for periods: %s", periods)
    return result
