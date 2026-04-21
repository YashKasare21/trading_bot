"""
Technical indicator wrapper around the `ta` library.

Uses forward-fill only (never backward-fill) to avoid look-ahead bias.
"""

import logging

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 80+ technical analysis indicators from the `ta` library.

    Look-ahead safety:
        - ta.add_all_ta_features is called with fillna=False so initial NaN
          rows are NOT silently filled with future data.
        - A single forward-fill pass is applied afterwards to propagate the
          last valid value forward (safe — uses only past data).
        - Rows still containing NaN after ffill are dropped.

    Args:
        df: Clean OHLCV DataFrame with columns [Open, High, Low, Close, Volume].

    Returns:
        Enriched DataFrame with all TA columns appended.
        Input columns [Open, High, Low, Close, Volume] are preserved.
    """
    enriched = ta.add_all_ta_features(
        df.copy(),
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=False,  # NEVER True — causes look-ahead bias
    )

    # Forward-fill only — propagates last known value (safe)
    enriched = enriched.ffill()

    before = len(enriched)
    enriched = enriched.dropna()
    dropped = before - len(enriched)
    if dropped:
        logger.debug("Dropped %d rows with NaN after TA ffill", dropped)

    logger.info(
        "Technical indicators added: %d features, %d rows remaining",
        enriched.shape[1],
        len(enriched),
    )
    return enriched
