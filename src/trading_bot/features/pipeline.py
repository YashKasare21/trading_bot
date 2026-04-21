"""
FeaturePipeline — single source of truth for all feature engineering.

Both Colab training and local inference call this class. Feature logic
must NEVER be duplicated outside this module.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from trading_bot.features.fourier import add_fourier_features
from trading_bot.features.regime import RegimeDetector
from trading_bot.features.technical import add_technical_indicators

logger = logging.getLogger(__name__)

_LAGGED_RETURN_DAYS = [1, 3, 5, 10, 20]


class FeaturePipeline:
    """
    Orchestrates the full feature engineering pipeline.

    Steps (in order):
        1. Technical indicators (ta library, 80+ features)
        2. Lagged log returns (1, 3, 5, 10, 20 days)
        3. Fourier cycle features
        4. HMM market regime
        5. Sentiment score merge (optional)
        6. Warm-up row removal + NaN cleanup
    """

    def __init__(
        self,
        window_size: int = 20,
        fourier_periods: list[int] | None = None,
        include_sentiment: bool = True,
        fit_regime: bool = False,
    ) -> None:
        """
        Args:
            window_size: Look-back window used by the RL environment.
                         At least this many warm-up rows are dropped.
            fourier_periods: Periods (days) for Fourier features.
                             Defaults to [3, 6, 12, 21].
            include_sentiment: Whether to merge the sentiment_score column.
            fit_regime: If True, fits a new HMM during fit_transform().
                        If False (inference), loads a previously saved model.
        """
        self.window_size = window_size
        self.fourier_periods: list[int] = fourier_periods or [3, 6, 12, 21]
        self.include_sentiment = include_sentiment
        self.fit_regime = fit_regime

        self._regime_detector = RegimeDetector()
        self._feature_names: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Apply the full pipeline, optionally fitting the HMM regime detector.

        Use during training (Colab). Fits the regime model when
        ``fit_regime=True``.

        Args:
            df: Clean OHLCV DataFrame (DatetimeIndex, float64 columns).
            sentiment_df: Optional DataFrame with a 'sentiment_score' column
                          and a DatetimeIndex. Merged by date.

        Returns:
            Feature-engineered DataFrame ready for the RL environment.
        """
        result = self._apply_pipeline(df, sentiment_df, fit=self.fit_regime)
        self._feature_names = list(result.columns)
        return result

    def transform(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Apply the pipeline without refitting any models.

        Use during inference (daily signal generation).

        Args:
            df: Clean OHLCV DataFrame.
            sentiment_df: Optional sentiment DataFrame.

        Returns:
            Feature-engineered DataFrame.
        """
        return self._apply_pipeline(df, sentiment_df, fit=False)

    def get_feature_names(self) -> list[str]:
        """
        Return the ordered list of feature column names.

        CRITICAL: Save this list alongside every trained RL model.
        Inference must use identical features in identical order.

        Returns:
            List of column name strings. Empty if fit_transform has not been called.
        """
        return list(self._feature_names)

    def save(self, path: Path) -> None:
        """
        Serialise the pipeline config, feature names, and regime model to *path*.

        Uses joblib compression for efficient storage of the HMM model.

        Args:
            path: Destination file path (e.g. 'data/models/pipeline_config.joblib').
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "window_size": self.window_size,
            "fourier_periods": self.fourier_periods,
            "include_sentiment": self.include_sentiment,
            "feature_names": self._feature_names,
            "regime_model": self._regime_detector._model,
        }
        joblib.dump(state, path, compress=3)
        logger.info("FeaturePipeline saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeaturePipeline":
        """
        Load a previously saved pipeline from *path*.

        Args:
            path: Path to a file saved by :meth:`save`.

        Returns:
            Restored FeaturePipeline instance.
        """
        state = joblib.load(path)
        pipeline = cls(
            window_size=state["window_size"],
            fourier_periods=state["fourier_periods"],
            include_sentiment=state["include_sentiment"],
            fit_regime=False,
        )
        pipeline._feature_names = state["feature_names"]
        pipeline._regime_detector._model = state["regime_model"]
        logger.info("FeaturePipeline loaded from %s", path)
        return pipeline

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_pipeline(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None,
        fit: bool,
    ) -> pd.DataFrame:
        """Run all pipeline steps in order."""
        # Step 1: Technical indicators
        result = add_technical_indicators(df)

        # Step 2: Lagged log returns
        for lag in _LAGGED_RETURN_DAYS:
            result[f"lag_{lag}d"] = np.log(
                result["Close"] / result["Close"].shift(lag)
            )

        # Step 3: Fourier features
        result = add_fourier_features(result, self.fourier_periods)

        # Step 4: HMM market regime
        if fit:
            self._regime_detector.fit(result)
            self._regime_detector.save()
        result = self._regime_detector.add_regime_feature(result)

        # Step 5: Sentiment merge
        if self.include_sentiment and sentiment_df is not None:
            result = result.join(
                sentiment_df[["sentiment_score"]],
                how="left",
            )
            result["sentiment_score"] = result["sentiment_score"].fillna(0.0)
        elif self.include_sentiment:
            result["sentiment_score"] = 0.0

        # Step 6: Drop warm-up rows and clean NaNs
        warmup = max(self.window_size, 30)
        result = result.iloc[warmup:]
        result = result.ffill()
        before = len(result)
        result = result.dropna()
        dropped = before - len(result)
        if dropped:
            logger.debug("Dropped %d residual NaN rows after pipeline", dropped)

        logger.info(
            "Pipeline complete: %d rows x %d features",
            len(result),
            result.shape[1],
        )
        return result
