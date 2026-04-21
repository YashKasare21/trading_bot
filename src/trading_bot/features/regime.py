"""
Hidden Markov Model market regime detector.

Identifies bull / sideways / bear regimes using log returns and realised
volatility as observation features. The fitted model is serialised to disk
so inference runs can load it without refitting.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm

from trading_bot.config import MODEL_DIR

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "regime_hmm.pkl"


class RegimeDetector:
    """
    Gaussian HMM with 3 hidden states (bull / sideways / bear).

    The mapping between integer state labels and economic regimes is
    determined empirically after fitting — state 0/1/2 do not have a
    fixed meaning until the model is inspected.
    """

    def __init__(self, n_components: int = 3) -> None:
        """
        Args:
            n_components: Number of hidden states (default: 3).
        """
        self._n_components = n_components
        self._model: hmm.GaussianHMM | None = None

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the HMM on log returns and rolling volatility.

        Args:
            df: DataFrame with at least a 'Close' column.

        Returns:
            self (for method chaining).
        """
        obs = self._build_features(df)
        self._model = hmm.GaussianHMM(
            n_components=self._n_components,
            covariance_type="diag",  # diag is more numerically stable than "full"
            n_iter=100,
            random_state=42,
        )
        self._model.fit(obs)
        logger.info(
            "HMM fitted: %d states, score=%.2f",
            self._n_components,
            self._model.score(obs),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict hidden state labels for each row of *df*.

        Args:
            df: DataFrame with at least a 'Close' column.

        Returns:
            Integer array of shape (n_rows,) with state labels 0/1/2.

        Raises:
            RuntimeError: If the model has not been fitted or loaded yet.
        """
        if self._model is None:
            raise RuntimeError("RegimeDetector must be fitted or loaded before calling predict().")
        obs = self._build_features(df)
        return self._model.predict(obs)

    def add_regime_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append a 'market_regime' integer column to *df*.

        The first observation has regime=0 since log returns require a
        prior row. The remaining rows are labelled by the HMM.

        Args:
            df: DataFrame with at least a 'Close' column.

        Returns:
            Copy of *df* with the 'market_regime' column added.
        """
        result = df.copy()
        labels = self.predict(df)
        # labels has one fewer row than df because log returns drop the first row
        regime = np.empty(len(df), dtype=np.int32)
        regime[0] = 0  # placeholder — first row has no prior return
        regime[1:] = labels
        result["market_regime"] = regime
        return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """
        Serialise the fitted model to *path*.

        Args:
            path: Full file path. Defaults to MODEL_DIR/regime_hmm.pkl.

        Returns:
            The path the model was saved to.
        """
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted RegimeDetector.")
        if path is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = MODEL_DIR / _MODEL_FILENAME
        joblib.dump(self._model, path, compress=3)
        logger.info("RegimeDetector saved to %s", path)
        return path

    def load(self, path: Path | None = None) -> "RegimeDetector":
        """
        Load a previously saved HMM from *path*.

        Args:
            path: Full file path. Defaults to MODEL_DIR/regime_hmm.pkl.

        Returns:
            self (for method chaining).
        """
        if path is None:
            path = MODEL_DIR / _MODEL_FILENAME
        self._model = joblib.load(path)
        logger.info("RegimeDetector loaded from %s", path)
        return self

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        """
        Build the (n_obs, 2) feature matrix for the HMM.

        Features:
            - log_return: log(close[t] / close[t-1])
            - volatility: abs(log_return) as a proxy for realised vol

        The first row is dropped because log returns require a prior value.
        """
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
        vol = np.abs(log_ret)
        return np.column_stack([log_ret, vol])
