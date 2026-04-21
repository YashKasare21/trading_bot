"""
EnsemblePredictor — core inference engine.

Loads production SAC + PPO models from ModelRegistry, runs both alongside a
rule-based RSI strategy, and generates a Signal when ≥ 2 of 3 votes agree.

Heavy dependencies (torch, stable_baselines3) are lazy-imported inside
methods so this module stays importable without the [training] extras.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_bot.config import MODEL_DIR, TIMEZONE, WINDOW_SIZE
from trading_bot.inference.risk import RiskManager
from trading_bot.inference.signal import Action, Confidence, Signal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_IST = ZoneInfo(TIMEZONE)

# Thresholds for mapping continuous RL action → discrete vote
_BUY_THRESHOLD = 0.1
_SELL_THRESHOLD = -0.1


class EnsemblePredictor:
    """
    Runs SAC + PPO + RSI ensemble to produce a daily trading Signal.

    Models are loaded lazily on the first call to predict() to keep
    import time short and avoid errors on machines without sb3/torch.
    """

    def __init__(
        self,
        registry_path: Path,
        model_dir: Path | None = None,
        pipeline_path: Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        registry_path:
            Path to the ``registry.json`` produced by ModelRegistry.
        model_dir:
            Directory containing model zip files.  Defaults to MODEL_DIR from
            config if not provided.
        pipeline_path:
            Path to a saved FeaturePipeline joblib.  When None the path is
            read from the production model's registry record.
        """
        from trading_bot.models.registry import ModelRegistry

        self._registry = ModelRegistry(registry_path)
        self._model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._pipeline_path = Path(pipeline_path) if pipeline_path else None

        # Lazy-loaded attributes — set on first call to _load_models()
        self._sac_model = None
        self._ppo_model = None
        self._pipeline = None
        self._models_loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        ticker: str,
        as_of_date: date | None = None,
        lookback_days: int = 365,
    ) -> Signal:
        """
        Full prediction pipeline for a single ticker.

        Parameters
        ----------
        ticker:
            NSE ticker symbol (e.g. ``"RELIANCE.NS"``).
        as_of_date:
            Evaluation date.  Defaults to today (IST).
        lookback_days:
            Calendar days of OHLCV history to fetch for TA warmup.

        Returns
        -------
        Signal
            Complete signal with action, confidence, price levels, and context.
        """
        from trading_bot.data.fetcher import MarketDataFetcher
        from trading_bot.data.sentiment import SentimentAnalyzer

        if as_of_date is None:
            as_of_date = datetime.now(_IST).date()

        start_date = pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)
        start = start_date.date()

        # 1. Fetch OHLCV
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_ohlcv(ticker, start, as_of_date)
        if df.empty:
            logger.warning("No OHLCV data for %s — returning HOLD signal.", ticker)
            return self._hold_signal(ticker, {}, 0.0, "Neutral", 0)

        # 2. Fetch sentiment
        sentiment_df: pd.DataFrame | None = None
        try:
            analyzer = SentimentAnalyzer()
            sentiment_df = analyzer.fetch_and_score_news(ticker, start, as_of_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment fetch failed for %s: %s — using zero.", ticker, exc)

        # 3. Feature pipeline — transform only, no fit
        pipeline = self._get_pipeline()
        featured_df = pipeline.transform(df, sentiment_df)
        if featured_df.empty or len(featured_df) < WINDOW_SIZE:
            logger.warning(
                "Insufficient featured rows for %s (%d < %d).",
                ticker, len(featured_df), WINDOW_SIZE,
            )
            return self._hold_signal(ticker, {}, 0.0, "Neutral", 0)

        # 4. Ensemble votes
        sac_vote = self._vote_rl("SAC", featured_df)
        ppo_vote = self._vote_rl("PPO", featured_df)
        rsi_vote = self._predict_rsi(featured_df)
        votes = {"SAC": sac_vote.value, "PPO": ppo_vote.value, "RSI": rsi_vote.value}
        logger.info("Votes for %s: %s", ticker, votes)

        # 5. Aggregate votes
        action, confidence, vote_count = self._aggregate_votes(
            [sac_vote, ppo_vote, rsi_vote]
        )

        # 6. Price levels from ATR-14
        current_price = float(df["Close"].iloc[-1])
        atr = self._compute_atr(df, window=14)
        risk_mgr = RiskManager()
        risk_levels = risk_mgr.calculate(current_price, atr, action)
        levels = {
            "entry_low": risk_levels.entry_low,
            "entry_high": risk_levels.entry_high,
            "stop_loss": risk_levels.stop_loss,
            "target": risk_levels.target_price,
            "risk_reward_ratio": risk_levels.risk_reward_ratio,
        }

        # 7. Context
        sentiment_score = 0.0
        sentiment_label = "Neutral"
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_score = float(sentiment_df["sentiment_score"].iloc[-1])
            sentiment_label = (
                "Positive" if sentiment_score > 0.1
                else "Negative" if sentiment_score < -0.1
                else "Neutral"
            )

        regime = 1  # default sideways
        if "regime" in featured_df.columns:
            regime = int(featured_df["regime"].iloc[-1])

        signal = Signal(
            ticker=ticker,
            action=action,
            confidence=confidence,
            generated_at=datetime.now(_IST),
            current_price=current_price,
            entry_low=levels["entry_low"],
            entry_high=levels["entry_high"],
            stop_loss=levels["stop_loss"],
            target=levels["target"],
            risk_reward_ratio=levels["risk_reward_ratio"],
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            market_regime=regime,
            atr_14=atr,
            votes=votes,
            vote_count=vote_count,
        )

        # 8. Log every signal (mandatory — AccuracyTracker needs full history)
        self.log_signal(signal)
        return signal

    def predict_batch(
        self,
        tickers: list[str],
        as_of_date: date | None = None,
    ) -> list[Signal]:
        """Run predict() for each ticker and return all signals (including HOLDs)."""
        signals: list[Signal] = []
        for ticker in tickers:
            try:
                sig = self.predict(ticker, as_of_date=as_of_date)
                signals.append(sig)
            except Exception as exc:  # noqa: BLE001
                logger.error("predict() failed for %s: %s", ticker, exc)
        return signals

    def log_signal(self, signal: Signal) -> None:
        """
        Persist signal to SQLite via DataStore for accuracy tracking.
        Uses a dedicated ``signals`` table (created on first write).
        """
        try:
            from trading_bot.data.store import DataStore

            store = DataStore()
            store.save_signal(signal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log signal for %s: %s", signal.ticker, exc)

    # ── RSI rule-based vote ─────────────────────────────────────────────────────

    def _predict_rsi(
        self,
        featured_df: pd.DataFrame,
        buy_threshold: int = 35,
        sell_threshold: int = 65,
    ) -> Action:
        """
        Rule-based vote from the latest RSI value.

        The ``ta`` library names the RSI column ``momentum_rsi``.
        Falls back to HOLD if the column is absent.
        """
        rsi_col = next(
            (c for c in featured_df.columns if "rsi" in c.lower()),
            None,
        )
        if rsi_col is None:
            logger.warning("No RSI column found in featured_df; defaulting to HOLD.")
            return Action.HOLD

        rsi_val = float(featured_df[rsi_col].iloc[-1])
        if rsi_val < buy_threshold:
            return Action.BUY
        if rsi_val > sell_threshold:
            return Action.SELL
        return Action.HOLD

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _vote_rl(self, algo: str, featured_df: pd.DataFrame) -> Action:
        """
        Get a single vote from the production RL model for *algo*.

        Falls back to HOLD if the model is unavailable (not on registry, or
        sb3 not installed — keeps the system functional on the local dev machine).
        """
        try:
            model = self._get_rl_model(algo)
            if model is None:
                return Action.HOLD
            obs = self._build_obs(featured_df)
            return self._predict_rl(model, obs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RL vote failed for %s/%s: %s — defaulting HOLD.", algo, algo, exc)
            return Action.HOLD

    def _predict_rl(self, model: object, obs: np.ndarray) -> Action:
        """Map deterministic model output to Action enum."""
        action_arr, _ = model.predict(obs, deterministic=True)  # type: ignore[union-attr]
        action_val = float(np.array(action_arr).flat[0])
        if action_val > _BUY_THRESHOLD:
            return Action.BUY
        if action_val < _SELL_THRESHOLD:
            return Action.SELL
        return Action.HOLD

    def _build_obs(self, featured_df: pd.DataFrame) -> np.ndarray:
        """Construct a (1, window_size × n_features) observation from the last rows."""
        window = featured_df.select_dtypes(include=[np.number]).iloc[-WINDOW_SIZE:]
        return window.values.flatten().reshape(1, -1).astype(np.float32)

    def _aggregate_votes(
        self, votes: list[Action]
    ) -> tuple[Action, Confidence, int]:
        """Majority-vote aggregation returning (action, confidence, vote_count)."""
        buy_count = sum(1 for v in votes if v == Action.BUY)
        sell_count = sum(1 for v in votes if v == Action.SELL)

        if buy_count >= 3:
            return Action.BUY, Confidence.HIGH, buy_count
        if buy_count == 2:
            return Action.BUY, Confidence.MEDIUM, buy_count
        if sell_count >= 3:
            return Action.SELL, Confidence.HIGH, sell_count
        if sell_count == 2:
            return Action.SELL, Confidence.MEDIUM, sell_count

        dominant_count = max(buy_count, sell_count)
        return Action.HOLD, Confidence.LOW, dominant_count

    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        """Compute ATR(window) from OHLCV DataFrame. Returns last value."""
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_series = tr.rolling(window).mean().dropna()
        if atr_series.empty:
            return float(df["Close"].std()) or 1.0
        return float(atr_series.iloc[-1])

    def _get_pipeline(self):
        """Load FeaturePipeline (cached after first load)."""
        if self._pipeline is not None:
            return self._pipeline

        from trading_bot.features.pipeline import FeaturePipeline

        path = self._pipeline_path
        if path is None:
            # Try to find it via the production SAC record
            record = self._registry.get_production_model("^NSEI", "SAC")
            if record and record.pipeline_path:
                path = Path(record.pipeline_path)

        if path and path.exists():
            self._pipeline = FeaturePipeline.load(path)
            logger.info("Loaded FeaturePipeline from %s", path)
        else:
            # Fallback: fresh unfitted pipeline (transform will work for basic features)
            self._pipeline = FeaturePipeline(window_size=WINDOW_SIZE, fit_regime=False)
            logger.warning(
                "No pipeline path found; using fresh FeaturePipeline (regime disabled)."
            )
        return self._pipeline

    def _get_rl_model(self, algo: str):
        """Load a production RL model from registry + disk (cached per algo)."""
        attr = f"_{algo.lower()}_model"
        if getattr(self, attr, None) is not None:
            return getattr(self, attr)

        try:
            from stable_baselines3 import PPO, SAC  # noqa: F401 — lazy

            from trading_bot.models.train import get_model_class

            record = self._registry.get_production_model("^NSEI", algo)
            if record is None:
                logger.info("No production %s model in registry — vote defaults to HOLD.", algo)
                return None

            model_path = Path(record.model_path)
            if not model_path.exists():
                logger.warning("Model file missing: %s", model_path)
                return None

            ModelClass = get_model_class(algo)  # noqa: N806
            model = ModelClass.load(str(model_path))
            setattr(self, attr, model)
            logger.info("Loaded %s model from %s (run: %s)", algo, model_path, record.run_name)
            return model
        except ImportError:
            logger.info(
                "stable_baselines3 not installed — %s vote defaults to HOLD.", algo
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s model: %s", algo, exc)
            return None

    def _hold_signal(
        self,
        ticker: str,
        votes: dict,
        sentiment_score: float,
        sentiment_label: str,
        regime: int,
    ) -> Signal:
        """Construct a HOLD signal with all price levels at current_price = 0."""
        return Signal(
            ticker=ticker,
            action=Action.HOLD,
            confidence=Confidence.LOW,
            generated_at=datetime.now(_IST),
            current_price=0.0,
            entry_low=0.0,
            entry_high=0.0,
            stop_loss=0.0,
            target=0.0,
            risk_reward_ratio=0.0,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            market_regime=regime,
            atr_14=0.0,
            votes=votes or {"SAC": "HOLD", "PPO": "HOLD", "RSI": "HOLD"},
            vote_count=0,
        )
