"""
Backtesting engine — runs trained RL models and rule-based strategies
against historical data and produces a full BacktestResult.

torch and stable_baselines3 are lazy-imported inside methods only so the
module can be imported without the [training] extras installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.metrics import compute_all
from trading_bot.config import TRANSACTION_COST_PCT, STT_RATE
from trading_bot.data.fetcher import MarketDataFetcher
from trading_bot.data.sentiment import SentimentAnalyzer
from trading_bot.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete result from a single backtest run."""

    strategy_name: str
    ticker: str
    start: date
    end: date
    initial_balance: float
    portfolio_values: pd.Series
    trade_log: pd.DataFrame
    metrics: dict
    buy_signals: list[date] = field(default_factory=list)
    sell_signals: list[date] = field(default_factory=list)


class BacktestEngine:
    """
    Runs RL models or rule-based strategies against historical data.

    Produces portfolio value series, a trade log, and a full metrics dict.
    """

    def __init__(self, pipeline_path: Path, model_dir: Path) -> None:
        """
        Args:
            pipeline_path: Path to a saved FeaturePipeline joblib file.
            model_dir: Directory containing model zip files and VecNormalize state files.
        """
        self._pipeline = FeaturePipeline.load(pipeline_path)
        self._model_dir = model_dir
        self._fetcher = MarketDataFetcher()
        self._sentiment = SentimentAnalyzer()
        logger.info("BacktestEngine initialised with pipeline from %s", pipeline_path)

    # ── Public interface ───────────────────────────────────────────────────────

    def run_model(
        self,
        model_name: str,
        algo: str,
        ticker: str,
        start: date,
        end: date,
        initial_balance: float = 100_000.0,
    ) -> BacktestResult:
        """
        Run a trained RL model over a historical period.

        The pipeline uses .transform() (not .fit_transform()) — no data leakage.
        The model runs in deterministic mode.

        Args:
            model_name: Filename stem of the model zip (e.g. 'ppo_nifty_final').
            algo: Algorithm class name — 'PPO' or 'SAC'.
            ticker: NSE ticker (e.g. '^NSEI').
            start: First date.
            end: Last date.
            initial_balance: Starting cash in INR.

        Returns:
            BacktestResult with portfolio values, trade log, and metrics.
        """
        # Lazy import — gymnasium + sb3 only available with [training] extras
        try:
            from stable_baselines3 import PPO, SAC  # noqa: F401
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is required for run_model(). "
                "Install with: uv pip install 'trading-bot[training]'"
            ) from exc

        from trading_bot.env.trading_env import StockTradingEnv

        algo_map = {"PPO": PPO, "SAC": SAC}
        if algo not in algo_map:
            raise ValueError(f"Unknown algo '{algo}'. Choose 'PPO' or 'SAC'.")

        _df, featured_df = self._prepare_features(ticker, start, end)
        env = DummyVecEnv([lambda: StockTradingEnv(featured_df, window_size=20, initial_balance=initial_balance)])

        vec_norm_filename = f"vec_normalize_{algo.lower()}.pkl"
        vec_norm_path = self._model_dir / vec_norm_filename
        if vec_norm_path.exists():
            env = VecNormalize.load(str(vec_norm_path), env)
            env.training = False
            env.norm_reward = False

        model_path = self._model_dir / f"{model_name}.zip"
        AlgoCls = algo_map[algo]
        model = AlgoCls.load(str(model_path), env=env)

        obs = env.reset()
        portfolio_values = []
        buy_signals: list[date] = []
        sell_signals: list[date] = []
        done = False

        inner_env: StockTradingEnv = env.envs[0]  # type: ignore[attr-defined]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, info = env.step(action)
            done = bool(terminated[0])

            step_info = info[0]
            portfolio_values.append(step_info["portfolio_value"])

            a = float(action[0][0])
            step_idx = inner_env.current_step
            sig_date = inner_env.df.index[step_idx] if hasattr(inner_env.df.index, "__iter__") else step_idx
            if a > 0.1:
                buy_signals.append(sig_date)
            elif a < -0.1:
                sell_signals.append(sig_date)

        pv_series = pd.Series(portfolio_values, name="portfolio_value")
        trade_log = pd.DataFrame(columns=["date", "action", "price", "shares", "value", "cash_after"])

        return BacktestResult(
            strategy_name=f"{algo}:{model_name}",
            ticker=ticker,
            start=start,
            end=end,
            initial_balance=initial_balance,
            portfolio_values=pv_series,
            trade_log=trade_log,
            metrics=compute_all(pv_series),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
        )

    def run_benchmark(
        self,
        ticker: str,
        start: date,
        end: date,
        initial_balance: float = 100_000.0,
    ) -> BacktestResult:
        """
        Buy-and-hold benchmark strategy.

        Buys on day 1 (applying transaction cost) and holds until the end.

        Args:
            ticker: NSE ticker.
            start: First date.
            end: Last date.
            initial_balance: Starting cash in INR.

        Returns:
            BacktestResult.
        """
        df = self._fetcher.fetch_ohlcv(ticker, start, end)
        if df.empty:
            raise ValueError(f"No OHLCV data for {ticker} between {start} and {end}")

        buy_price = float(df["Close"].iloc[0])
        entry_cost = buy_price * (1.0 + TRANSACTION_COST_PCT)
        # Hold back 1% as cash buffer
        shares = (initial_balance * 0.99) / entry_cost
        cash_reserve = initial_balance - shares * entry_cost

        portfolio_values = []
        for price in df["Close"]:
            pv = shares * float(price) + cash_reserve
            portfolio_values.append(max(pv, 0.0))

        pv_series = pd.Series(portfolio_values, index=df.index, name="portfolio_value")
        buy_date = df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
        trade_log = pd.DataFrame([{
            "date": buy_date,
            "action": "BUY",
            "price": buy_price,
            "shares": shares,
            "value": shares * buy_price,
            "cash_after": cash_reserve,
        }])

        return BacktestResult(
            strategy_name="Buy&Hold",
            ticker=ticker,
            start=start,
            end=end,
            initial_balance=initial_balance,
            portfolio_values=pv_series,
            trade_log=trade_log,
            metrics=compute_all(pv_series),
            buy_signals=[buy_date],
            sell_signals=[],
        )

    def run_rule_based(
        self,
        ticker: str,
        start: date,
        end: date,
        initial_balance: float = 100_000.0,
        rsi_buy: int = 30,
        rsi_sell: int = 70,
    ) -> BacktestResult:
        """
        RSI mean-reversion strategy.

        Buy when RSI < rsi_buy, sell when RSI > rsi_sell.
        This is the minimum baseline the RL model must beat.

        Args:
            ticker: NSE ticker.
            start: First date.
            end: Last date.
            initial_balance: Starting cash in INR.
            rsi_buy: RSI threshold to trigger buy (default: 30).
            rsi_sell: RSI threshold to trigger sell (default: 70).

        Returns:
            BacktestResult.
        """
        import ta

        df = self._fetcher.fetch_ohlcv(ticker, start, end)
        if df.empty:
            raise ValueError(f"No OHLCV data for {ticker} between {start} and {end}")

        rsi_series = ta.momentum.RSIIndicator(df["Close"]).rsi()

        cash = initial_balance
        shares = 0.0
        portfolio_values = []
        trade_log_rows = []
        buy_signals: list[date] = []
        sell_signals: list[date] = []

        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row["Close"])
            rsi_val = float(rsi_series.iloc[i]) if i < len(rsi_series) else float("nan")

            if not np.isnan(rsi_val):
                if rsi_val < rsi_buy and cash > 0 and shares == 0:
                    entry_cost_per_share = price * (1.0 + TRANSACTION_COST_PCT)
                    new_shares = cash / entry_cost_per_share
                    cash -= new_shares * entry_cost_per_share
                    shares += new_shares
                    sig_date = idx.date() if hasattr(idx, "date") else idx
                    buy_signals.append(sig_date)
                    trade_log_rows.append({
                        "date": sig_date,
                        "action": "BUY",
                        "price": price,
                        "shares": new_shares,
                        "value": new_shares * price,
                        "cash_after": cash,
                    })

                elif rsi_val > rsi_sell and shares > 0:
                    gross = shares * price
                    proceeds = gross * (1.0 - TRANSACTION_COST_PCT - STT_RATE)
                    cash += proceeds
                    sig_date = idx.date() if hasattr(idx, "date") else idx
                    sell_signals.append(sig_date)
                    trade_log_rows.append({
                        "date": sig_date,
                        "action": "SELL",
                        "price": price,
                        "shares": shares,
                        "value": shares * price,
                        "cash_after": cash,
                    })
                    shares = 0.0

            pv = cash + shares * price
            portfolio_values.append(pv)

        pv_series = pd.Series(portfolio_values, index=df.index, name="portfolio_value")
        trade_df = (
            pd.DataFrame(trade_log_rows)
            if trade_log_rows
            else pd.DataFrame(columns=["date", "action", "price", "shares", "value", "cash_after"])
        )

        return BacktestResult(
            strategy_name=f"RSI({rsi_buy}/{rsi_sell})",
            ticker=ticker,
            start=start,
            end=end,
            initial_balance=initial_balance,
            portfolio_values=pv_series,
            trade_log=trade_df,
            metrics=compute_all(pv_series),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
        )

    def compare(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Build a side-by-side comparison of multiple backtest results.

        Args:
            results: Mapping of strategy name to BacktestResult.

        Returns:
            DataFrame with metrics as rows and strategy names as columns.
        """
        all_metrics: dict[str, dict] = {name: r.metrics for name, r in results.items()}
        return pd.DataFrame(all_metrics)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_features(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch OHLCV + sentiment and run through the pipeline."""
        df = self._fetcher.fetch_ohlcv(ticker, start, end)
        if df.empty:
            raise ValueError(f"No OHLCV data for {ticker} between {start} and {end}")

        try:
            sentiment_df = self._sentiment.fetch_and_score_news(ticker, start, end)
        except Exception as exc:
            logger.warning("Sentiment fetch failed (%s) — using zero sentiment.", exc)
            sentiment_df = None

        featured_df = self._pipeline.transform(df, sentiment_df)
        return df, featured_df
