"""
StockTradingEnv v2 — Gymnasium-compatible continuous-action trading environment.

Key design decisions (from CLAUDE.md — do not reverse without discussion):
- Action space: Box(-1, 1) continuous — agent controls position sizing
- Reward: rolling Sharpe + drawdown penalty (see reward.py)
- Transaction costs: Zerodha model (0.03% + STT on sell)
- Stop-loss: force-close and terminate if drawdown from episode peak > 15%
- Observation: windowed (window_size, n_features + 4) — no look-ahead
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.config import INITIAL_BALANCE, TRANSACTION_COST_PCT, STT_RATE, WINDOW_SIZE
from trading_bot.env.reward import combined_reward

logger = logging.getLogger(__name__)

# Lazy import — gymnasium is only available in training (Colab) and test environments
try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

_DEAD_ZONE = 0.1          # actions in [-0.1, 0.1] are treated as HOLD
_STOP_LOSS_PCT = 0.15     # force-close if drawdown from episode peak > 15%
_SHARPE_WINDOW = 20       # rolling window for Sharpe calculation


class StockTradingEnv:
    """
    Continuous-action RL trading environment for a single NSE stock.

    The environment operates on the OUTPUT of FeaturePipeline — the
    DataFrame is assumed to be fully featured with no NaNs.

    Observation space:
        Box(-inf, inf, shape=(window_size, n_features + 4), dtype=float32)
        The +4 portfolio state columns appended per time-step:
            [cash_normalised, stock_value_normalised, drawdown, position_pct]

    Action space:
        Box(-1.0, 1.0, shape=(1,), dtype=float32)
        action > +0.1:  buy  (action_value * cash_balance) worth of stock
        action < -0.1:  sell (|action_value| * stock_value) worth of stock
        [-0.1, 0.1]:    hold (dead zone — prevents overtrading)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
    ) -> None:
        """
        Args:
            df: Feature-engineered DataFrame from FeaturePipeline.
                Must have a 'Close' column and a DatetimeIndex.
            window_size: Number of time-steps per observation.
            initial_balance: Starting cash balance in INR.
        """
        if not _GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for StockTradingEnv. "
                "Install it with: uv pip install 'trading-bot[training]'"
            )

        # gymnasium.Env requires these two attributes for check_env compatibility
        self.observation_space: spaces.Box
        self.action_space: spaces.Box

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self._n_features = self.df.shape[1]
        self._obs_shape = (window_size, self._n_features + 4)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Runtime state (initialised in reset)
        self.current_step: int = 0
        self.cash_balance: float = initial_balance
        self.stock_holdings: float = 0.0
        self.portfolio_history: list[float] = []
        self._returns_history: deque = deque(maxlen=_SHARPE_WINDOW)
        self._episode_peak: float = initial_balance

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.window_size  # first valid step (need window rows before it)
        self.cash_balance = self.initial_balance
        self.stock_holdings = 0.0
        self.portfolio_history = [self.initial_balance]
        self._returns_history = deque(maxlen=_SHARPE_WINDOW)
        self._episode_peak = self.initial_balance

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time-step.

        The action is priced at the NEXT bar's Close (next-bar open approximation)
        to avoid look-ahead bias.

        Args:
            action: Array of shape (1,) in [-1.0, 1.0].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        a = float(np.clip(action[0], -1.0, 1.0))

        # Price action at the NEXT bar to avoid look-ahead
        next_step = min(self.current_step + 1, len(self.df) - 1)
        execution_price = float(self.df["Close"].iloc[next_step])

        prev_portfolio = self.cash_balance + self.stock_holdings * float(
            self.df["Close"].iloc[self.current_step]
        )

        # ── Execute trade ──────────────────────────────────────────────────────
        if a > _DEAD_ZONE:
            # BUY: spend a fraction of cash
            spend = a * self.cash_balance
            shares = spend / execution_price
            cost = spend * (1.0 + TRANSACTION_COST_PCT)
            if cost <= self.cash_balance:
                self.cash_balance -= cost
                self.stock_holdings += shares
                logger.debug("BUY %.4f shares @ %.2f (cost %.2f)", shares, execution_price, cost)

        elif a < -_DEAD_ZONE:
            # SELL: liquidate a fraction of holdings
            sell_fraction = abs(a)
            shares_to_sell = sell_fraction * self.stock_holdings
            if shares_to_sell > 0:
                gross = shares_to_sell * execution_price
                stt = gross * STT_RATE
                brokerage = gross * TRANSACTION_COST_PCT
                net_proceeds = gross - stt - brokerage
                self.cash_balance += net_proceeds
                self.stock_holdings -= shares_to_sell
                logger.debug(
                    "SELL %.4f shares @ %.2f (net %.2f)", shares_to_sell, execution_price, net_proceeds
                )

        # ── Advance step ───────────────────────────────────────────────────────
        self.current_step += 1
        current_price = float(self.df["Close"].iloc[self.current_step])
        current_portfolio = self.cash_balance + self.stock_holdings * current_price

        self.portfolio_history.append(current_portfolio)
        self._episode_peak = max(self._episode_peak, current_portfolio)

        # ── Reward ─────────────────────────────────────────────────────────────
        step_return = (current_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)
        reward = combined_reward(
            self._returns_history,
            self.portfolio_history,
            step_return,
            current_portfolio,
        )

        # ── Stop-loss / termination ────────────────────────────────────────────
        drawdown = (self._episode_peak - current_portfolio) / (self._episode_peak + 1e-8)
        stop_loss_hit = drawdown > _STOP_LOSS_PCT

        if stop_loss_hit and self.stock_holdings > 0:
            # Force-liquidate all holdings
            gross = self.stock_holdings * current_price
            self.cash_balance += gross * (1.0 - TRANSACTION_COST_PCT - STT_RATE)
            self.stock_holdings = 0.0
            logger.info("Stop-loss triggered at step %d (drawdown=%.2f%%)", self.current_step, drawdown * 100)

        at_end = self.current_step >= len(self.df) - 1
        terminated = at_end or stop_loss_hit
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Print a brief portfolio summary to stdout."""
        info = self._get_info()
        print(
            f"Step {self.current_step:5d} | "
            f"Portfolio: {info['portfolio_value']:>12,.2f} | "
            f"Cash: {info['cash']:>12,.2f} | "
            f"Holdings: {info['stock_holdings']:.4f}"
        )

    def validate_env(self) -> None:
        """
        Run stable_baselines3 environment checker.

        Only usable when stable_baselines3 is installed (Colab / dev).
        """
        try:
            from stable_baselines3.common.env_checker import check_env

            check_env(self)
            logger.info("Environment validation passed.")
        except ImportError:
            logger.warning("stable_baselines3 not installed — skipping validate_env().")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the (window_size, n_features + 4) observation array.

        Uses rows [current_step - window_size : current_step] — strictly
        historical data, no look-ahead.
        """
        start = self.current_step - self.window_size
        end = self.current_step
        market_data = self.df.iloc[start:end].values.astype(np.float32)

        current_price = float(self.df["Close"].iloc[self.current_step - 1])
        stock_value = self.stock_holdings * current_price

        cash_norm = self.cash_balance / self.initial_balance
        stock_norm = stock_value / self.initial_balance
        drawdown = (self._episode_peak - (self.cash_balance + stock_value)) / (
            self._episode_peak + 1e-8
        )
        position_pct = stock_value / (self.cash_balance + stock_value + 1e-8)

        portfolio_state = np.tile(
            np.array([[cash_norm, stock_norm, drawdown, position_pct]], dtype=np.float32),
            (self.window_size, 1),
        )
        return np.hstack([market_data, portfolio_state])

    def _get_info(self) -> dict[str, Any]:
        """Return diagnostic info dict for the current step."""
        current_price = float(self.df["Close"].iloc[self.current_step])
        portfolio_value = self.cash_balance + self.stock_holdings * current_price
        return {
            "portfolio_value": portfolio_value,
            "cash": self.cash_balance,
            "stock_holdings": self.stock_holdings,
            "current_price": current_price,
            "step": self.current_step,
        }
