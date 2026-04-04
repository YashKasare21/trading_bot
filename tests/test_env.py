"""
Tests for StockTradingEnv v2.

All tests use the lightweight sample_featured_df fixture from conftest.py
— no real API calls or heavy model training involved.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy_action() -> np.ndarray:
    return np.array([0.5], dtype=np.float32)


def _sell_action() -> np.ndarray:
    return np.array([-0.5], dtype=np.float32)


def _hold_action() -> np.ndarray:
    return np.array([0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_returns_correct_shape(trading_env):
    """Observation shape must be (window_size, n_features + 4)."""
    obs, _ = trading_env.reset()
    n_feat = trading_env.df.shape[1]
    expected = (trading_env.window_size, n_feat + 4)
    assert obs.shape == expected, f"Expected {expected}, got {obs.shape}"


def test_reset_initialises_balance(trading_env):
    """Portfolio value immediately after reset equals initial_balance."""
    trading_env.reset()
    info = trading_env._get_info()
    assert abs(info["portfolio_value"] - trading_env.initial_balance) < 1e-2


def test_step_buy_decreases_cash(trading_env):
    """A buy action (action > 0.1) must reduce the cash balance."""
    trading_env.reset()
    initial_cash = trading_env.cash_balance
    trading_env.step(_buy_action())
    assert trading_env.cash_balance < initial_cash


def test_step_sell_increases_cash(trading_env):
    """After buying, a sell action must increase cash above post-buy level."""
    trading_env.reset()
    trading_env.step(_buy_action())
    cash_after_buy = trading_env.cash_balance
    trading_env.step(_sell_action())
    assert trading_env.cash_balance > cash_after_buy


def test_step_hold_no_trade(trading_env):
    """A hold action (dead zone) must leave stock_holdings unchanged."""
    trading_env.reset()
    # Buy first so we have something to potentially sell
    trading_env.step(_buy_action())
    holdings_before = trading_env.stock_holdings
    trading_env.step(_hold_action())
    assert trading_env.stock_holdings == pytest.approx(holdings_before)


def test_step_done_at_end(trading_env):
    """Episode must terminate when current_step reaches the last row."""
    trading_env.reset()
    terminated = False
    max_steps = len(trading_env.df)
    for _ in range(max_steps):
        _, _, terminated, _, _ = trading_env.step(_hold_action())
        if terminated:
            break
    assert terminated, "Episode never terminated after exhausting all data rows"


def test_stop_loss_triggers(sample_featured_df):
    """Episode must terminate when portfolio drawdown exceeds 15%."""
    from trading_bot.env.trading_env import StockTradingEnv

    # Build a price series that collapses 50% immediately after the first buy
    df = sample_featured_df.copy()
    # Spike price high at step window_size, then crater
    df.loc[df.index[20], "Close"] = 1_000_000.0   # inflated peak
    df.loc[df.index[21:], "Close"] = 1.0           # near-zero after

    env = StockTradingEnv(df, window_size=20, initial_balance=100_000.0)
    env.reset()
    env.step(_buy_action())   # buy at the inflated price → portfolio peak

    # Force the episode peak manually to be much higher so drawdown calc triggers
    env._episode_peak = 200_000.0  # 100k portfolio vs 200k peak = 50% drawdown

    _, _, terminated, _, _ = env.step(_hold_action())
    assert terminated, "Stop-loss did not trigger on >15% drawdown"


def test_reward_is_float(trading_env):
    """Reward returned by step() must be a Python float."""
    trading_env.reset()
    _, reward, _, _, _ = trading_env.step(_buy_action())
    assert isinstance(reward, float)


def test_observation_no_lookahead(trading_env):
    """
    Observation at step t must contain only rows prior to step t.

    We verify that the Close values in the observation window are drawn
    exclusively from indices < current_step (i.e., historical data).
    """
    trading_env.reset()
    obs, _ = trading_env.reset()

    step_t = trading_env.current_step  # window_size after reset
    close_in_obs = obs[:, trading_env.df.columns.get_loc("Close")]

    # All close values in the observation must exist in rows before current_step
    df_close_history = trading_env.df["Close"].iloc[: step_t].values
    for val in close_in_obs:
        assert any(np.isclose(val, df_close_history, atol=1e-3)), (
            f"Observation contains Close value {val:.4f} not found in "
            f"historical rows 0:{step_t}"
        )


def test_transaction_cost_applied(trading_env):
    """
    Buying then immediately selling at the same price must leave the portfolio
    below the starting balance due to transaction costs.
    """
    trading_env.reset()
    initial_portfolio = trading_env.cash_balance

    trading_env.step(_buy_action())
    trading_env.step(_sell_action())

    current_portfolio = trading_env.cash_balance + trading_env.stock_holdings * float(
        trading_env.df["Close"].iloc[trading_env.current_step]
    )
    assert current_portfolio < initial_portfolio, (
        "Round-trip trade should reduce portfolio value due to transaction costs"
    )
