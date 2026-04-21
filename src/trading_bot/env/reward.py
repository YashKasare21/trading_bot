"""
Reward functions for the RL trading environment.

Combines a rolling Sharpe ratio signal with an escalating drawdown penalty
to discourage both underperformance and catastrophic losses.
"""

from collections import deque


def sharpe_reward(
    returns_history: deque,
    current_return: float,
    risk_free: float = 0.0,
) -> float:
    """
    Compute a rolling Sharpe-based reward.

    Appends *current_return* to the history deque and returns the
    rolling Sharpe ratio.  Returns 0.0 when fewer than 5 observations
    are available to avoid noise from a tiny sample.

    Args:
        returns_history: Deque (typically maxlen=20) of past step returns.
        current_return: The return for the current step.
        risk_free: Daily risk-free rate (default: 0.0).

    Returns:
        Rolling Sharpe ratio as a float.
    """
    returns_history.append(current_return)
    if len(returns_history) < 5:
        return 0.0

    import statistics

    mean = statistics.mean(returns_history) - risk_free
    std = statistics.stdev(returns_history)
    return mean / (std + 1e-8)


def drawdown_penalty(
    portfolio_history: list,
    current_value: float,
    threshold: float = 0.10,
) -> float:
    """
    Apply an escalating penalty when drawdown exceeds *threshold*.

    Args:
        portfolio_history: List of historical portfolio values.
        current_value: Current portfolio value.
        threshold: Drawdown fraction above which a penalty is applied (default: 10%).

    Returns:
        Negative penalty scaled by how far the drawdown exceeds the threshold.
        Returns 0.0 if the drawdown is within the acceptable range.
    """
    if not portfolio_history:
        return 0.0
    peak = max(portfolio_history)
    if peak <= 0:
        return 0.0
    drawdown = (peak - current_value) / peak
    if drawdown > threshold:
        return -(drawdown - threshold) * 2.0
    return 0.0


def combined_reward(
    returns_history: deque,
    portfolio_history: list,
    current_return: float,
    current_value: float,
) -> float:
    """
    Primary reward signal: rolling Sharpe + drawdown penalty.

    Args:
        returns_history: Deque of past step returns (passed to sharpe_reward).
        portfolio_history: List of past portfolio values (passed to drawdown_penalty).
        current_return: Portfolio return for the current step.
        current_value: Absolute portfolio value after the current step.

    Returns:
        Combined scalar reward.
    """
    sharpe = sharpe_reward(returns_history, current_return)
    penalty = drawdown_penalty(portfolio_history, current_value)
    return sharpe + penalty
