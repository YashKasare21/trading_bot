"""
Pure-function backtesting metrics for portfolio evaluation.

No external backtesting library dependency — all functions work standalone
with pandas Series. Designed for daily OHLCV / portfolio granularity.

Risk-free rate default: 6.7% p.a. (approximate Indian 10-year bond yield).
"""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_RISK_FREE_ANNUAL_DEFAULT = 0.067


def compute_daily_returns(portfolio_values: pd.Series) -> pd.Series:
    """
    Compute log returns from a portfolio value series.

    Args:
        portfolio_values: Series of portfolio values indexed by date.

    Returns:
        Series of log returns (first value dropped — requires prior value).
    """
    return np.log(portfolio_values / portfolio_values.shift(1)).dropna()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = _RISK_FREE_ANNUAL_DEFAULT,
) -> float:
    """
    Annualised Sharpe ratio.

    Args:
        returns: Series of daily returns (log or simple — consistent use assumed).
        risk_free_annual: Annual risk-free rate (default: 6.7% Indian 10yr bond).

    Returns:
        Annualised Sharpe. Returns 0.0 if standard deviation is zero.
    """
    if len(returns) < 2:
        return 0.0
    risk_free_daily = risk_free_annual / _TRADING_DAYS
    excess = returns - risk_free_daily
    std = excess.std()
    if std < 1e-10 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * math.sqrt(_TRADING_DAYS))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: float = _RISK_FREE_ANNUAL_DEFAULT,
) -> float:
    """
    Annualised Sortino ratio (penalises only downside volatility).

    Args:
        returns: Series of daily returns.
        risk_free_annual: Annual risk-free rate.

    Returns:
        Annualised Sortino. Returns float('inf') if there are no negative returns.
        Returns 0.0 if downside deviation is zero.
    """
    if len(returns) < 2:
        return 0.0
    risk_free_daily = risk_free_annual / _TRADING_DAYS
    excess = returns - risk_free_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = math.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * math.sqrt(_TRADING_DAYS))


def max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.

    Args:
        portfolio_values: Series of portfolio values.

    Returns:
        Max drawdown as a positive fraction (e.g. 0.23 means 23% drawdown).
    """
    if len(portfolio_values) < 2:
        return 0.0
    cummax = portfolio_values.cummax()
    drawdown = (cummax - portfolio_values) / cummax
    return float(drawdown.max())


def cagr(
    portfolio_values: pd.Series,
    trading_days: int = _TRADING_DAYS,
) -> float:
    """
    Compound Annual Growth Rate.

    Args:
        portfolio_values: Series of portfolio values.
        trading_days: Number of trading days per year (default: 252).

    Returns:
        CAGR as a fraction (e.g. 0.15 means 15% p.a.).
    """
    if len(portfolio_values) < 2:
        return 0.0
    start_val = float(portfolio_values.iloc[0])
    end_val = float(portfolio_values.iloc[-1])
    if start_val <= 0:
        return 0.0
    n_years = len(portfolio_values) / trading_days
    if n_years <= 0:
        return 0.0
    return float((end_val / start_val) ** (1.0 / n_years) - 1.0)


def win_rate(returns: pd.Series) -> float:
    """
    Fraction of periods with a positive return.

    Args:
        returns: Series of daily returns.

    Returns:
        Win rate in [0.0, 1.0]. Returns 0.0 if no returns.
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """
    Ratio of total gains to total losses (absolute values).

    Args:
        returns: Series of daily returns.

    Returns:
        Profit factor. Returns float('inf') if there are no losing days.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def calmar_ratio(portfolio_values: pd.Series) -> float:
    """
    CAGR divided by maximum drawdown.

    Args:
        portfolio_values: Series of portfolio values.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    mdd = max_drawdown(portfolio_values)
    if mdd == 0:
        return 0.0
    return float(cagr(portfolio_values) / mdd)


def compute_all(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series | None = None,
) -> dict:
    """
    Compute the full set of performance metrics.

    Args:
        portfolio_values: Series of portfolio values indexed by date.
        benchmark_values: Optional buy-and-hold benchmark for alpha/beta calculation.

    Returns:
        Dict with keys:
            sharpe, sortino, max_drawdown, cagr, win_rate, profit_factor, calmar,
            total_return_pct, best_day_pct, worst_day_pct, avg_daily_return_pct.
        If benchmark_values provided, also includes: benchmark_cagr, alpha, beta.

        Percentage values stored as percentages (e.g. 12.3 not 0.123).
    """
    returns = compute_daily_returns(portfolio_values)
    total_return = (float(portfolio_values.iloc[-1]) / float(portfolio_values.iloc[0]) - 1.0) * 100.0

    result: dict = {
        "sharpe": round(sharpe_ratio(returns), 4),
        "sortino": round(sortino_ratio(returns), 4),
        "max_drawdown": round(max_drawdown(portfolio_values) * 100.0, 4),
        "cagr": round(cagr(portfolio_values) * 100.0, 4),
        "win_rate": round(win_rate(returns) * 100.0, 4),
        "profit_factor": round(profit_factor(returns), 4),
        "calmar": round(calmar_ratio(portfolio_values), 4),
        "total_return_pct": round(total_return, 4),
        "best_day_pct": round(float(returns.max() * 100.0), 4) if len(returns) else 0.0,
        "worst_day_pct": round(float(returns.min() * 100.0), 4) if len(returns) else 0.0,
        "avg_daily_return_pct": round(float(returns.mean() * 100.0), 4) if len(returns) else 0.0,
    }

    if benchmark_values is not None:
        bench_returns = compute_daily_returns(benchmark_values)
        result["benchmark_cagr"] = round(cagr(benchmark_values) * 100.0, 4)

        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) >= 2:
            port_aligned = returns.loc[common_idx].values
            bench_aligned = bench_returns.loc[common_idx].values
            cov_matrix = np.cov(port_aligned, bench_aligned)
            beta_val = float(cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10))
            alpha_daily = float(np.mean(port_aligned) - beta_val * np.mean(bench_aligned))
            result["beta"] = round(beta_val, 4)
            result["alpha"] = round(alpha_daily * _TRADING_DAYS * 100.0, 4)
        else:
            result["beta"] = 0.0
            result["alpha"] = 0.0

    return result
