"""
Tests for BacktestReporter.

All tests use synthetic data — no real market data or API calls.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_portfolio(n: int = 252, start_val: float = 100_000.0, daily_ret: float = 0.001) -> pd.Series:
    values = [start_val * (1 + daily_ret) ** i for i in range(n)]
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(values, index=dates, name="portfolio_value")


def _make_trade_log(n_pairs: int = 3) -> pd.DataFrame:
    """Build alternating BUY/SELL trade log rows."""
    rows = []
    buy_date = pd.Timestamp("2022-02-01")
    for i in range(n_pairs):
        buy_price = 1000.0 + i * 50
        sell_price = buy_price * 1.05
        sell_date = buy_date + pd.Timedelta(days=20)
        rows.append({
            "date": buy_date.date(),
            "action": "BUY",
            "price": buy_price,
            "shares": 10.0,
            "value": buy_price * 10,
            "cash_after": 0.0,
        })
        rows.append({
            "date": sell_date.date(),
            "action": "SELL",
            "price": sell_price,
            "shares": 10.0,
            "value": sell_price * 10,
            "cash_after": sell_price * 10,
        })
        buy_date = sell_date + pd.Timedelta(days=5)
    return pd.DataFrame(rows)


def _make_result(n: int = 252, with_trades: bool = True):
    from trading_bot.backtest.engine import BacktestResult
    from trading_bot.backtest.metrics import compute_all

    pv = _make_portfolio(n)
    trade_log = _make_trade_log() if with_trades else pd.DataFrame(
        columns=["date", "action", "price", "shares", "value", "cash_after"]
    )
    return BacktestResult(
        strategy_name="TestStrategy",
        ticker="^NSEI",
        start=date(2022, 1, 1),
        end=date(2022, 12, 31),
        initial_balance=100_000.0,
        portfolio_values=pv,
        trade_log=trade_log,
        metrics=compute_all(pv),
        buy_signals=[date(2022, 2, 1)],
        sell_signals=[date(2022, 3, 1)],
    )


# ── metrics_table ─────────────────────────────────────────────────────────────


def test_metrics_table_shape():
    """metrics_table must return a two-column DataFrame with Metric and Value."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result()
    table = reporter.metrics_table(result)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["Metric", "Value"]
    assert len(table) == len(result.metrics)


def test_metrics_table_contains_sharpe():
    """Sharpe key must appear in the metrics table."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result()
    table = reporter.metrics_table(result)
    assert "sharpe" in table["Metric"].values


# ── comparison_table ──────────────────────────────────────────────────────────


def test_comparison_table_columns_match_strategies():
    """comparison_table must produce one column per strategy."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    r1 = _make_result()
    r2 = _make_result()
    table = reporter.comparison_table({"Strategy_A": r1, "Strategy_B": r2})
    assert isinstance(table, pd.DataFrame)
    assert "Strategy_A" in table.columns
    assert "Strategy_B" in table.columns


def test_comparison_table_rows_are_metrics():
    """Rows must correspond to metric names."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    r1 = _make_result()
    table = reporter.comparison_table({"Only": r1})
    assert "sharpe" in table.index


# ── monthly_returns_matrix ────────────────────────────────────────────────────


def test_monthly_returns_matrix_columns():
    """Month names Jan..Dec must be the columns."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(n=252)
    matrix = reporter.monthly_returns_matrix(result)
    expected_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert list(matrix.columns) == expected_months


def test_monthly_returns_matrix_years_are_index():
    """Index must contain integer years."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(n=252)
    matrix = reporter.monthly_returns_matrix(result)
    assert all(isinstance(y, (int, np.integer)) for y in matrix.index)


# ── trade_summary ─────────────────────────────────────────────────────────────


def test_trade_summary_empty_log():
    """Empty trade log should return zeros for all metrics."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(with_trades=False)
    summary = reporter.trade_summary(result)
    assert summary["total_trades"] == 0
    assert summary["avg_hold_days"] == 0.0
    assert summary["avg_trade_return_pct"] == 0.0


def test_trade_summary_positive_return():
    """3 pairs each with 5% gain should yield avg_trade_return_pct ≈ 5.0."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(with_trades=True)
    summary = reporter.trade_summary(result)
    assert summary["total_trades"] == 6  # 3 pairs × 2
    assert summary["avg_trade_return_pct"] == pytest.approx(5.0, abs=0.1)
    assert summary["largest_win_pct"] >= summary["avg_trade_return_pct"]
    assert summary["avg_hold_days"] > 0


# ── save_html_report ──────────────────────────────────────────────────────────


def test_save_html_report_creates_file():
    """save_html_report must write a non-empty .html file."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(with_trades=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        reporter.save_html_report(result, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_save_html_report_contains_strategy_name():
    """The HTML file must contain the strategy name and ticker."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(with_trades=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        reporter.save_html_report(result, output_path)
        html = output_path.read_text(encoding="utf-8")
        assert result.strategy_name in html
        assert result.ticker in html


def test_save_html_report_creates_parent_dirs():
    """Reporter must create missing parent directories."""
    from trading_bot.backtest.reporter import BacktestReporter

    reporter = BacktestReporter()
    result = _make_result(with_trades=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "report.html"
        reporter.save_html_report(result, output_path)
        assert output_path.exists()
