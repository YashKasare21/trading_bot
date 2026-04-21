"""
BacktestReporter — generates human-readable reports from BacktestResult objects.

No heavy dependencies: pandas + pathlib only. HTML output uses inline styles
so files open anywhere without external CSS.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_bot.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class BacktestReporter:
    """Generates tabular and HTML reports from BacktestResult objects."""

    def metrics_table(self, result: BacktestResult) -> pd.DataFrame:
        """
        Format a single result's metrics as a clean two-column DataFrame.

        Args:
            result: A completed BacktestResult.

        Returns:
            DataFrame with columns ['Metric', 'Value'].
        """
        rows = [{"Metric": k, "Value": v} for k, v in result.metrics.items()]
        return pd.DataFrame(rows)

    def comparison_table(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Side-by-side metrics comparison for multiple strategies.

        Args:
            results: Mapping of strategy name to BacktestResult.

        Returns:
            DataFrame with metrics as rows and strategy names as columns.
        """
        frames = {name: pd.Series(r.metrics, name=name) for name, r in results.items()}
        return pd.concat(frames, axis=1)

    def monthly_returns_matrix(self, result: BacktestResult) -> pd.DataFrame:
        """
        Monthly returns pivot table (rows=year, columns=month Jan..Dec).

        Values are monthly return percentages.

        Args:
            result: A completed BacktestResult.

        Returns:
            DataFrame indexed by year with month-name columns.
        """
        pv = result.portfolio_values
        if not hasattr(pv.index, "month"):
            pv.index = pd.to_datetime(pv.index)

        monthly = pv.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100.0

        pivot = pd.DataFrame(
            index=sorted(monthly_returns.index.year.unique()),
            columns=_MONTH_NAMES,
            dtype=float,
        )
        for dt, val in monthly_returns.items():
            year = dt.year
            month_name = _MONTH_NAMES[dt.month - 1]
            pivot.loc[year, month_name] = round(float(val), 2)

        return pivot

    def trade_summary(self, result: BacktestResult) -> dict:
        """
        Summarise trading activity from the trade log.

        Args:
            result: A completed BacktestResult.

        Returns:
            Dict with keys: total_trades, avg_hold_days, avg_trade_return_pct,
            largest_win_pct, largest_loss_pct.
        """
        log = result.trade_log
        if log.empty:
            return {
                "total_trades": 0,
                "avg_hold_days": 0.0,
                "avg_trade_return_pct": 0.0,
                "largest_win_pct": 0.0,
                "largest_loss_pct": 0.0,
            }

        buys = log[log["action"] == "BUY"].reset_index(drop=True)
        sells = log[log["action"] == "SELL"].reset_index(drop=True)
        n_pairs = min(len(buys), len(sells))

        hold_days: list[float] = []
        trade_returns: list[float] = []

        for i in range(n_pairs):
            buy_price = float(buys.loc[i, "price"])
            sell_price = float(sells.loc[i, "price"])
            buy_date = pd.Timestamp(buys.loc[i, "date"])
            sell_date = pd.Timestamp(sells.loc[i, "date"])

            hold_days.append((sell_date - buy_date).days)
            if buy_price > 0:
                trade_returns.append((sell_price / buy_price - 1.0) * 100.0)

        return {
            "total_trades": n_pairs * 2,
            "avg_hold_days": round(sum(hold_days) / len(hold_days), 1) if hold_days else 0.0,
            "avg_trade_return_pct": round(sum(trade_returns) / len(trade_returns), 2) if trade_returns else 0.0,
            "largest_win_pct": round(max(trade_returns), 2) if trade_returns else 0.0,
            "largest_loss_pct": round(min(trade_returns), 2) if trade_returns else 0.0,
        }

    def save_html_report(self, result: BacktestResult, output_path: Path) -> None:
        """
        Save a self-contained HTML report for a single BacktestResult.

        Includes: metrics table, monthly returns heatmap, trade log.
        No external CSS — inline styles only.

        Args:
            result: A completed BacktestResult.
            output_path: Destination .html file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_html = self.metrics_table(result).to_html(index=False, border=0)

        monthly = self.monthly_returns_matrix(result)
        monthly_html = self._monthly_matrix_to_html(monthly)

        trade_html = result.trade_log.to_html(index=False, border=0) if not result.trade_log.empty else "<p>No trades recorded.</p>"

        summary = self.trade_summary(result)
        summary_rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in summary.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Report: {result.strategy_name} — {result.ticker}</title>
<style>
  body {{ font-family: monospace; background: #0d1117; color: #e6edf3; padding: 2em; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; margin-bottom: 1.5em; width: 100%; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 6px 10px; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #21262d; }}
  .pos {{ color: #3fb950; }}
  .neg {{ color: #f85149; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p class="meta">Strategy: <b>{result.strategy_name}</b> &nbsp;|&nbsp;
Ticker: <b>{result.ticker}</b> &nbsp;|&nbsp;
Period: {result.start} → {result.end} &nbsp;|&nbsp;
Initial Balance: ₹{result.initial_balance:,.0f}</p>

<h2>Performance Metrics</h2>
{metrics_html}

<h2>Trade Summary</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
{summary_rows}
</tbody></table>

<h2>Monthly Returns (%)</h2>
{monthly_html}

<h2>Trade Log</h2>
{trade_html}
</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", output_path)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _monthly_matrix_to_html(df: pd.DataFrame) -> str:
        """Render the monthly returns matrix with green/red cell colouring."""
        header = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in df.columns) + "</tr>"
        rows = []
        for year, row in df.iterrows():
            cells = f"<td><b>{year}</b></td>"
            for val in row:
                if pd.isna(val):
                    cells += "<td>—</td>"
                elif val > 0:
                    cells += f'<td class="pos">{val:.2f}%</td>'
                else:
                    cells += f'<td class="neg">{val:.2f}%</td>'
            rows.append(f"<tr>{cells}</tr>")

        return (
            "<table><thead>"
            + header
            + "</thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
