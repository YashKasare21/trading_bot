"""
Backtest Lab — interactive candlestick chart with trade entries and metrics.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app import load_ohlcv, load_registry, load_signals_df  # noqa: E402

st.set_page_config(
    page_title="Backtest Lab · AI Trading",
    page_icon="🔬",
    layout="wide",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Running RSI strategy backtest…")
def _run_rsi_backtest(ticker: str, start_str: str, end_str: str) -> dict:
    """
    Run RSI rule-based strategy + buy-and-hold benchmark inline.

    Returns a dict with keys:
        ohlcv (DataFrame), buy_dates, sell_dates, rsi (Series),
        portfolio_values (Series), bench_values (Series), metrics, bench_metrics.
    """
    import numpy as np
    import ta
    from trading_bot.config import STT_RATE, TRANSACTION_COST_PCT
    from trading_bot.data.fetcher import MarketDataFetcher

    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)

    fetcher = MarketDataFetcher()
    df = fetcher.fetch_ohlcv(ticker, start, end)
    if df.empty:
        return {}

    rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # RSI strategy
    initial_balance = 100_000.0
    cash, shares = initial_balance, 0.0
    portfolio_values: list[float] = []
    buy_dates: list = []
    sell_dates: list = []
    buy_prices: list[float] = []
    sell_prices: list[float] = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        rsi_val = float(rsi_series.iloc[i]) if i < len(rsi_series) and not pd.isna(rsi_series.iloc[i]) else float("nan")

        if not pd.isna(rsi_val):
            if rsi_val < 30 and cash > 0 and shares == 0:
                cost_per = price * (1.0 + TRANSACTION_COST_PCT)
                shares = cash / cost_per
                cash = 0.0
                buy_dates.append(idx)
                buy_prices.append(price)
            elif rsi_val > 70 and shares > 0:
                proceeds = shares * price * (1.0 - TRANSACTION_COST_PCT - STT_RATE)
                cash = proceeds
                shares = 0.0
                sell_dates.append(idx)
                sell_prices.append(price)

        portfolio_values.append(cash + shares * price)

    pv_series = pd.Series(portfolio_values, index=df.index)

    # Buy-and-hold benchmark
    buy_price_bh = float(df["Close"].iloc[0])
    bh_shares = (initial_balance * 0.99) / (buy_price_bh * (1.0 + TRANSACTION_COST_PCT))
    bh_cash = initial_balance - bh_shares * buy_price_bh * (1.0 + TRANSACTION_COST_PCT)
    bench_values = bh_shares * df["Close"] + bh_cash

    from trading_bot.backtest.metrics import compute_all
    metrics = compute_all(pv_series, benchmark_values=bench_values)
    bench_metrics = compute_all(bench_values)

    return {
        "ohlcv": df,
        "rsi": rsi_series,
        "buy_dates": buy_dates,
        "buy_prices": buy_prices,
        "sell_dates": sell_dates,
        "sell_prices": sell_prices,
        "portfolio_values": pv_series,
        "bench_values": bench_values,
        "metrics": metrics,
        "bench_metrics": bench_metrics,
    }


def _metric_card(label: str, value: str, delta: str | None = None, help_text: str = "") -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


# ── Page layout ────────────────────────────────────────────────────────────────
st.title("🔬 Backtest Lab")
st.caption("RSI mean-reversion strategy · vs Buy & Hold benchmark · Local OHLCV")
st.divider()

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Backtest Controls")

    # Tickers from DB + registry fallback
    signals_df = load_signals_df()
    registry = load_registry()

    db_tickers = sorted(signals_df["ticker"].unique().tolist()) if not signals_df.empty else []
    reg_tickers = sorted({r["ticker"] for r in registry}) if registry else []
    all_tickers = sorted(set(db_tickers + reg_tickers + ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS"]))

    selected_ticker = st.selectbox("Ticker", options=all_tickers, index=0)

    end_default = date.today()
    start_default = end_default - timedelta(days=365)
    start_date = st.date_input("From", value=start_default, max_value=end_default)
    end_date = st.date_input("To", value=end_default, min_value=start_date)

    run_button = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    st.divider()
    st.caption("Strategy: RSI(14) — buy < 30, sell > 70")
    st.caption("Costs: Zerodha model (0.03% + STT on sell)")

# ── Main content ───────────────────────────────────────────────────────────────
if not run_button:
    st.info("Select a ticker and date range in the sidebar, then click **▶ Run Backtest**.")
    st.stop()

result = _run_rsi_backtest(selected_ticker, str(start_date), str(end_date))

if not result:
    st.error(
        f"No OHLCV data for **{selected_ticker}** between {start_date} and {end_date}. "
        "Run inference first to populate the cache, or check yfinance connectivity."
    )
    st.stop()

df: pd.DataFrame = result["ohlcv"]
metrics: dict = result["metrics"]
bench_metrics: dict = result["bench_metrics"]

# ── Performance metrics ────────────────────────────────────────────────────────
st.subheader(f"Performance — {selected_ticker}  ·  {start_date} → {end_date}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(
    "CAGR",
    f"{metrics.get('cagr', 0):.1f}%",
    delta=f"{metrics.get('cagr', 0) - bench_metrics.get('cagr', 0):+.1f}% vs B&H",
    help="Compound Annual Growth Rate",
)
c2.metric(
    "Sharpe",
    f"{metrics.get('sharpe', 0):.2f}",
    delta=f"{metrics.get('sharpe', 0) - bench_metrics.get('sharpe', 0):+.2f} vs B&H",
    help="Annualised Sharpe (rf=6.7% Indian 10yr)",
)
c3.metric(
    "Max Drawdown",
    f"{metrics.get('max_drawdown', 0):.1f}%",
    delta=f"{metrics.get('max_drawdown', 0) - bench_metrics.get('max_drawdown', 0):+.1f}% vs B&H",
    delta_color="inverse",
    help="Peak-to-trough drawdown",
)
c4.metric(
    "Win Rate",
    f"{metrics.get('win_rate', 0):.1f}%",
    help="% of trading days with positive return",
)
c5.metric(
    "Total Return",
    f"{metrics.get('total_return_pct', 0):.1f}%",
    delta=f"{metrics.get('total_return_pct', 0) - bench_metrics.get('total_return_pct', 0):+.1f}% vs B&H",
)
c6.metric(
    "Sortino",
    f"{metrics.get('sortino', 0):.2f}",
    help="Sortino ratio (penalises downside only)",
)

st.divider()

# ── Candlestick chart ──────────────────────────────────────────────────────────
st.subheader("Price Chart · Trade Entries")

fig = go.Figure()

# OHLCV candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=selected_ticker,
        increasing_line_color="#2ecc71",
        decreasing_line_color="#e74c3c",
        increasing_fillcolor="#0d3321",
        decreasing_fillcolor="#3d0d0d",
    )
)

# Buy markers (green up-triangles)
if result["buy_dates"]:
    fig.add_trace(
        go.Scatter(
            x=result["buy_dates"],
            y=result["buy_prices"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=13,
                color="#2ecc71",
                line=dict(color="#ffffff", width=1),
            ),
            name="BUY",
            hovertemplate="BUY: ₹%{y:,.0f}<extra></extra>",
        )
    )

# Sell markers (red down-triangles)
if result["sell_dates"]:
    fig.add_trace(
        go.Scatter(
            x=result["sell_dates"],
            y=result["sell_prices"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=13,
                color="#e74c3c",
                line=dict(color="#ffffff", width=1),
            ),
            name="SELL",
            hovertemplate="SELL: ₹%{y:,.0f}<extra></extra>",
        )
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    xaxis_rangeslider_visible=False,
    height=480,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="#1a2030"),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1a2030",
        tickprefix="₹",
        tickformat=",.0f",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Portfolio vs Benchmark ─────────────────────────────────────────────────────
st.subheader("Portfolio Value vs Buy & Hold")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=result["portfolio_values"].index,
        y=result["portfolio_values"].values,
        mode="lines",
        name="RSI Strategy",
        line=dict(color="#2ecc71", width=2),
    )
)
fig2.add_trace(
    go.Scatter(
        x=result["bench_values"].index,
        y=result["bench_values"].values,
        mode="lines",
        name="Buy & Hold",
        line=dict(color="#3498db", width=1.5, dash="dot"),
    )
)
fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1a2030",
        tickprefix="₹",
        tickformat=",.0f",
    ),
    xaxis=dict(showgrid=True, gridcolor="#1a2030"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True)

# ── RSI indicator ──────────────────────────────────────────────────────────────
with st.expander("RSI(14) Indicator"):
    rsi_series: pd.Series = result["rsi"].dropna()
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=rsi_series.index,
            y=rsi_series.values,
            mode="lines",
            name="RSI(14)",
            line=dict(color="#f39c12", width=1.5),
        )
    )
    fig3.add_hline(y=70, line_dash="dash", line_color="#e74c3c", opacity=0.6, annotation_text="Overbought 70")
    fig3.add_hline(y=30, line_dash="dash", line_color="#2ecc71", opacity=0.6, annotation_text="Oversold 30")
    fig3.add_hrect(y0=30, y1=70, fillcolor="#1a2030", opacity=0.3, layer="below", line_width=0)
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1a2030"),
        xaxis=dict(showgrid=True, gridcolor="#1a2030"),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Trade log ──────────────────────────────────────────────────────────────────
with st.expander("Trade Log"):
    trades = []
    for dt, px in zip(result["buy_dates"], result["buy_prices"]):
        trades.append({"Date": dt, "Action": "BUY", "Price": f"₹{px:,.2f}"})
    for dt, px in zip(result["sell_dates"], result["sell_prices"]):
        trades.append({"Date": dt, "Action": "SELL", "Price": f"₹{px:,.2f}"})

    if trades:
        trade_df = (
            pd.DataFrame(trades)
            .sort_values("Date")
            .reset_index(drop=True)
        )
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        st.caption(
            f"{len(result['buy_dates'])} buys · {len(result['sell_dates'])} sells"
        )
    else:
        st.caption("No trades triggered in this date range (RSI stayed between 30–70).")

# ── Full metrics table ─────────────────────────────────────────────────────────
with st.expander("Full Metrics Comparison"):
    metrics_compare = pd.DataFrame(
        {"RSI Strategy": metrics, "Buy & Hold": bench_metrics}
    ).round(4)
    st.dataframe(metrics_compare, use_container_width=True)
