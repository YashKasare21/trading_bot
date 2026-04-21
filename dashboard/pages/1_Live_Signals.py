"""
Live Signals — latest EOD inference output per ticker.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Re-use cached loaders from the entry point (same Streamlit process)
from app import load_signals_df  # noqa: E402  (Streamlit pages share parent sys.path)

_IST = ZoneInfo("Asia/Kolkata")

_REGIME_LABELS = {0: "🐻 Bearish", 1: "➡️ Sideways", 2: "🐂 Bullish"}
_ACTION_COLORS = {
    "BUY": "background-color: #0d3321; color: #2ecc71; font-weight: bold",
    "SELL": "background-color: #3d0d0d; color: #e74c3c; font-weight: bold",
    "HOLD": "background-color: #1a1a2e; color: #95a5a6",
}
_CONF_COLORS = {
    "HIGH": "color: #f39c12; font-weight: bold",
    "MEDIUM": "color: #e67e22",
    "LOW": "color: #7f8c8d",
}

st.set_page_config(
    page_title="Live Signals · AI Trading",
    page_icon="⚡",
    layout="wide",
)


# ── Header ─────────────────────────────────────────────────────────────────────
now_ist = datetime.now(_IST)
weekday = now_ist.weekday()
hour, minute = now_ist.hour, now_ist.minute

is_market_hours = weekday < 5 and (9, 15) <= (hour, minute) <= (15, 30)
market_status = "🟢 Market Open" if is_market_hours else "🔴 Market Closed"

st.title("⚡ Live Signals")
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.caption(f"Today: **{now_ist.strftime('%A, %d %B %Y')}** · {market_status}")
with h_col2:
    if not is_market_hours and weekday < 5:
        if (hour, minute) < (15, 45):
            st.caption("🕐 Next inference: **15:45 IST** today")
        else:
            st.caption("✅ Inference complete for today")
    elif weekday >= 5:
        st.caption("📅 Next inference: **Monday 15:45 IST**")

st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
raw_df = load_signals_df()

if raw_df.empty:
    st.info(
        "No signals found yet. Run `python main_inference.py --now` "
        "to generate the first batch of signals."
    )
    st.stop()

# Latest signal per ticker
latest_df = (
    raw_df.sort_values("generated_at", ascending=False)
    .drop_duplicates(subset="ticker", keep="first")
    .copy()
)

# ── KPI Metrics Row ────────────────────────────────────────────────────────────
avg_regime_raw = latest_df["market_regime"].mean() if "market_regime" in latest_df.columns else 1.0
avg_regime = round(avg_regime_raw)
regime_label = _REGIME_LABELS.get(avg_regime, "Unknown")

avg_sentiment = latest_df["sentiment_score"].mean() if "sentiment_score" in latest_df.columns else 0.0
if avg_sentiment > 0.1:
    sentiment_label = "😊 Positive"
elif avg_sentiment < -0.1:
    sentiment_label = "😟 Negative"
else:
    sentiment_label = "😐 Neutral"

actionable = latest_df[latest_df["action"].isin(["BUY", "SELL"])]
buy_count = (latest_df["action"] == "BUY").sum()
sell_count = (latest_df["action"] == "SELL").sum()
hold_count = (latest_df["action"] == "HOLD").sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("HMM Market Regime", regime_label)
m2.metric("Aggregate Sentiment", sentiment_label, f"{avg_sentiment:+.2f}")
m3.metric("Active Tickers", len(latest_df))
m4.metric("BUY Signals", int(buy_count), delta=None)
m5.metric("SELL Signals", int(sell_count), delta=None)

st.divider()

# ── Signals Table ──────────────────────────────────────────────────────────────
st.subheader("Today's Signals")

filter_col, refresh_col = st.columns([4, 1])
with filter_col:
    action_filter = st.multiselect(
        "Filter by Action",
        options=["BUY", "SELL", "HOLD"],
        default=["BUY", "SELL", "HOLD"],
    )
with refresh_col:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

display_df = latest_df[latest_df["action"].isin(action_filter)].copy()

if display_df.empty:
    st.warning("No signals match the selected filter.")
    st.stop()

# Build display columns
display_df["Entry Zone"] = display_df.apply(
    lambda r: f"₹{r['entry_low']:,.0f} – ₹{r['entry_high']:,.0f}", axis=1
)
display_df["Stop-Loss"] = display_df["stop_loss"].apply(lambda x: f"₹{x:,.0f}")
display_df["Target"] = display_df["target"].apply(lambda x: f"₹{x:,.0f}")
display_df["R/R"] = display_df["risk_reward_ratio"].apply(
    lambda x: f"1 : {x:.1f}" if x > 0 else "–"
)
display_df["Votes"] = display_df["vote_count"].apply(lambda x: f"{x}/3")
display_df["Regime"] = display_df["market_regime"].map(_REGIME_LABELS).fillna("Unknown")
display_df["Price"] = display_df["current_price"].apply(lambda x: f"₹{x:,.2f}")

table_cols = [
    "ticker", "action", "confidence", "Votes",
    "Price", "Entry Zone", "Stop-Loss", "Target", "R/R",
    "Regime", "sentiment_label",
]
table_df = display_df[table_cols].rename(columns={
    "ticker": "Ticker",
    "action": "Action",
    "confidence": "Confidence",
    "sentiment_label": "Sentiment",
})


def _style_action(val: str) -> str:
    return _ACTION_COLORS.get(val, "")


def _style_confidence(val: str) -> str:
    return _CONF_COLORS.get(val, "")


styled = (
    table_df.style
    .map(_style_action, subset=["Action"])
    .map(_style_confidence, subset=["Confidence"])
    .set_properties(**{"text-align": "right"}, subset=["Price", "Entry Zone", "Stop-Loss", "Target", "R/R"])
)

st.dataframe(styled, use_container_width=True, hide_index=True, height=450)

# ── Signal Detail Expander ─────────────────────────────────────────────────────
st.divider()
st.subheader("Signal Detail")
selected_ticker = st.selectbox(
    "Select ticker to inspect",
    options=sorted(display_df["ticker"].tolist()),
)

if selected_ticker:
    row = display_df[display_df["ticker"] == selected_ticker].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Action", row["action"])
    d2.metric("Confidence", f"{row['confidence']} ({row['Votes']})")
    d3.metric("Current Price", f"₹{row['current_price']:,.2f}")
    d4.metric("ATR-14", f"₹{row['atr_14']:,.2f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Entry Zone", row["Entry Zone"])
    p2.metric("Stop-Loss", row["Stop-Loss"])
    p3.metric("Target", row["Target"])

    with st.expander("Ensemble Vote Breakdown"):
        votes_raw = row.get("votes", {})
        if isinstance(votes_raw, dict) and votes_raw:
            vote_df = pd.DataFrame(
                [{"Model": k, "Vote": v} for k, v in votes_raw.items()]
            )
            st.dataframe(vote_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Vote data unavailable.")

    with st.expander("Signal History — last 10"):
        hist = (
            raw_df[raw_df["ticker"] == selected_ticker]
            .sort_values("generated_at", ascending=False)
            .head(10)[["generated_at", "action", "confidence", "current_price", "stop_loss", "target"]]
        )
        hist.columns = ["Generated At", "Action", "Confidence", "Price", "SL", "TP"]
        st.dataframe(hist, use_container_width=True, hide_index=True)
