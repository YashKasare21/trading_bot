"""
AI Trading Control Room — Streamlit entry point.

Run from project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Trading Control Room",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
_IST = ZoneInfo("Asia/Kolkata")
DB_PATH = ROOT / "data" / "cache" / "trading_bot.db"
REGISTRY_PATH = ROOT / "data" / "models" / "registry.json"


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_registry() -> list[dict]:
    """Load all model records from registry.json (cached 5 min)."""
    if not REGISTRY_PATH.exists():
        return []
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_signals_df(limit: int = 500) -> pd.DataFrame:
    """
    Load recent inference signals from SQLite (cached 60 s).

    Returns an empty DataFrame if the DB or signals table is absent.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"SELECT * FROM signals ORDER BY generated_at DESC LIMIT {limit}",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Load OHLCV from SQLite price_data table (cached 1 hr).

    Falls back to an empty DataFrame if table or ticker absent.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume "
            "FROM price_data "
            "WHERE ticker = ? "
            "ORDER BY date DESC "
            f"LIMIT {days}",
            conn,
            params=(ticker,),
        )
        conn.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
    except Exception:
        return pd.DataFrame()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Trading\nControl Room")
    st.caption("NSE / Nifty50 · EOD Ensemble RL")
    st.divider()

    now_ist = datetime.now(_IST)
    weekday = now_ist.weekday()
    hour, minute = now_ist.hour, now_ist.minute

    if weekday < 5 and (9, 15) <= (hour, minute) <= (15, 30):
        st.success("🟢 Market: Open")
    elif weekday >= 5:
        st.info("🔵 Market: Weekend")
    else:
        st.warning("🔴 Market: Closed")

    st.caption(f"🕐 {now_ist.strftime('%H:%M IST · %d %b %Y')}")
    st.divider()

    registry = load_registry()
    prod_models = [r for r in registry if r.get("is_production")]
    st.metric("Production Models", len(prod_models))

    signals_df = load_signals_df()
    active_tickers = signals_df["ticker"].nunique() if not signals_df.empty else 0
    st.metric("Tracked Tickers", active_tickers)


# ── Home page ──────────────────────────────────────────────────────────────────
st.title("📈 AI Trading Control Room")
st.caption("NSE / Nifty50 · End-of-Day Ensemble RL Inference · Personal System")
st.divider()

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### ⚡ Live Signals")
    st.markdown(
        "Latest BUY / SELL / HOLD signals from today's EOD inference run. "
        "Includes entry zone, stop-loss, target, and ensemble vote breakdown."
    )
    st.page_link("pages/1_Live_Signals.py", label="Open Live Signals →")

with col_b:
    st.markdown("### 🔬 Backtest Lab")
    st.markdown(
        "Interactive candlestick chart with historical trade entries. "
        "Analyse CAGR, Sharpe, Max Drawdown, and Win Rate per ticker."
    )
    st.page_link("pages/2_Backtest_Lab.py", label="Open Backtest Lab →")

with col_c:
    st.markdown("### 🗂️ Model Registry")
    if registry:
        prod_df = pd.DataFrame(prod_models)[
            ["run_name", "algo", "ticker", "created_at"]
        ] if prod_models else pd.DataFrame(columns=["run_name", "algo", "ticker", "created_at"])
        st.dataframe(prod_df, use_container_width=True, hide_index=True)
    else:
        st.info("No models registered yet. Train and promote a model first.")

st.divider()
st.caption(
    "Data stored locally at `data/cache/trading_bot.db` · "
    "Models at `data/models/` · "
    "Inference runs Mon–Fri at 15:45 IST"
)
