"""
AI Trading Control Room — Streamlit entry point.

Run from project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, declarative_base

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

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

# ── SQLAlchemy models for Virtual Portfolio ───────────────────────────────────
Base = declarative_base()


class OpenPosition(Base):
    __tablename__ = "open_positions"

    id = __import__("sqlalchemy").Column(
        __import__("sqlalchemy").Integer, primary_key=True, autoincrement=True
    )
    ticker = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)
    trade_type = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)
    entry_price = __import__("sqlalchemy").Column(__import__("sqlalchemy").Float, nullable=False)
    quantity = __import__("sqlalchemy").Column(__import__("sqlalchemy").Integer, nullable=False)
    stop_loss = __import__("sqlalchemy").Column(__import__("sqlalchemy").Float, nullable=False)
    target = __import__("sqlalchemy").Column(__import__("sqlalchemy").Float, nullable=False)
    confidence = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)
    entry_date = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)


def get_engine():
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url:
        return create_engine(db_url, echo=False)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


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
    """Load recent inference signals via SQLAlchemy (cached 60 s)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM signals ORDER BY generated_at DESC LIMIT {limit}",
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame:
    """Load OHLCV from price_data table via SQLAlchemy (cached 1 hr)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume "
                "FROM price_data "
                "WHERE ticker = :ticker "
                "ORDER BY date DESC "
                f"LIMIT {days}",
                conn,
                params={"ticker": ticker},
            )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_open_positions() -> pd.DataFrame:
    """
    Load open positions from PostgreSQL/SQLite (cached 60 s).
    """
    try:
        engine = get_engine()
        with Session(engine) as session:
            positions = session.execute(select(OpenPosition)).scalars().all()
        if not positions:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "id": p.id,
                    "ticker": p.ticker,
                    "trade_type": p.trade_type,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "stop_loss": p.stop_loss,
                    "target": p.target,
                    "confidence": p.confidence,
                    "entry_date": p.entry_date,
                }
                for p in positions
            ]
        )
        for col in ("entry_price", "quantity", "stop_loss", "target"):
            df[col] = df[col].astype(float)
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

col_a, col_b, col_c, col_d = st.columns(4)

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
    st.markdown("### 📊 Virtual Portfolio")
    positions_df = load_open_positions()
    if positions_df.empty:
        st.info("No active trades currently open.")
    else:
        total_deployed = (positions_df["entry_price"] * positions_df["quantity"]).sum()
        st.metric("Total Capital Deployed", f"₹{total_deployed:,.0f}")
        st.dataframe(
            positions_df.style.format(
                {
                    "entry_price": "₹{:.2f}",
                    "stop_loss": "₹{:.2f}",
                    "target": "₹{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with col_d:
    st.markdown("### 🗂️ Model Registry")
    if registry:
        prod_df = (
            pd.DataFrame(prod_models)[["run_name", "algo", "ticker", "created_at"]]
            if prod_models
            else pd.DataFrame(columns=["run_name", "algo", "ticker", "created_at"])
        )
        st.dataframe(prod_df, use_container_width=True, hide_index=True)
    else:
        st.info("No models registered yet. Train and promote a model first.")

st.divider()
st.caption(
    "Data stored in Supabase (DATABASE_URL) · "
    "Models at `data/models/` · "
    "Inference runs Mon–Fri at 15:45 IST"
)
