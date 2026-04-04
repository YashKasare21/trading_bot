"""
Configuration module for the AI Trading Bot.

Loads all settings from environment variables via python-dotenv.
This is the single source of truth for all constants used across the project.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Tickers ───────────────────────────────────────────────────────────────────
_raw_tickers = os.getenv("NSE_TICKERS", "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,^NSEI")
NSE_TICKERS: list[str] = [t.strip() for t in _raw_tickers.split(",") if t.strip()]

# ── Portfolio ─────────────────────────────────────────────────────────────────
INITIAL_BALANCE: int = int(os.getenv("INITIAL_BALANCE", "100000"))
WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "20"))

# ── Transaction costs (Zerodha model) ────────────────────────────────────────
TRANSACTION_COST_PCT: float = 0.0003   # 0.03% delivery brokerage
STT_RATE: float = 0.001                 # 0.1% Securities Transaction Tax on sell

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "data/models"))
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "data/cache"))

# ── Timezone & Market Hours ───────────────────────────────────────────────────
TIMEZONE: str = "Asia/Kolkata"
MARKET_CLOSE_HOUR: int = 15
MARKET_CLOSE_MINUTE: int = 30

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
