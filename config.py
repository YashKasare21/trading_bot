# Configuration file for the AI Trading Bot

# Environment Configuration
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_COST_PCT = 0.001

# Data Configuration
DATA_FILE = "data.csv"
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-12-31"

# Model Configuration
MODEL_FILE = "ppo_trading_bot.zip"
MODEL_PATH = "ppo_trading_bot.zip"

# Gemini API Configuration (API key handled via Streamlit secrets)
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Do NOT hardcode API keys here in production!