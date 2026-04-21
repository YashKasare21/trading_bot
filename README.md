# AI Trading Bot — NSE/Nifty50

> **Status: Phase 1 — Foundation**

A personal AI-powered trading system for Indian stock markets (NSE/Nifty50).
Post-3:30pm IST, the system ingests EOD market data + news sentiment, runs ensemble RL inference,
and delivers a structured signal (BUY/SELL/HOLD + entry/stop/target) via Telegram.

## Architecture

```
yfinance (OHLCV)          NewsAPI (headlines)
       │                         │
       ▼                         ▼
  data/fetcher.py         data/sentiment.py
       │                    (Gemini API)
       └──────────┬──────────────┘
                  ▼
         features/pipeline.py
         (TA + Fourier + HMM + Sentiment)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   [Training]         [Inference]
  Colab notebook    inference/predictor.py
  PPO + SAC             │
                        ▼
               inference/notifier.py
               (Telegram signal delivery)
```

## Setup

```bash
git clone https://github.com/YashKasare21/trading_bot.git
cd trading_bot
cp .env.example .env
# Fill in API keys in .env
pip install uv
uv pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=src/trading_bot --cov-report=term-missing
```

## Training (Google Colab only)

Open `notebooks/03_training_colab.ipynb` in Google Colab.
Add `GEMINI_API_KEY` and `NEWS_API_KEY` to Colab Secrets.
Run all cells. Models are saved to your Google Drive.

> **Never run training locally** — torch/stable-baselines3 are Colab-only dependencies.

## Folder Structure

```
trading_bot/
├── .github/workflows/ci.yml        # pytest + ruff on every push
├── src/trading_bot/
│   ├── config.py                   # all constants, env var loading
│   ├── data/
│   │   ├── fetcher.py              # yfinance NSE data fetcher (with SQLite cache)
│   │   ├── sentiment.py            # Gemini batch sentiment scorer + SQLite cache
│   │   └── store.py                # SQLAlchemy SQLite data store
│   ├── features/
│   │   ├── pipeline.py             # FeaturePipeline (THE shared class)
│   │   ├── technical.py            # ta library wrapper (80+ indicators)
│   │   ├── fourier.py              # FFT cycle features
│   │   └── regime.py               # HMM market regime detector
│   ├── env/
│   │   ├── trading_env.py          # StockTradingEnv v2 (continuous Box action space)
│   │   └── reward.py               # Sharpe + drawdown reward functions
│   ├── models/                     # train.py + tune.py (Phase 3)
│   ├── backtest/                   # engine.py + metrics.py (Phase 2)
│   ├── inference/                  # predictor + scheduler + notifier (Phase 4)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_colab.ipynb     # GPU training — runs on Colab
│   └── 04_backtest_analysis.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_env.py
│   ├── test_features.py
│   └── test_inference.py
├── dashboard/app.py                # Streamlit signal dashboard (Phase 5)
├── data/                           # gitignored — cache, models, SQLite
├── .env.example
├── pyproject.toml
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance, NewsAPI |
| Sentiment | Google Gemini 2.0 Flash |
| Features | ta, numpy (FFT), hmmlearn (HMM) |
| RL Training | stable-baselines3 (PPO + SAC), Gymnasium |
| Hyperparameter Tuning | Optuna |
| Backtesting | vectorbt, quantstats |
| Scheduling | APScheduler |
| Signals | python-telegram-bot |
| Storage | SQLAlchemy + SQLite |
| Package Manager | uv |
| CI | GitHub Actions (ruff + pytest) |

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Current | Foundation — scaffold, FeaturePipeline, env v2, CI |
| 2 | Planned | Real sentiment pipeline, backtesting metrics |
| 3 | Planned | Multi-algo training, Optuna tuning, walk-forward validation |
| 4 | Planned | Inference engine, APScheduler, Telegram bot |
| 5 | Planned | LSTM policy, multi-stock, MLflow tracking |
