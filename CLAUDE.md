# CLAUDE.md — AI Trading Bot (Personal System)
# Read this file at the start of EVERY session. It is the source of truth.

## What this project is
A personal AI-powered trading system for Indian stock markets (NSE/Nifty50).
NOT a portfolio project — this is a real system I will use to generate trading signals.
Goal: post-3:30pm IST, system ingests EOD market data + news sentiment, runs ensemble RL inference,
and delivers a structured signal (BUY/SELL/HOLD + entry/stop/target) via Telegram.

## My hardware & toolchain
- Local machine: 8GB RAM, no GPU (VS Code only — no heavy training here)
- Training: Google Colab (free T4/Pro A100) — Colab notebooks in notebooks/
- Coding agent: Claude Code (you) — implement modules, write tests, refactor
- UI design: Google Stitch MCP — designs first, then Streamlit implementation
- Source control: GitHub (private repo: YashKasare21/trading_bot)
- Package manager: uv (NOT pip directly)

## Absolute rules — never break these
1. NEVER install torch, stable-baselines3, or FinRL locally. These are Colab-only.
2. NEVER commit .env files or API keys. Always use python-dotenv + .env.example pattern.
3. NEVER run training scripts locally. They belong in notebooks/03_training_colab.ipynb.
4. ALWAYS use the shared FeaturePipeline class for both training and inference — never duplicate feature logic.
5. ALWAYS write tests alongside new modules. Target 80%+ coverage on core logic.
6. ALWAYS use conventional commits: feat:, fix:, refactor:, test:, chore:
7. Data files (parquet, sqlite, model zips) go in data/ which is gitignored.

## Project structure (target)
```
trading_bot/
├── .github/workflows/ci.yml        # pytest + ruff on every push
├── src/trading_bot/
│   ├── data/
│   │   ├── fetcher.py              # yfinance NSE data fetcher
│   │   ├── sentiment.py            # Gemini batch sentiment scorer + SQLite cache
│   │   └── store.py                # Parquet/SQLite local data store
│   ├── features/
│   │   ├── pipeline.py             # FeaturePipeline class (THE shared class)
│   │   ├── technical.py            # ta library wrapper (80+ indicators)
│   │   ├── fourier.py              # FFT cycle features
│   │   └── regime.py               # HMM market regime detector
│   ├── env/
│   │   ├── trading_env.py          # StockTradingEnv v2 (continuous Box action space)
│   │   └── reward.py               # Sharpe + drawdown reward functions
│   ├── models/
│   │   ├── train.py                # PPO / SAC / TD3 training (Colab runs this)
│   │   └── tune.py                 # Optuna hyperparameter search
│   ├── backtest/
│   │   ├── engine.py               # vectorbt / quantstats wrapper
│   │   └── metrics.py              # Sharpe, max drawdown, CAGR, win rate
│   ├── inference/
│   │   ├── predictor.py            # ensemble signal generator
│   │   ├── scheduler.py            # APScheduler post-3:30pm IST trigger
│   │   └── notifier.py             # Telegram bot signal delivery
│   └── config.py                   # all constants, env var loading
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_colab.ipynb     # GPU training, runs on Colab
│   └── 04_backtest_analysis.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_env.py
│   ├── test_features.py
│   └── test_inference.py
├── dashboard/
│   └── app.py                      # Streamlit signal dashboard
├── data/                           # gitignored — cache, models, parquet
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Key technical decisions (don't reverse these without discussion)
- Action space: Box(-1, 1, shape=(1,)) — continuous, agent decides position size
- Reward: rolling Sharpe ratio (window=20) with drawdown penalty
- Primary algo: SAC (handles continuous actions better than PPO)
- Baseline algo: PPO (kept for comparison)
- Ensemble rule: signal only when ≥2 of [SAC, PPO, RSI-rule-based] agree
- Features: ta library + Fourier(3,6,12 periods) + lagged returns(1,3,5,10,20d) + HMM regime + Gemini sentiment
- Sentiment training: real historical news via NewsAPI, NOT np.random
- Indian market focus: NSE tickers (RELIANCE.NS, ^NSEI, etc.), IST timezone aware
- Transaction costs: Zerodha model (0.03% delivery + STT)

## Phases
Phase 1 (COMPLETE): Foundation — project scaffold, FeaturePipeline, env v2 skeleton, CI
Phase 2 (COMPLETE): Real sentiment pipeline, backtesting metrics, reporter, tests
Phase 3 (IN PROGRESS): Multi-algo training, Optuna tuning, walk-forward validation
  Files added:
  - src/trading_bot/models/train.py — TrainingConfig, train_agent, load_agent, build_env
  - src/trading_bot/models/tune.py — make_objective, run_tuning, best_config_from_study
  - src/trading_bot/models/walk_forward.py — generate_windows, run_walk_forward, summarise
  - src/trading_bot/models/registry.py — ModelRegistry, ModelRecord (atomic JSON registry)
  - tests/test_models.py — 16 tests (no sb3/torch needed locally)
  Walk-forward viability threshold: mean Sharpe > 0.5 (do not change without discussion)
Phase 4 (next): Inference engine + signal delivery
  - src/trading_bot/inference/predictor.py — loads production model from registry, runs ensemble
  - src/trading_bot/inference/scheduler.py — APScheduler post-3:30pm IST cron
  - src/trading_bot/inference/notifier.py — Telegram bot signal formatting + delivery
  - src/trading_bot/inference/signal.py — Signal dataclass (ticker, action, confidence, entry, stop, target)
  - dashboard/app.py — Streamlit dashboard (Signal cards + backtest tearsheet + accuracy tracker)
Phase 5: LSTM policy, multi-stock, MLflow tracking

## APIs in use
- GEMINI_API_KEY: Google AI Studio (sentiment scoring)
- NEWS_API_KEY: newsapi.org (historical headlines)
- TELEGRAM_BOT_TOKEN: @BotFather (signal delivery)
- TELEGRAM_CHAT_ID: your personal chat ID

## Original prototype (for reference during migration)
The original flat files (training_colab.py, streamlit_app.py, gemini_utils.py, config.py)
live at https://github.com/YashKasare21/trading_bot
Migrate logic from these — do not copy bugs:
- training_colab.py: StockTradingEnv, add_signals → migrate to env/ and features/
- gemini_utils.py: get_gemini_sentiment → migrate to data/sentiment.py (remove streamlit dependency)
- config.py: constants → migrate to src/trading_bot/config.py with env var loading
- streamlit_app.py: UI logic → migrate to dashboard/app.py after pipeline is solid