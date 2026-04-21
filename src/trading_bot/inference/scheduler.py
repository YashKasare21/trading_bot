"""
APScheduler-based trigger for daily EOD inference.

Runs run_daily_inference() every weekday (Mon–Fri) at 15:45 IST —
15 minutes after Nifty50 market close — so final EOD candles are settled.
"""

from __future__ import annotations

import logging
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from trading_bot.config import NSE_TICKERS, TIMEZONE
from trading_bot.inference.notifier import SignalDispatcher
from trading_bot.inference.predictor import EnsemblePredictor

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path("data/models/registry.json")
_RUN_HOUR = 15
_RUN_MINUTE = 45


def run_daily_inference(
    tickers: list[str] | None = None,
    registry_path: Path = _REGISTRY_PATH,
) -> None:
    """
    Full inference pipeline for one EOD cycle.

    Strictly two phases — model evaluation completes for ALL tickers before
    SignalDispatcher is called even once, guaranteeing exactly one dispatch
    per ticker regardless of evaluation errors or retries.

    Phase 1 — Evaluate (no Telegram calls):
        For each unique ticker: fetch OHLCV + sentiment → FeaturePipeline
        → SAC/PPO/RSI ensemble votes → RiskManager → Signal.

    Phase 2 — Dispatch (no model calls):
        For each collected Signal: call SignalDispatcher.dispatch() once.
    """
    raw_tickers = tickers or NSE_TICKERS
    # Deduplicate while preserving order — duplicate entries in NSE_TICKERS
    # would otherwise cause multiple dispatches for the same ticker.
    seen: set[str] = set()
    unique_tickers = [t for t in raw_tickers if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    logger.info("=== EOD inference started for %d tickers ===", len(unique_tickers))

    predictor = EnsemblePredictor(registry_path=registry_path)
    notifier = SignalDispatcher()

    # ── Phase 1: model evaluation ─────────────────────────────────────────────
    signals = []
    errors: list[tuple[str, Exception]] = []

    for ticker in unique_tickers:
        try:
            signal = predictor.predict(ticker)
            logger.info(
                "[%s] action=%s confidence=%s price=%.2f",
                ticker,
                signal.action.value,
                signal.confidence.value,
                signal.current_price,
            )
            signals.append(signal)
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference failed for %s: %s", ticker, exc)
            errors.append((ticker, exc))

    # ── Phase 2: dispatch (strictly after all model evaluation is complete) ───
    for signal in signals:
        notifier.dispatch(signal)

    for ticker, exc in errors:
        notifier.send_text(f"⚠️ Inference error for {ticker}: {exc}")

    logger.info(
        "=== EOD inference complete — %d signals dispatched, %d errors ===",
        len(signals),
        len(errors),
    )


def start_scheduler(
    tickers: list[str] | None = None,
    registry_path: Path = _REGISTRY_PATH,
    hour: int = _RUN_HOUR,
    minute: int = _RUN_MINUTE,
) -> None:
    """
    Start the BlockingScheduler.  Call from main_inference.py.

    Parameters
    ----------
    tickers:
        Override the default NSE_TICKERS list from config.
    registry_path:
        Path to registry.json.
    hour, minute:
        IST time for the daily job.  Defaults to 15:45.
    """
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=hour,
        minute=minute,
        timezone=TIMEZONE,
    )

    scheduler.add_job(
        run_daily_inference,
        trigger=trigger,
        kwargs={"tickers": tickers, "registry_path": registry_path},
        id="eod_inference",
        name="EOD Nifty50 Inference",
        misfire_grace_time=300,   # tolerate up to 5-min delay (e.g. system sleep)
        coalesce=True,            # run once even if multiple fires were missed
        replace_existing=True,    # prevent duplicate job if scheduler is restarted
                                  # without stopping the previous instance first
    )

    logger.info(
        "Scheduler armed: EOD inference runs Mon–Fri at %02d:%02d IST.", hour, minute
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
    except Exception as exc:
        logger.critical("Scheduler crashed: %s", exc)
        raise
