"""
main_inference.py — EOD inference entry point.

Usage
-----
# Start the live scheduler (runs Mon–Fri at 15:45 IST):
    python main_inference.py

# Run a single inference pass immediately (useful for testing):
    python main_inference.py --now

# Override tickers:
    python main_inference.py --tickers RELIANCE.NS,TCS.NS --now
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY = Path("data/models/registry.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE EOD Inference Engine")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run inference once immediately, then exit.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated NSE tickers (overrides config NSE_TICKERS).",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(_DEFAULT_REGISTRY),
        help="Path to registry.json.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=15,
        help="IST hour for scheduled run (default 15).",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=45,
        help="IST minute for scheduled run (default 45).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    tickers: list[str] | None = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    registry_path = Path(args.registry)

    if args.now:
        logger.info("--now flag set: running single inference pass.")
        from trading_bot.inference.scheduler import run_daily_inference
        run_daily_inference(tickers=tickers, registry_path=registry_path)
        sys.exit(0)

    from trading_bot.inference.scheduler import start_scheduler
    start_scheduler(
        tickers=tickers,
        registry_path=registry_path,
        hour=args.hour,
        minute=args.minute,
    )


if __name__ == "__main__":
    main()
