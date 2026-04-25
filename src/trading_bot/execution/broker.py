"""
VirtualBroker — execution engine for paper trading.

Integrates with Supabase to track open positions and trade history.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, Float, Integer, String, create_engine, delete, select
from sqlalchemy.orm import Session, declarative_base

from trading_bot.config import CACHE_DIR
from trading_bot.inference.signal import Action, Confidence, Signal

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_INVESTMENT_AMOUNT = 100_000.0

Base = declarative_base()


class OpenPosition(Base):
    """SQLAlchemy model for open_positions table."""

    __tablename__ = "open_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    trade_type = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    confidence = Column(String, nullable=False)
    entry_date = Column(String, nullable=False)


class TradeHistory(Base):
    """SQLAlchemy model for trade_history table."""

    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    trade_type = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    pnl = Column(Float, nullable=False)
    pnl_percentage = Column(Float, nullable=False)
    entry_date = Column(String, nullable=False)
    exit_date = Column(String, nullable=False)
    exit_reason = Column(String, nullable=False)


class VirtualBroker:
    """Virtual broker for paper trading using Supabase/PostgreSQL."""

    def __init__(self, db_url: str | None = None) -> None:
        if db_url is None:
            db_url = __import__("os").getenv("DATABASE_URL", "")
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)

        if db_url:
            self._engine: Engine = create_engine(db_url, echo=False)
            logger.info("VirtualBroker connected to PostgreSQL via DATABASE_URL.")
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            db_path = CACHE_DIR / "trading_bot.db"
            self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
            logger.debug("VirtualBroker initialised at SQLite: %s", db_path)

        Base.metadata.create_all(self._engine)

    def process_eod(
        self,
        signals: list[Signal],
        prices: dict[str, float],
    ) -> None:
        """
        End-of-day processing: check exits and execute new entries.

        Args:
            signals: Today's Signal objects from the inference pipeline.
            prices: Dict mapping ticker -> current price.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")

        self._check_exits(prices, today_str)
        self._execute_entries(signals, prices, today_str)

    def _check_exits(self, prices: dict[str, float], today_str: str) -> None:
        """Check targets and stop-losses for open positions."""
        with Session(self._engine) as session:
            positions = session.execute(select(OpenPosition)).scalars().all()

        for pos in positions:
            current_price = prices.get(pos.ticker)
            if current_price is None:
                continue

            exit_reason = None
            if pos.trade_type == "BUY":
                if current_price >= pos.target:
                    exit_reason = "TARGET"
                elif current_price <= pos.stop_loss:
                    exit_reason = "STOP_LOSS"
            elif pos.trade_type == "SELL":
                if current_price <= pos.target:
                    exit_reason = "TARGET"
                elif current_price >= pos.stop_loss:
                    exit_reason = "STOP_LOSS"

            if exit_reason:
                self._close_position(pos, current_price, exit_reason, today_str)

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        exit_reason: str,
        today_str: str,
    ) -> None:
        """Close a position and record to trade_history."""
        pnl = (exit_price - pos.entry_price) * pos.quantity
        pnl_percentage = (
            (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
        )

        with Session(self._engine) as session:
            session.execute(delete(OpenPosition).where(OpenPosition.id == pos.id))

            session.add(
                TradeHistory(
                    ticker=pos.ticker,
                    trade_type=pos.trade_type,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    entry_date=pos.entry_date,
                    exit_date=today_str,
                    exit_reason=exit_reason,
                )
            )
            session.commit()

        logger.info(
            "Closed %s %s @ ₹%.2f (exit: %s, PnL: ₹%.2f, %.2f%%)",
            pos.ticker,
            pos.trade_type,
            exit_price,
            exit_reason,
            pnl,
            pnl_percentage,
        )

    def _execute_entries(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        today_str: str,
    ) -> None:
        """Execute new entry signals as BUY orders."""
        with Session(self._engine) as session:
            open_tickers = {
                row.ticker for row in session.execute(select(OpenPosition)).scalars().all()
            }

        for signal in signals:
            if signal.action != Action.BUY:
                continue
            if signal.confidence not in (Confidence.HIGH, Confidence.MEDIUM):
                continue
            if signal.ticker in open_tickers:
                continue

            current_price = prices.get(signal.ticker, signal.current_price)
            if current_price <= 0:
                continue

            quantity = int(_INVESTMENT_AMOUNT / current_price)
            if quantity <= 0:
                continue

            with Session(self._engine) as session:
                session.add(
                    OpenPosition(
                        ticker=signal.ticker,
                        trade_type="BUY",
                        entry_price=current_price,
                        quantity=quantity,
                        stop_loss=signal.stop_loss,
                        target=signal.target,
                        confidence=signal.confidence.value,
                        entry_date=today_str,
                    )
                )
                session.commit()

            logger.info(
                "Opened BUY position: %s @ ₹%.2f qty=%d (target=₹%.2f, stop=₹%.2f)",
                signal.ticker,
                current_price,
                quantity,
                signal.target,
                signal.stop_loss,
            )
