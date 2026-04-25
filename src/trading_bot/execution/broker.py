"""
VirtualBroker — execution engine for paper trading.

Integrates with Supabase to track open positions and trade history.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from trading_bot.config import CACHE_DIR
from trading_bot.inference.signal import Action, Confidence, Signal

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_INVESTMENT_AMOUNT = 100_000.0


class OpenPosition:
    """SQLAlchemy model for open_positions table."""

    def __init__(
        self,
        id: int | None,
        ticker: str,
        trade_type: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        confidence: str,
        entry_date: str,
    ) -> None:
        self.id = id
        self.ticker = ticker
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.target = target
        self.confidence = confidence
        self.entry_date = entry_date


class TradeHistory:
    """SQLAlchemy model for trade_history table."""

    def __init__(
        self,
        id: int | None,
        ticker: str,
        trade_type: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float,
        pnl_percentage: float,
        entry_date: str,
        exit_date: str,
        exit_reason: str,
    ) -> None:
        self.id = id
        self.ticker = ticker
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.quantity = quantity
        self.pnl = pnl
        self.pnl_percentage = pnl_percentage
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.exit_reason = exit_reason


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

        self._create_tables()

    def _create_tables(self) -> None:
        from sqlalchemy import Column, Float, Integer, MetaData, String, Table

        metadata = MetaData()

        Table(
            "open_positions",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("ticker", String, nullable=False),
            Column("trade_type", String, nullable=False),
            Column("entry_price", Float, nullable=False),
            Column("quantity", Integer, nullable=False),
            Column("stop_loss", Float, nullable=False),
            Column("target", Float, nullable=False),
            Column("confidence", String, nullable=False),
            Column("entry_date", String, nullable=False),
        )

        Table(
            "trade_history",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("ticker", String, nullable=False),
            Column("trade_type", String, nullable=False),
            Column("entry_price", Float, nullable=False),
            Column("exit_price", Float, nullable=False),
            Column("quantity", Integer, nullable=False),
            Column("pnl", Float, nullable=False),
            Column("pnl_percentage", Float, nullable=False),
            Column("entry_date", String, nullable=False),
            Column("exit_date", String, nullable=False),
            Column("exit_reason", String, nullable=False),
        )

        metadata.create_all(self._engine)

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
            result = session.execute(select(OpenPosition))
            positions = [
                OpenPosition(
                    id=row[0],
                    ticker=row[1],
                    trade_type=row[2],
                    entry_price=row[3],
                    quantity=row[4],
                    stop_loss=row[5],
                    target=row[6],
                    confidence=row[7],
                    entry_date=row[8],
                )
                for row in result.fetchall()
            ]

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
            session.execute("DELETE FROM open_positions WHERE id = :id", {"id": pos.id})

            session.add(
                TradeHistory(
                    id=None,
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
            result = session.execute(select(OpenPosition))
            open_tickers = {row[1] for row in result.fetchall()}

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
                        id=None,
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
