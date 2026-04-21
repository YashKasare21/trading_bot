"""
RiskManager — ATR-based stop-loss and take-profit calculator.

Produces entry zone, stop-loss, and target price for BUY/SELL signals.
All levels use a fixed 1:2 risk/reward ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from trading_bot.inference.signal import Action

logger = logging.getLogger(__name__)

_SL_MULTIPLIER = 1.5   # ATR units below/above entry
_TP_MULTIPLIER = 3.0   # ATR units above/below entry (1:2 R/R)
_ENTRY_HALF_SPREAD = 0.5  # ± ATR around current price for entry zone


@dataclass(frozen=True)
class RiskLevels:
    entry_price: float   # mid of entry zone
    entry_low: float
    entry_high: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    atr: float


class RiskManager:
    """
    Calculates dynamic SL / TP levels from the current close price and ATR-14.

    Parameters
    ----------
    sl_multiplier:
        ATR multiple used for stop distance.  Default 1.5.
    tp_multiplier:
        ATR multiple used for target distance.  Default 3.0 (1:2 R/R).
    """

    def __init__(
        self,
        sl_multiplier: float = _SL_MULTIPLIER,
        tp_multiplier: float = _TP_MULTIPLIER,
    ) -> None:
        self._sl_mult = sl_multiplier
        self._tp_mult = tp_multiplier

    def calculate(
        self,
        close: float,
        atr: float,
        action: Action,
    ) -> RiskLevels:
        """
        Compute risk levels for a given action.

        Parameters
        ----------
        close:
            Latest closing price.
        atr:
            14-period ATR value.
        action:
            BUY / SELL / HOLD.

        Returns
        -------
        RiskLevels
            Immutable dataclass with entry, SL, and TP fields.
            For HOLD all fields equal *close* and R/R = 0.
        """
        if atr <= 0:
            logger.warning("ATR=%.4f is non-positive; defaulting to 1.0", atr)
            atr = 1.0

        if action == Action.BUY:
            entry_low = close - _ENTRY_HALF_SPREAD * atr
            entry_high = close + _ENTRY_HALF_SPREAD * atr
            entry_mid = close
            stop_loss = entry_low - self._sl_mult * atr
            target = entry_high + self._tp_mult * atr

        elif action == Action.SELL:
            entry_low = close - _ENTRY_HALF_SPREAD * atr
            entry_high = close + _ENTRY_HALF_SPREAD * atr
            entry_mid = close
            stop_loss = entry_high + self._sl_mult * atr
            target = entry_low - self._tp_mult * atr

        else:  # HOLD
            return RiskLevels(
                entry_price=close,
                entry_low=close,
                entry_high=close,
                stop_loss=close,
                target_price=close,
                risk_reward_ratio=0.0,
                atr=atr,
            )

        rr = self._tp_mult / self._sl_mult

        return RiskLevels(
            entry_price=round(entry_mid, 2),
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2),
            atr=round(atr, 4),
        )
