"""
Public API for trading_bot.models.

All heavy dependencies (torch, stable_baselines3, optuna) are lazy-imported
inside function bodies. This package is safely importable on machines without
those packages (e.g. the local dev machine used for inference and backtesting).
"""

from .registry import ModelRecord, ModelRegistry
from .train import TrainingConfig, load_agent, train_agent
from .tune import run_tuning
from .walk_forward import WalkForwardResult, run_walk_forward, summarise

__all__ = [
    # Training
    "TrainingConfig",
    "train_agent",
    "load_agent",
    # Tuning
    "run_tuning",
    # Walk-forward
    "run_walk_forward",
    "WalkForwardResult",
    "summarise",
    # Registry
    "ModelRegistry",
    "ModelRecord",
]
