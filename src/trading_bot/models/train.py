"""
Training module for RL trading agents.
Supports PPO, SAC, TD3. Designed to run on Google Colab with GPU.
Import torch/sb3 lazily inside functions — never at module level.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """All hyperparameters for a single training run."""

    algo: str = "SAC"                    # "PPO", "SAC", "TD3"
    ticker: str = "^NSEI"
    train_start: date = field(default_factory=lambda: date(2018, 1, 1))
    train_end: date = field(default_factory=lambda: date(2022, 12, 31))
    total_timesteps: int = 500_000
    window_size: int = 20
    initial_balance: float = 100_000.0

    # Shared hyperparams
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99

    # PPO-specific
    n_steps: int = 2048
    clip_range: float = 0.2
    n_epochs: int = 10
    ent_coef: float = 0.01

    # SAC-specific
    buffer_size: int = 100_000
    learning_starts: int = 1000
    tau: float = 0.005
    ent_coef_sac: str = "auto"       # SAC entropy coef (auto-tuned)

    # TD3-specific
    policy_delay: int = 2
    target_policy_noise: float = 0.2

    # Checkpointing
    checkpoint_freq: int = 50_000
    save_dir: Path = field(default_factory=lambda: Path("data/models"))
    run_name: str = ""               # auto-generated if empty: "{algo}_{ticker}_{date}"

    def __post_init__(self):
        # Coerce save_dir to Path in case a str was passed
        if not isinstance(self.save_dir, Path):
            self.save_dir = Path(self.save_dir)

        if not self.run_name:
            self.run_name = (
                f"{self.algo.lower()}_"
                f"{self.ticker.replace('^', '').replace('.', '_')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_to_json_safe(config: TrainingConfig) -> dict:
    """Convert TrainingConfig to a JSON-serialisable dict."""
    raw = dataclasses.asdict(config)
    return {
        k: v.isoformat() if isinstance(v, date) else str(v) if isinstance(v, Path) else v
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_class(algo: str) -> type:
    """
    Return the SB3 model class for *algo*.

    Lazy-imports stable_baselines3 so the module stays importable without sb3.

    Parameters
    ----------
    algo:
        One of "PPO", "SAC", "TD3" (case-insensitive).

    Raises
    ------
    ValueError
        For unknown algorithm names.
    """
    import stable_baselines3 as sb3  # lazy import

    mapping = {
        "PPO": sb3.PPO,
        "SAC": sb3.SAC,
        "TD3": sb3.TD3,
    }
    key = algo.upper()
    if key not in mapping:
        raise ValueError(
            f"Unknown algo '{algo}'. Supported: {list(mapping.keys())}"
        )
    return mapping[key]


def build_env(featured_df: "pd.DataFrame", config: TrainingConfig, normalize: bool = True):
    """
    Wrap StockTradingEnv in DummyVecEnv and optionally VecNormalize.

    Parameters
    ----------
    featured_df:
        Output of FeaturePipeline — a pd.DataFrame already aligned to the
        desired split (train or eval).
    config:
        TrainingConfig instance supplying window_size and initial_balance.
    normalize:
        If True, wraps the DummyVecEnv in VecNormalize.

    Returns
    -------
    (vec_env, vec_normalize_or_none)
        vec_normalize_or_none is the VecNormalize wrapper when normalize=True,
        otherwise None.
        When normalize=True, both elements are the VecNormalize instance
        (which IS the training env).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # lazy

    from trading_bot.env.trading_env import StockTradingEnv

    def _make_env():
        return StockTradingEnv(
            df=featured_df,
            window_size=config.window_size,
            initial_balance=config.initial_balance,
        )

    vec_env = DummyVecEnv([_make_env])

    if normalize:
        vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        return vec_norm, vec_norm

    return vec_env, None


def train_agent(
    config: TrainingConfig,
    featured_df: "pd.DataFrame",
    sentiment_df: "pd.DataFrame | None" = None,
) -> dict:
    """
    Full training pipeline for a single RL agent.

    Parameters
    ----------
    config:
        Hyperparameters and paths.
    featured_df:
        Feature-engineered DataFrame (output of FeaturePipeline).
        Rows are split 80/20 into train / eval internally.
    sentiment_df:
        Optional sentiment DataFrame (currently unused; reserved for future
        multi-input architectures).

    Returns
    -------
    dict with keys: run_name, model_path, vec_norm_path, config_path,
    final_metrics.
    """
    _ = sentiment_df  # reserved for future multi-input use

    from stable_baselines3.common.callbacks import (  # lazy
        CheckpointCallback,
        EvalCallback,
        ProgressBarCallback,
    )

    from trading_bot.backtest.metrics import compute_all  # lazy

    # ------------------------------------------------------------------
    # 1. Train / eval split (80 / 20)
    # ------------------------------------------------------------------
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx].reset_index(drop=True)
    eval_df = featured_df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Training split: %d rows | Eval split: %d rows", len(train_df), len(eval_df)
    )

    # ------------------------------------------------------------------
    # 2. Build environments
    # ------------------------------------------------------------------
    train_env, vec_norm = build_env(train_df, config, normalize=True)
    eval_env, _ = build_env(eval_df, config, normalize=False)

    # ------------------------------------------------------------------
    # 3. Run directory
    # ------------------------------------------------------------------
    run_dir: Path = config.save_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ------------------------------------------------------------------
    # 4. Instantiate model
    # ------------------------------------------------------------------
    ModelClass = get_model_class(config.algo)  # noqa: N806

    algo_upper = config.algo.upper()
    if algo_upper == "PPO":
        model = ModelClass(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            clip_range=config.clip_range,
            n_epochs=config.n_epochs,
            ent_coef=config.ent_coef,
            verbose=1,
        )
    elif algo_upper == "SAC":
        model = ModelClass(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=config.tau,
            ent_coef=config.ent_coef_sac,
            verbose=1,
        )
    elif algo_upper == "TD3":
        model = ModelClass(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=config.tau,
            policy_delay=config.policy_delay,
            target_policy_noise=config.target_policy_noise,
            verbose=1,
        )
    else:
        raise ValueError(f"Unhandled algo: {config.algo}")

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=config.algo.lower(),
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )
    progress_cb = ProgressBarCallback()

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info(
        "Starting training: algo=%s, timesteps=%d", config.algo, config.total_timesteps
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_cb, eval_cb, progress_cb],
    )
    logger.info("Training complete.")

    # ------------------------------------------------------------------
    # 7. Persist artefacts
    # ------------------------------------------------------------------
    model_path = run_dir / f"{config.algo.lower()}_final"
    model.save(str(model_path))
    logger.info("Model saved to %s.zip", model_path)

    vec_norm_path: Path | None = None
    if vec_norm is not None:
        vec_norm_path = run_dir / f"vec_normalize_{config.algo.lower()}.pkl"
        vec_norm.save(str(vec_norm_path))
        logger.info("VecNormalize stats saved to %s", vec_norm_path)

    # Serialize TrainingConfig to JSON
    config_path = run_dir / "training_config.json"
    config_dict = _config_to_json_safe(config)
    config_path.write_text(json.dumps(config_dict, indent=2))
    logger.info("Training config saved to %s", config_path)

    # Feature names
    feature_names_path = run_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(list(featured_df.columns), indent=2))
    logger.info("Feature names saved to %s", feature_names_path)

    # ------------------------------------------------------------------
    # 8. Eval metrics — run trained model deterministically on eval split
    # ------------------------------------------------------------------
    portfolio_values = _collect_portfolio_values(model, eval_df, config)
    final_metrics = compute_all(portfolio_values)
    logger.info("Final eval metrics: %s", final_metrics)

    return {
        "run_name": config.run_name,
        "model_path": str(model_path.with_suffix(".zip")),
        "vec_norm_path": str(vec_norm_path) if vec_norm_path else None,
        "config_path": str(config_path),
        "final_metrics": final_metrics,
    }


def _collect_portfolio_values(
    model: object, eval_df: "pd.DataFrame", config: TrainingConfig
) -> list[float]:
    """
    Run *model* deterministically through *eval_df* and collect portfolio values.

    Uses a fresh (non-normalised) DummyVecEnv so that portfolio balance is on
    the original monetary scale expected by compute_all().
    """
    raw_env, _ = build_env(eval_df, config, normalize=False)

    obs = raw_env.reset()
    done = False
    portfolio_values: list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, dones, infos = raw_env.step(action)
        done = bool(dones[0])
        # StockTradingEnv stores portfolio_value in info
        info = infos[0] if infos else {}
        if "portfolio_value" in info:
            portfolio_values.append(float(info["portfolio_value"]))

    raw_env.close()

    if not portfolio_values:
        logger.warning(
            "No 'portfolio_value' key found in env info dict; "
            "returning empty list for metrics."
        )

    return portfolio_values


def load_agent(model_dir: Path, algo: str):
    """
    Load a trained model from *model_dir*.

    Parameters
    ----------
    model_dir:
        Directory produced by train_agent() — must contain
        ``{algo_lower}_final.zip``.  Optionally contains
        ``vec_normalize_{algo_lower}.pkl``.
    algo:
        Algorithm name ("PPO", "SAC", "TD3").

    Returns
    -------
    (model, vec_norm_path_or_none)
        model               — loaded SB3 model ready for inference.
        vec_norm_path_or_none — Path to the VecNormalize pkl if it exists,
                                else None.  Caller is responsible for
                                reconstructing the normalised env.
    """
    model_dir = Path(model_dir)
    ModelClass = get_model_class(algo)  # noqa: N806

    model_path = model_dir / f"{algo.lower()}_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = ModelClass.load(str(model_path))
    logger.info("Loaded model from %s", model_path)

    vec_norm_path = model_dir / f"vec_normalize_{algo.lower()}.pkl"
    if vec_norm_path.exists():
        logger.info("VecNormalize pkl found at %s", vec_norm_path)
        return model, vec_norm_path

    return model, None
