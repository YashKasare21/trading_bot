"""
Optuna hyperparameter tuning for RL trading agents (PPO / SAC).

All heavy imports (optuna, stable_baselines3, matplotlib) are lazy —
this module stays importable without those packages installed.
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import optuna
    import pandas as pd

    from trading_bot.models.train import TrainingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Net-arch mapping (PPO only)
# ---------------------------------------------------------------------------
_NET_ARCH_MAP: dict[str, list[int]] = {
    "small":  [64, 64],
    "medium": [256, 256],
    "large":  [256, 256, 128],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_objective(
    featured_df: "pd.DataFrame",
    algo: str,
    train_end_idx: int,
    n_timesteps: int = 100_000,
    n_eval_episodes: int = 3,
) -> "Callable[[optuna.Trial], float]":
    """
    Build and return an Optuna objective closure for *algo*.

    The returned callable is suitable for ``study.optimize(objective, ...)``.
    All heavy dependencies are imported lazily inside the closure so this
    factory function itself requires no special packages.

    Parameters
    ----------
    featured_df:
        Full feature DataFrame (train + validation rows).
    algo:
        Algorithm name — ``"PPO"`` or ``"SAC"`` (case-insensitive).
    train_end_idx:
        Integer row index that separates the train split (``[:train_end_idx]``)
        from the validation split (``[train_end_idx:]``).
    n_timesteps:
        Number of environment steps used *per trial* for training.
    n_eval_episodes:
        Number of evaluation episodes used to collect portfolio values.
        (Currently informational; evaluation runs a single episode per env.)

    Returns
    -------
    Callable
        Optuna-compatible objective function ``(trial) -> float``.
    """

    algo_upper = algo.upper()

    def objective(trial: "optuna.Trial") -> float:  # noqa: ANN001
        # Lazy imports — all inside closure so no top-level dependency
        import optuna as _optuna  # noqa: F401 (needed for TrialPruned)
        import pandas as pd

        from trading_bot.backtest.metrics import compute_all
        from trading_bot.models.train import TrainingConfig, build_env, get_model_class

        # ------------------------------------------------------------------
        # 1. Sample hyperparameters
        # ------------------------------------------------------------------
        if algo_upper == "PPO":
            learning_rate: float = trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            )
            n_steps: int = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
            batch_size: int = trial.suggest_categorical("batch_size", [64, 128, 256])
            gamma: float = trial.suggest_float("gamma", 0.95, 0.999)
            clip_range: float = trial.suggest_float("clip_range", 0.1, 0.4)
            ent_coef: float = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
            net_arch_key: str = trial.suggest_categorical(
                "net_arch", ["small", "medium", "large"]
            )
            policy_kwargs = {"net_arch": _NET_ARCH_MAP[net_arch_key]}

        elif algo_upper == "SAC":
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            buffer_size: int = trial.suggest_categorical(
                "buffer_size", [50_000, 100_000, 200_000]
            )
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
            gamma = trial.suggest_float("gamma", 0.95, 0.999)
            tau: float = trial.suggest_float("tau", 1e-3, 5e-2, log=True)
            learning_starts: int = trial.suggest_categorical(
                "learning_starts", [1000, 5000, 10000]
            )

            # (SAC model kwargs are passed explicitly below)

        else:
            raise ValueError(
                f"make_objective only supports PPO and SAC, got '{algo}'"
            )

        # ------------------------------------------------------------------
        # 2. Build config (no checkpointing, temp save_dir)
        # ------------------------------------------------------------------
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"optuna_{algo_upper}_trial{trial.number}_"))

        config = TrainingConfig(
            algo=algo_upper,
            total_timesteps=n_timesteps,
            save_dir=tmp_dir,
            checkpoint_freq=n_timesteps + 1,  # effectively no checkpoints
            # Override scalar hypers that live on the config dataclass
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            **(
                {
                    "n_steps": n_steps,
                    "clip_range": clip_range,
                    "ent_coef": ent_coef,
                }
                if algo_upper == "PPO"
                else {
                    "buffer_size": buffer_size,
                    "tau": tau,
                    "learning_starts": learning_starts,
                }
            ),
        )

        # ------------------------------------------------------------------
        # 3. Build environments
        # ------------------------------------------------------------------
        train_df = featured_df.iloc[:train_end_idx].reset_index(drop=True)
        val_df = featured_df.iloc[train_end_idx:].reset_index(drop=True)

        try:
            train_env, _ = build_env(train_df, config, normalize=True)
            val_env_raw, _ = build_env(val_df, config, normalize=False)

            # ------------------------------------------------------------------
            # 4. Instantiate model with trial params
            # ------------------------------------------------------------------
            ModelClass = get_model_class(algo_upper)  # noqa: N806

            if algo_upper == "PPO":
                model = ModelClass(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                )
            else:  # SAC
                model = ModelClass(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                    learning_starts=learning_starts,
                    verbose=0,
                )

            # ------------------------------------------------------------------
            # 5. Train
            # ------------------------------------------------------------------
            model.learn(total_timesteps=n_timesteps)

            # ------------------------------------------------------------------
            # 6. Evaluate on validation split
            # ------------------------------------------------------------------
            obs = val_env_raw.reset()
            done = False
            portfolio_values: list[float] = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, dones, infos = val_env_raw.step(action)
                done = bool(dones[0])
                info = infos[0] if infos else {}
                if "portfolio_value" in info:
                    portfolio_values.append(float(info["portfolio_value"]))

            val_env_raw.close()
            train_env.close()

            if len(portfolio_values) < 2:
                logger.warning(
                    "Trial %d: fewer than 2 portfolio values collected; returning -999.0",
                    trial.number,
                )
                return -999.0

            # compute_all expects a pd.Series
            pv_series = pd.Series(portfolio_values, name="portfolio_value")
            metrics = compute_all(pv_series)
            sharpe: float = float(metrics.get("sharpe", -999.0))

            # ------------------------------------------------------------------
            # 7. Prune unpromising trials
            # ------------------------------------------------------------------
            if sharpe < -0.5:
                raise _optuna.exceptions.TrialPruned(
                    f"Sharpe {sharpe:.3f} < -0.5 — pruning trial {trial.number}"
                )

            return sharpe

        except _optuna.exceptions.TrialPruned:
            raise  # let Optuna handle pruning cleanly
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Trial %d raised an exception (%s: %s) — returning -999.0",
                trial.number,
                type(exc).__name__,
                exc,
            )
            return -999.0

    return objective


def run_tuning(
    featured_df: "pd.DataFrame",
    algo: str = "SAC",
    n_trials: int = 50,
    n_timesteps_per_trial: int = 100_000,
    study_name: str = "",
    storage_path: "Path | None" = None,
    n_jobs: int = 1,
) -> dict:
    """
    Run an Optuna hyperparameter search for *algo* on *featured_df*.

    Parameters
    ----------
    featured_df:
        Full feature DataFrame — split 80/20 internally.
    algo:
        ``"PPO"`` or ``"SAC"`` (case-insensitive).
    n_trials:
        Number of Optuna trials to run.
    n_timesteps_per_trial:
        Training timesteps per trial (keep low for speed; 50k–200k typical).
    study_name:
        Optuna study name.  Auto-generated as
        ``tune_{algo}_{YYYYmmdd_HHMM}`` if empty.
    storage_path:
        Optional :class:`pathlib.Path` to a SQLite file.  When supplied the
        study is persisted and can be resumed across sessions.
    n_jobs:
        Parallel trial workers.  Use ``1`` when SB3/torch are not fork-safe.

    Returns
    -------
    dict
        Keys: ``best_params``, ``best_value``, ``study``, ``best_config``.
    """
    import optuna  # lazy

    from trading_bot.models.train import TrainingConfig  # lazy

    # ------------------------------------------------------------------
    # Study name + storage
    # ------------------------------------------------------------------
    if not study_name:
        study_name = f"tune_{algo.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    storage: "str | None" = None
    if storage_path is not None:
        storage = f"sqlite:///{Path(storage_path)}"
        logger.info("Using SQLite storage: %s", storage)

    # ------------------------------------------------------------------
    # Create / resume study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # ------------------------------------------------------------------
    # Build objective
    # ------------------------------------------------------------------
    train_end_idx = int(len(featured_df) * 0.8)
    objective = make_objective(
        featured_df=featured_df,
        algo=algo,
        train_end_idx=train_end_idx,
        n_timesteps=n_timesteps_per_trial,
    )

    # ------------------------------------------------------------------
    # Run optimisation
    # ------------------------------------------------------------------
    logger.info(
        "Starting Optuna study '%s' | algo=%s | n_trials=%d | n_jobs=%d",
        study_name,
        algo.upper(),
        n_trials,
        n_jobs,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # Log each completed trial
    # ------------------------------------------------------------------
    for trial in study.trials:
        if trial.value is not None:
            logger.info(
                "Trial %d: Sharpe=%.3f | params=%s",
                trial.number,
                trial.value,
                trial.params,
            )

    # ------------------------------------------------------------------
    # Assemble return value
    # ------------------------------------------------------------------
    base_config = TrainingConfig(algo=algo.upper())
    result = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "best_config": best_config_from_study(study, algo, base_config),
    }

    logger.info(
        "Tuning complete. Best Sharpe=%.4f | best_params=%s",
        study.best_value,
        study.best_params,
    )
    return result


def best_config_from_study(
    study: "optuna.Study",
    algo: str,
    base_config: "TrainingConfig",
) -> "TrainingConfig":
    """
    Apply ``study.best_params`` onto a copy of *base_config*.

    Only parameters that correspond to fields on
    :class:`~trading_bot.models.train.TrainingConfig` are applied.
    ``net_arch`` is intentionally skipped because it is not a
    ``TrainingConfig`` field (it lives in ``policy_kwargs``).

    Parameters
    ----------
    study:
        Completed Optuna study.
    algo:
        Algorithm name (used to set ``TrainingConfig.algo``).
    base_config:
        Base configuration — tuned params are layered on top of this.

    Returns
    -------
    TrainingConfig
        Updated configuration with best hyperparameters applied.
    """
    config_fields = {f.name for f in dataclasses.fields(base_config)}

    applicable: dict = {}
    for key, value in study.best_params.items():
        if key == "net_arch":
            continue  # policy_kwargs concern — not a TrainingConfig field
        if key in config_fields:
            applicable[key] = value

    # Always set algo from the argument so the config stays consistent
    applicable["algo"] = algo.upper()

    return dataclasses.replace(base_config, **applicable)


def plot_optimization_history(study: "optuna.Study", output_path: Path) -> None:
    """
    Save an optimisation-history plot for *study* to *output_path*.

    Parameters
    ----------
    study:
        Completed (or in-progress) Optuna study.
    output_path:
        File path (e.g. ``Path("reports/optuna_history.png")``).
        Parent directory must exist.
    """
    import matplotlib.pyplot as plt  # lazy
    import optuna.visualization.matplotlib as optuna_vis  # lazy

    output_path = Path(output_path)

    fig = optuna_vis.plot_optimization_history(study)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved optimization history plot to %s", output_path)
