"""
Tests for training module, walk-forward validation, and model registry.

Heavy dependencies (torch, stable_baselines3, optuna) are mocked throughout.
No real RL training occurs in these tests.
"""

from __future__ import annotations

import dataclasses
import datetime
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_record(
    run_name: str = "sac_nsei_20240101_1200",
    algo: str = "SAC",
    ticker: str = "^NSEI",
    sharpe: float = 1.2,
    is_production: bool = False,
):
    from trading_bot.models.registry import ModelRecord

    return ModelRecord(
        run_name=run_name,
        algo=algo,
        ticker=ticker,
        train_start="2018-01-01",
        train_end="2022-12-31",
        total_timesteps=100_000,
        model_path=f"data/models/{run_name}/sac_final.zip",
        vec_norm_path=f"data/models/{run_name}/vec_normalize_sac.pkl",
        pipeline_path="data/models/pipeline.joblib",
        feature_names=["close", "rsi", "sentiment_score"],
        hyperparams={"lr": 3e-4, "gamma": 0.99},
        train_metrics={"sharpe": sharpe, "cagr": 18.5, "max_drawdown": 12.0},
        val_metrics={"mean_sharpe": sharpe * 0.8},
        created_at=datetime.datetime.now().isoformat(),
        notes="test record",
        is_production=is_production,
    )


def _make_registry(tmp_path: Path):
    from trading_bot.models.registry import ModelRegistry

    return ModelRegistry(tmp_path / "registry.json")


def _make_featured_df(n: int = 100) -> pd.DataFrame:
    """Tiny synthetic featured DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed=42)
    close = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, n).astype(float),
            "rsi": rng.uniform(20, 80, n),
            "sentiment_score": rng.uniform(-1, 1, n),
        },
        index=dates,
    )
    return df


# ── Task 1 — TrainingConfig ───────────────────────────────────────────────────


def test_training_config_auto_run_name():
    """run_name is auto-generated when left empty, and must contain the algo name."""
    from trading_bot.models.train import TrainingConfig

    cfg = TrainingConfig(algo="SAC")
    assert cfg.run_name != ""
    assert "sac" in cfg.run_name.lower()


def test_training_config_custom_run_name():
    """A provided run_name must be preserved exactly."""
    from trading_bot.models.train import TrainingConfig

    cfg = TrainingConfig(algo="PPO", run_name="my_custom_run")
    assert cfg.run_name == "my_custom_run"


def test_training_config_save_dir_coerced_to_path():
    """Passing save_dir as a string must be coerced to Path."""
    from trading_bot.models.train import TrainingConfig

    cfg = TrainingConfig(save_dir="some/dir")  # type: ignore[arg-type]
    assert isinstance(cfg.save_dir, Path)


# ── Task 3 — generate_windows ────────────────────────────────────────────────


def test_generate_windows_count():
    """
    2018-01 → 2024-12, min_train_years=2, test_months=3 (expanding).
    Expected: (2024-12 - 2020-01) / 3 months = ~20 windows.
    At minimum 16 windows (conservative bound).
    """
    from trading_bot.models.walk_forward import generate_windows

    windows = generate_windows(
        start=date(2018, 1, 1),
        end=date(2024, 12, 31),
        min_train_years=2,
        test_months=3,
        expanding=True,
    )
    assert len(windows) >= 16, f"Expected ≥16 windows, got {len(windows)}"


def test_generate_windows_no_overlap():
    """No two test windows should overlap in time."""
    from trading_bot.models.walk_forward import generate_windows

    windows = generate_windows(
        start=date(2018, 1, 1),
        end=date(2024, 12, 31),
        min_train_years=2,
        test_months=3,
    )
    for i in range(len(windows) - 1):
        assert windows[i].test_end <= windows[i + 1].test_start, (
            f"Overlap between window {windows[i].window_id} and {windows[i+1].window_id}"
        )


def test_generate_windows_no_lookahead():
    """train_end must be <= test_start for every window (zero look-ahead)."""
    from trading_bot.models.walk_forward import generate_windows

    windows = generate_windows(
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
        min_train_years=2,
        test_months=3,
    )
    for w in windows:
        assert w.train_end <= w.test_start, (
            f"Window {w.window_id}: train_end={w.train_end} > test_start={w.test_start}"
        )


def test_generate_windows_expanding():
    """In expanding mode, every window's train_start must equal the global start."""
    from trading_bot.models.walk_forward import generate_windows

    start = date(2018, 1, 1)
    windows = generate_windows(
        start=start,
        end=date(2023, 12, 31),
        min_train_years=2,
        test_months=3,
        expanding=True,
    )
    for w in windows:
        assert w.train_start == start, (
            f"Window {w.window_id}: train_start={w.train_start} != global start={start}"
        )


def test_generate_windows_rolling():
    """
    In rolling mode, every window's train duration must be the same
    (fixed-length window).  We check that the train length in days is
    approximately constant (within ±5 days to allow for month-boundary rounding).
    """
    from trading_bot.models.walk_forward import generate_windows

    windows = generate_windows(
        start=date(2018, 1, 1),
        end=date(2024, 12, 31),
        min_train_years=2,
        test_months=3,
        expanding=False,
    )
    durations = [
        (w.train_end - w.train_start).days for w in windows
    ]
    min_d, max_d = min(durations), max(durations)
    assert max_d - min_d <= 5, (
        f"Rolling window train durations vary too much: min={min_d}, max={max_d}"
    )


# ── Task 3 — WalkForwardResult OOS continuity ─────────────────────────────────


def test_walk_forward_result_oos_continuity():
    """
    When multiple non-empty OOS series exist, the concatenated
    portfolio_values_oos must have no gaps (index is monotonic).

    We test this by directly constructing a WalkForwardResult.
    """
    from trading_bot.models.walk_forward import WalkForwardResult, WalkForwardWindow

    # Two non-overlapping portfolio series
    idx1 = pd.date_range("2021-01-01", periods=60, freq="B")
    idx2 = pd.date_range("2021-04-01", periods=60, freq="B")
    s1 = pd.Series(100_000.0 + np.arange(60) * 100, index=idx1, name="oos")
    s2 = pd.Series(106_000.0 + np.arange(60) * 100, index=idx2, name="oos")
    oos = pd.concat([s1, s2])

    windows = [
        WalkForwardWindow(1, date(2018, 1, 1), date(2021, 1, 1), date(2021, 1, 1), date(2021, 4, 1)),
        WalkForwardWindow(2, date(2018, 1, 1), date(2021, 4, 1), date(2021, 4, 1), date(2021, 7, 1)),
    ]
    result = WalkForwardResult(
        windows=windows,
        window_results=[{"sharpe": 1.2}, {"sharpe": 0.9}],
        aggregate_metrics={"mean_sharpe": 1.05, "std_sharpe": 0.15,
                           "mean_cagr": 15.0, "std_cagr": 3.0,
                           "mean_max_drawdown": 10.0, "std_max_drawdown": 2.0,
                           "mean_win_rate": 55.0, "std_win_rate": 2.0,
                           "n_windows": 2},
        portfolio_values_oos=oos,
        is_viable=True,
    )

    assert result.portfolio_values_oos.index.is_monotonic_increasing
    assert len(result.portfolio_values_oos) == 120


# ── Task 4 — ModelRegistry ────────────────────────────────────────────────────


def test_registry_register_and_retrieve(tmp_path):
    """Registering a record and listing by ticker must return exactly that record."""
    registry = _make_registry(tmp_path)
    record = _make_record(run_name="sac_test_001", ticker="^NSEI")
    registry.register(record)

    results = registry.list_models(ticker="^NSEI")
    assert len(results) == 1
    assert results[0].run_name == "sac_test_001"


def test_registry_register_update_in_place(tmp_path):
    """Re-registering with the same run_name must update, not duplicate."""
    registry = _make_registry(tmp_path)
    registry.register(_make_record(run_name="sac_test_002", sharpe=0.9))
    updated = _make_record(run_name="sac_test_002", sharpe=1.5)
    registry.register(dataclasses.replace(updated, notes="updated"))

    results = registry.list_models()
    assert len(results) == 1
    assert results[0].train_metrics["sharpe"] == pytest.approx(1.5)


def test_registry_promote_demotes_others(tmp_path):
    """Promoting one model must set all other (ticker, algo) models to is_production=False."""
    registry = _make_registry(tmp_path)
    registry.register(_make_record(run_name="sac_v1", ticker="^NSEI", algo="SAC"))
    registry.register(_make_record(run_name="sac_v2", ticker="^NSEI", algo="SAC"))

    registry.promote_to_production("sac_v2")

    records = {r.run_name: r for r in registry.list_models()}
    assert records["sac_v2"].is_production is True
    assert records["sac_v1"].is_production is False


def test_registry_promote_unknown_raises(tmp_path):
    """Promoting a non-existent run_name must raise KeyError."""
    registry = _make_registry(tmp_path)
    with pytest.raises(KeyError, match="ghost_run"):
        registry.promote_to_production("ghost_run")


def test_registry_best_model_by_sharpe(tmp_path):
    """best_model must return the record with the highest train Sharpe."""
    registry = _make_registry(tmp_path)
    registry.register(_make_record(run_name="sac_a", ticker="^NSEI", sharpe=0.8))
    registry.register(_make_record(run_name="sac_b", ticker="^NSEI", sharpe=1.9))
    registry.register(_make_record(run_name="sac_c", ticker="^NSEI", sharpe=1.1))

    best = registry.best_model(ticker="^NSEI", metric="sharpe")
    assert best is not None
    assert best.run_name == "sac_b"


def test_registry_list_filtered_by_ticker(tmp_path):
    """list_models(ticker=...) must return only records matching that ticker."""
    registry = _make_registry(tmp_path)
    registry.register(_make_record(run_name="nsei_1", ticker="^NSEI"))
    registry.register(_make_record(run_name="reliance_1", ticker="RELIANCE.NS"))
    registry.register(_make_record(run_name="nsei_2", ticker="^NSEI"))

    nsei_records = registry.list_models(ticker="^NSEI")
    assert len(nsei_records) == 2
    assert all(r.ticker == "^NSEI" for r in nsei_records)


def test_registry_atomic_write_survives_reload(tmp_path):
    """Records written to disk must be fully recoverable after reloading the registry."""
    r1 = _make_registry(tmp_path)
    r1.register(_make_record(run_name="persist_test", ticker="^NSEI"))

    # Reload from same file
    r2 = _make_registry(tmp_path)
    records = r2.list_models()
    assert len(records) == 1
    assert records[0].run_name == "persist_test"
