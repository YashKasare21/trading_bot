"""
Walk-forward validation for RL trading agents.

Splits a feature DataFrame into sequential train/test windows, trains a fresh
agent per window, and aggregates out-of-sample performance metrics to assess
strategy robustness.

Walk-forward is preferred over a single train/test split because it simulates
the live trading workflow: the model is periodically retrained as new data
arrives and evaluated only on unseen future data.

Lazy imports (torch, stable_baselines3, optuna) are NEVER at module level.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """Describes the date boundaries of a single walk-forward fold."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class WalkForwardResult:
    """Aggregated results from a full walk-forward validation run."""

    windows: list[WalkForwardWindow]
    window_results: list[dict]          # metrics dict per window (from compute_all)
    aggregate_metrics: dict             # mean + std of each metric across windows
    portfolio_values_oos: pd.Series     # concatenated out-of-sample portfolio values
    is_viable: bool                     # True if mean Sharpe > 0.5 across all windows


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


def generate_windows(
    start: date,
    end: date,
    min_train_years: int = 2,
    test_months: int = 3,
    expanding: bool = True,
) -> list[WalkForwardWindow]:
    """
    Generate walk-forward validation windows over a date range.

    Two modes are supported:
    - **Expanding** (default/recommended): the training window always begins at
      ``start`` and grows by ``test_months`` each iteration. The model sees more
      history every fold, mimicking how a live system accumulates data.
    - **Rolling**: the training window is a fixed-length sliding window of
      ``min_train_years`` years, suitable when you want recency-weighted training.

    Key invariant: ``train_end <= test_start`` — zero look-ahead guaranteed.

    Parameters
    ----------
    start:
        Earliest date in the dataset.
    end:
        Latest date in the dataset (exclusive upper bound for testing).
    min_train_years:
        Minimum training history in years before the first test period begins.
    test_months:
        Length of each out-of-sample test period in months.
    expanding:
        If True use an expanding training window; if False use a rolling window.

    Returns
    -------
    list[WalkForwardWindow]
        Ordered list of windows, each with non-overlapping test periods that
        together tile the post-warmup portion of the dataset.
    """
    from dateutil.relativedelta import relativedelta  # lazy: already a yfinance dep

    windows: list[WalkForwardWindow] = []
    window_id = 1

    # First test period starts after the minimum training history
    test_start = start + relativedelta(years=min_train_years)

    while test_start < end:
        if expanding:
            train_start = start
        else:
            # Rolling: fixed-length train window
            train_start = test_start - relativedelta(months=min_train_years * 12)

        train_end = test_start  # exclusive: test immediately follows train
        test_end_candidate = test_start + relativedelta(months=test_months)
        test_end = min(test_end_candidate, end)

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        window_id += 1
        test_start = test_start + relativedelta(months=test_months)

    return windows


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------


def run_walk_forward(
    featured_df: pd.DataFrame,
    config: "TrainingConfig",  # noqa: F821 — resolved at runtime
    windows: list[WalkForwardWindow] | None = None,
    min_train_years: int = 2,
    test_months: int = 3,
) -> WalkForwardResult:
    """
    Execute a full walk-forward validation run.

    For each window:
    1. Slices ``featured_df`` into train and test portions by date.
    2. Creates a fresh :class:`~trading_bot.models.train.TrainingConfig` for
       the window (same hyperparameters, updated date range and run name).
    3. Trains a new agent via :func:`~trading_bot.models.train.train_agent`.
    4. Collects ``final_metrics`` from the training result as window metrics.
       (Full OOS inference requires BacktestEngine and is deferred to Phase 4;
       placeholder OOS portfolio series are returned for now.)
    5. Assembles aggregate statistics and an overall viability flag.

    Parameters
    ----------
    featured_df:
        Feature-engineered DataFrame indexed by datetime. Must cover the full
        date range needed by all windows.
    config:
        Base :class:`~trading_bot.models.train.TrainingConfig` supplying
        hyperparameters and the algorithm choice. ``train_start``/``train_end``
        will be overridden per window.
    windows:
        Pre-computed list of :class:`WalkForwardWindow` instances. When None,
        windows are generated automatically from the DataFrame's index range.
    min_train_years:
        Passed to :func:`generate_windows` when ``windows`` is None.
    test_months:
        Passed to :func:`generate_windows` when ``windows`` is None.

    Returns
    -------
    WalkForwardResult
        Contains per-window metrics, aggregate statistics, concatenated OOS
        portfolio series (placeholder), and a viability flag.
    """
    from trading_bot.models.train import TrainingConfig, train_agent  # lazy

    # ------------------------------------------------------------------
    # 1. Generate windows if not provided
    # ------------------------------------------------------------------
    if windows is None:
        start_date = featured_df.index[0].date()
        end_date = featured_df.index[-1].date()
        windows = generate_windows(
            start=start_date,
            end=end_date,
            min_train_years=min_train_years,
            test_months=test_months,
            expanding=True,
        )

    if not windows:
        raise ValueError(
            "No walk-forward windows could be generated from the provided data. "
            "Ensure featured_df covers at least min_train_years of history."
        )

    logger.info(
        "Starting walk-forward validation: %d windows, algo=%s",
        len(windows),
        config.algo,
    )

    window_results: list[dict] = []
    oos_series_list: list[pd.Series] = []

    # ------------------------------------------------------------------
    # 2. Iterate windows
    # ------------------------------------------------------------------
    for w in windows:
        # Slice to train / test date ranges
        train_slice = featured_df.loc[
            str(w.train_start): str(w.train_end)
        ].copy()
        test_slice = featured_df.loc[
            str(w.test_start): str(w.test_end)
        ].copy()

        if train_slice.empty:
            logger.warning(
                "Window %d: empty train slice (%s → %s); skipping.",
                w.window_id,
                w.train_start,
                w.train_end,
            )
            continue

        # 3. Build a window-specific config
        window_config: TrainingConfig = dataclasses.replace(
            config,
            train_start=w.train_start,
            train_end=w.train_end,
            run_name=f"{config.algo.lower()}_wf_window_{w.window_id}",
        )

        # 4. Train
        logger.info(
            "Window %d/%d: training on %s → %s (%d rows), test %s → %s (%d rows)",
            w.window_id,
            len(windows),
            w.train_start,
            w.train_end,
            len(train_slice),
            w.test_start,
            w.test_end,
            len(test_slice),
        )

        result = train_agent(window_config, train_slice)
        metrics = result["final_metrics"]

        # 5. Placeholder OOS portfolio series (real OOS inference deferred to Phase 4)
        oos_placeholder = pd.Series(
            dtype=float,
            name=f"window_{w.window_id}",
        )
        oos_series_list.append(oos_placeholder)

        # 6. Collect
        window_results.append(metrics)

        logger.info(
            "Window %d: Train %s→%s | Test %s→%s | Sharpe=%.3f",
            w.window_id,
            w.train_start,
            w.train_end,
            w.test_start,
            w.test_end,
            metrics.get("sharpe", float("nan")),
        )

    if not window_results:
        raise RuntimeError(
            "All walk-forward windows were skipped (empty slices). "
            "Check that featured_df is correctly indexed and covers the window dates."
        )

    # ------------------------------------------------------------------
    # 7. Aggregate metrics
    # ------------------------------------------------------------------
    def _safe_mean(values: list[float]) -> float:
        clean = [v for v in values if v is not None and v == v]  # drop NaN
        return statistics.mean(clean) if clean else 0.0

    def _safe_std(values: list[float]) -> float:
        clean = [v for v in values if v is not None and v == v]
        return statistics.stdev(clean) if len(clean) >= 2 else 0.0

    sharpe_vals = [m.get("sharpe", 0.0) for m in window_results]
    cagr_vals = [m.get("cagr", 0.0) for m in window_results]
    dd_vals = [m.get("max_drawdown", 0.0) for m in window_results]
    wr_vals = [m.get("win_rate", 0.0) for m in window_results]

    aggregate_metrics = {
        "mean_sharpe": _safe_mean(sharpe_vals),
        "std_sharpe": _safe_std(sharpe_vals),
        "mean_cagr": _safe_mean(cagr_vals),
        "std_cagr": _safe_std(cagr_vals),
        "mean_max_drawdown": _safe_mean(dd_vals),
        "std_max_drawdown": _safe_std(dd_vals),
        "mean_win_rate": _safe_mean(wr_vals),
        "std_win_rate": _safe_std(wr_vals),
        "n_windows": len(windows),
    }

    # 8. Concatenated OOS portfolio series (placeholders for now)
    non_empty = [s for s in oos_series_list if not s.empty]
    if non_empty:
        portfolio_values_oos = pd.concat(non_empty)
    else:
        portfolio_values_oos = pd.Series(dtype=float, name="oos_portfolio")

    # 9. Viability: mean Sharpe > 0.5
    is_viable = aggregate_metrics["mean_sharpe"] > 0.5

    logger.info(
        "Walk-forward complete: mean_sharpe=%.3f, viable=%s",
        aggregate_metrics["mean_sharpe"],
        is_viable,
    )

    return WalkForwardResult(
        windows=windows,
        window_results=window_results,
        aggregate_metrics=aggregate_metrics,
        portfolio_values_oos=portfolio_values_oos,
        is_viable=is_viable,
    )


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------


def summarise(result: WalkForwardResult) -> str:
    """
    Return a human-readable walk-forward validation summary.

    No third-party formatting libraries are used — output is constructed
    entirely with f-strings for portability.

    Parameters
    ----------
    result:
        Completed :class:`WalkForwardResult` from :func:`run_walk_forward`.

    Returns
    -------
    str
        Multi-line summary including aggregate statistics and a per-window
        breakdown table.
    """
    agg = result.aggregate_metrics
    n = agg["n_windows"]
    viable_str = "YES" if result.is_viable else "NO"

    lines: list[str] = [
        "Walk-Forward Validation Summary",
        "================================",
        f"Windows: {n}",
        f"Mean Sharpe:      {agg['mean_sharpe']:.2f} \u00b1 {agg['std_sharpe']:.2f}",
        f"Mean CAGR:       {agg['mean_cagr']:.1f}% \u00b1 {agg['std_cagr']:.1f}%",
        f"Mean Max DD:     {agg['mean_max_drawdown']:.1f}% \u00b1 {agg['std_max_drawdown']:.1f}%",
        f"Mean Win Rate:   {agg['mean_win_rate']:.1f}% \u00b1 {agg['std_win_rate']:.1f}%",
        f"Overall viable:  {viable_str} (mean Sharpe > 0.5)",
        "",
        "Per-window breakdown:",
        f"{'Window':<7}| {'Train Period':<27}| {'Test Period':<16}| {'Sharpe':<8}| {'CAGR':<8}| {'MaxDD'}",
        f"{'-------':<7}|{'---------------------------':<28}|{'----------------':<17}|{'--------':<9}|{'--------':<9}|{'-------'}",
    ]

    for w, m in zip(result.windows, result.window_results):
        sharpe = m.get("sharpe", float("nan"))
        cagr_val = m.get("cagr", float("nan"))
        dd_val = m.get("max_drawdown", float("nan"))
        train_period = f"{w.train_start} \u2192 {w.train_end}"
        test_period = f"{w.test_start} \u2192 {w.test_end}"
        cagr_str = f"{cagr_val:.1f}%"
        dd_str = f"{dd_val:.1f}%"
        lines.append(
            f"{w.window_id:<7}| {train_period:<27}| {test_period:<16}| "
            f"{sharpe:<8.2f}| {cagr_str:<8}| {dd_str}"
        )

    return "\n".join(lines)
