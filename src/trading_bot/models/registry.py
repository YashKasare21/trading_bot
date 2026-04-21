"""
Local model registry.

Every trained model gets a :class:`ModelRecord` entry persisted in a single
``registry.json`` file.  Writes are atomic (write to ``.tmp`` then ``os.replace``)
so a Colab disconnect during a write cannot corrupt the file.

No MLflow / external service dependency — plain JSON, works offline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("data/models/registry.json")


@dataclass
class ModelRecord:
    """Complete metadata for one trained model run."""

    run_name: str
    algo: str
    ticker: str
    train_start: str        # ISO date string e.g. "2018-01-01"
    train_end: str
    total_timesteps: int
    model_path: str         # relative to project root (or absolute on Colab drive)
    vec_norm_path: str
    pipeline_path: str
    feature_names: list[str]
    hyperparams: dict
    train_metrics: dict     # metrics on training period (from compute_all)
    val_metrics: dict       # metrics on held-out / walk-forward validation period
    created_at: str         # ISO datetime string
    notes: str = ""
    is_production: bool = False  # at most one production model per (ticker, algo)


class ModelRegistry:
    """
    Manages a ``registry.json`` file that tracks all trained models.

    All mutating methods (register, promote_to_production) are atomic:
    changes are written to a ``.tmp`` sibling file and then renamed into
    place, so a crash mid-write leaves the previous registry intact.

    Parameters
    ----------
    registry_path:
        Path to the JSON file.  Created on first write if it does not exist.
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(self, record: ModelRecord) -> None:
        """
        Add or update a model record, then persist to disk.

        If a record with the same ``run_name`` already exists it is replaced
        in-place (preserving order).

        Parameters
        ----------
        record:
            Completed :class:`ModelRecord` from a training run.
        """
        records = self._load_all()
        # Replace existing or append
        for i, r in enumerate(records):
            if r.run_name == record.run_name:
                records[i] = record
                break
        else:
            records.append(record)

        self._save_all(records)
        logger.info("Registered model: %s", record.run_name)

    def get_production_model(self, ticker: str, algo: str) -> ModelRecord | None:
        """
        Return the single production model for a ``(ticker, algo)`` pair.

        Returns ``None`` if no production model has been designated yet.

        Parameters
        ----------
        ticker:
            NSE ticker symbol (e.g. ``"^NSEI"``).
        algo:
            Algorithm name (e.g. ``"SAC"``).
        """
        for r in self._load_all():
            if r.ticker == ticker and r.algo.upper() == algo.upper() and r.is_production:
                return r
        return None

    def promote_to_production(self, run_name: str) -> None:
        """
        Set ``is_production=True`` for *run_name* and demote all other models
        for the same ``(ticker, algo)`` pair.

        Parameters
        ----------
        run_name:
            The ``run_name`` of the model to promote.

        Raises
        ------
        KeyError
            If no record with *run_name* exists in the registry.
        """
        records = self._load_all()
        target = next((r for r in records if r.run_name == run_name), None)
        if target is None:
            raise KeyError(f"No model with run_name='{run_name}' found in registry.")

        ticker, algo = target.ticker, target.algo.upper()

        updated: list[ModelRecord] = []
        for r in records:
            if r.ticker == ticker and r.algo.upper() == algo:
                # Demote everyone in this (ticker, algo) group first
                r = ModelRecord(**{**asdict(r), "is_production": False})
            updated.append(r)

        # Now promote the target
        updated = [
            ModelRecord(**{**asdict(r), "is_production": True})
            if r.run_name == run_name
            else r
            for r in updated
        ]

        self._save_all(updated)
        logger.info("Promoted '%s' to production (%s / %s)", run_name, ticker, algo)

    def list_models(self, ticker: str = "", algo: str = "") -> list[ModelRecord]:
        """
        Return all records, optionally filtered by ticker and/or algo.

        Parameters
        ----------
        ticker:
            Filter to models matching this ticker.  Empty string means no filter.
        algo:
            Filter to models matching this algo (case-insensitive).  Empty string
            means no filter.

        Returns
        -------
        list[ModelRecord]
            Matching records, ordered by ``created_at`` ascending.
        """
        records = self._load_all()
        if ticker:
            records = [r for r in records if r.ticker == ticker]
        if algo:
            records = [r for r in records if r.algo.upper() == algo.upper()]
        return records

    def compare_metrics(self, run_names: list[str]) -> pd.DataFrame:
        """
        Build a side-by-side comparison DataFrame for the specified runs.

        Parameters
        ----------
        run_names:
            List of ``run_name`` strings to compare.

        Returns
        -------
        pd.DataFrame
            Rows are metric names, columns are run names.  ``train_metrics``
            and ``val_metrics`` are merged into a flat dict per model, with
            ``train_`` and ``val_`` prefixes respectively.
        """
        all_records = {r.run_name: r for r in self._load_all()}
        flat: dict[str, dict] = {}

        for name in run_names:
            record = all_records.get(name)
            if record is None:
                logger.warning("run_name '%s' not found in registry; skipping.", name)
                continue
            combined: dict = {
                **{f"train_{k}": v for k, v in record.train_metrics.items()},
                **{f"val_{k}": v for k, v in record.val_metrics.items()},
            }
            flat[name] = combined

        if not flat:
            return pd.DataFrame()

        return pd.DataFrame(flat)

    def best_model(self, ticker: str, metric: str = "sharpe") -> ModelRecord | None:
        """
        Return the record with the highest value for *metric* in its
        ``train_metrics`` dict.

        Parameters
        ----------
        ticker:
            Filter to models for this ticker.
        metric:
            Metric key to rank by (default: ``"sharpe"``).

        Returns
        -------
        ModelRecord | None
            Best model, or ``None`` if the registry is empty / no ticker match.
        """
        candidates = self.list_models(ticker=ticker)
        if not candidates:
            return None

        def _score(r: ModelRecord) -> float:
            return float(r.train_metrics.get(metric, float("-inf")))

        return max(candidates, key=_score)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_all(self) -> list[ModelRecord]:
        """Load all records from the JSON file, or return [] if it doesn't exist."""
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [ModelRecord(**entry) for entry in raw]
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.error("Failed to parse registry at %s: %s", self._path, exc)
            return []

    def _save_all(self, records: list[ModelRecord]) -> None:
        """Atomically write *records* to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self._path.with_suffix(".json.tmp")
        serialised = [asdict(r) for r in records]
        tmp_path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")

        # Atomic rename — on POSIX this is guaranteed atomic
        os.replace(tmp_path, self._path)
        logger.debug("Registry written to %s (%d records)", self._path, len(records))
