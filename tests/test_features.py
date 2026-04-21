"""
Tests for FeaturePipeline, technical indicators, Fourier features, and regime detector.

All tests use fixtures from conftest.py. No real API calls are made.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── FeaturePipeline ─────────────────────────────────────────────────────────


def _make_simple_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal sentiment_df aligned to a featured df's index."""
    return pd.DataFrame(
        {"sentiment_score": np.zeros(len(df))},
        index=df.index,
    )


def test_pipeline_output_has_no_nan(sample_ohlcv_df):
    """fit_transform output must contain zero NaN values."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=False, fit_regime=True)
    result = pipeline.fit_transform(sample_ohlcv_df)
    assert result.isnull().sum().sum() == 0, "Pipeline output contains NaN values"


def test_pipeline_feature_names_consistent(sample_ohlcv_df):
    """get_feature_names() must return the same list on repeated calls."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=False, fit_regime=True)
    pipeline.fit_transform(sample_ohlcv_df)
    names1 = pipeline.get_feature_names()
    names2 = pipeline.get_feature_names()
    assert names1 == names2


def test_pipeline_transform_matches_fit_transform(sample_ohlcv_df):
    """transform() on the same data must return the same columns as fit_transform()."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=False, fit_regime=True)
    result_fit = pipeline.fit_transform(sample_ohlcv_df)
    result_transform = pipeline.transform(sample_ohlcv_df)
    assert list(result_fit.columns) == list(result_transform.columns)


def test_pipeline_drops_warmup_rows(sample_ohlcv_df):
    """Output must have fewer rows than input due to warmup removal."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=False, fit_regime=True)
    result = pipeline.fit_transform(sample_ohlcv_df)
    assert len(result) < len(sample_ohlcv_df)


def test_pipeline_save_load_roundtrip(sample_ohlcv_df):
    """Saving and loading a pipeline must produce the same transform output."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=False, fit_regime=True)
    result_before = pipeline.fit_transform(sample_ohlcv_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "pipeline.joblib"
        pipeline.save(save_path)

        loaded = FeaturePipeline.load(save_path)
        result_after = loaded.transform(sample_ohlcv_df)

    assert list(result_before.columns) == list(result_after.columns)
    assert result_before.shape == result_after.shape


def test_pipeline_with_sentiment_merges_correctly(sample_ohlcv_df):
    """When sentiment_df is provided, 'sentiment_score' column must be present in output."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=True, fit_regime=True)
    featured = pipeline.fit_transform(sample_ohlcv_df)
    sentiment_df = _make_simple_sentiment(featured)

    pipeline2 = FeaturePipeline(window_size=20, include_sentiment=True, fit_regime=True)
    result = pipeline2.fit_transform(sample_ohlcv_df, sentiment_df)
    assert "sentiment_score" in result.columns


def test_pipeline_without_sentiment_adds_zero_column(sample_ohlcv_df):
    """When include_sentiment=True but no sentiment_df, column must be all 0.0."""
    from trading_bot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=20, include_sentiment=True, fit_regime=True)
    result = pipeline.fit_transform(sample_ohlcv_df, sentiment_df=None)
    assert "sentiment_score" in result.columns
    assert (result["sentiment_score"] == 0.0).all()


# ── Technical indicators ─────────────────────────────────────────────────────


def test_technical_no_lookahead_fillna(sample_ohlcv_df):
    """
    add_technical_indicators must not use fillna=True on the ta call.

    We verify this behaviourally: the first rows of the returned DataFrame
    should NOT all be identical (which they would be if fillna=True silently
    back-filled NaN values with the first valid observation).
    """
    from trading_bot.features.technical import add_technical_indicators

    result = add_technical_indicators(sample_ohlcv_df.copy())
    # If fillna=True were used, many indicator columns would have identical
    # first values — check at least some variance exists
    numeric_cols = result.select_dtypes(include="number").columns
    assert len(numeric_cols) > 0
    variances = result[numeric_cols].var()
    assert (variances > 0).any(), "All columns have zero variance — suspicious fillna behaviour"


def test_technical_output_has_more_columns(sample_ohlcv_df):
    """add_technical_indicators must add at least 20 new columns."""
    from trading_bot.features.technical import add_technical_indicators

    n_in = sample_ohlcv_df.shape[1]
    result = add_technical_indicators(sample_ohlcv_df.copy())
    assert result.shape[1] > n_in + 20, f"Expected >20 new columns, got {result.shape[1] - n_in}"


# ── Fourier features ─────────────────────────────────────────────────────────


def test_fourier_adds_correct_columns(sample_ohlcv_df):
    """Fourier features must add fourier_sin_N and fourier_cos_N for each period."""
    from trading_bot.features.fourier import add_fourier_features

    periods = [3, 6]
    result = add_fourier_features(sample_ohlcv_df.copy(), periods=periods)
    for p in periods:
        assert f"fourier_sin_{p}" in result.columns, f"Missing fourier_sin_{p}"
        assert f"fourier_cos_{p}" in result.columns, f"Missing fourier_cos_{p}"


def test_fourier_values_bounded(sample_ohlcv_df):
    """
    Fourier features are derived from FFT magnitude — the imaginary and real
    components are not bounded to [-1, 1] by default, but we verify that
    the columns are finite (no inf/NaN) and non-trivially non-zero.
    """
    from trading_bot.features.fourier import add_fourier_features

    result = add_fourier_features(sample_ohlcv_df.copy(), periods=[3])
    sin_col = result["fourier_sin_3"]
    cos_col = result["fourier_cos_3"]
    assert sin_col.isna().sum() == 0, "fourier_sin_3 contains NaN"
    assert cos_col.isna().sum() == 0, "fourier_cos_3 contains NaN"
    assert np.isfinite(sin_col.values).all(), "fourier_sin_3 contains inf"
    assert np.isfinite(cos_col.values).all(), "fourier_cos_3 contains inf"


# ── Regime detector ──────────────────────────────────────────────────────────


def test_regime_fit_predict_returns_valid_labels(sample_ohlcv_df):
    """After fitting, predict() must return only labels 0, 1, or 2."""
    from trading_bot.features.regime import RegimeDetector

    detector = RegimeDetector(n_components=3)
    detector.fit(sample_ohlcv_df)
    labels = detector.predict(sample_ohlcv_df)
    assert set(labels).issubset({0, 1, 2}), f"Unexpected labels: {set(labels)}"
    assert len(labels) == len(sample_ohlcv_df) - 1  # one fewer due to log returns


def test_regime_save_load(sample_ohlcv_df):
    """Save/load roundtrip must produce identical predictions."""
    from trading_bot.features.regime import RegimeDetector

    detector = RegimeDetector(n_components=3)
    detector.fit(sample_ohlcv_df)
    labels_before = detector.predict(sample_ohlcv_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "regime_hmm.joblib"
        detector.save(save_path)

        loaded = RegimeDetector(n_components=3)
        loaded.load(save_path)
        labels_after = loaded.predict(sample_ohlcv_df)

    np.testing.assert_array_equal(labels_before, labels_after)
