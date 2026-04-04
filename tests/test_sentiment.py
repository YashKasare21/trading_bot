"""
Tests for SentimentAnalyzer.

All Gemini API and NewsAPI calls are mocked — no real HTTP requests.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_gemini_response(label: str) -> MagicMock:
    """Build a mock requests.Response that returns *label* from Gemini."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": label}]}}
        ]
    }
    return mock_resp


def _make_analyzer():
    """Return a SentimentAnalyzer with an in-memory DataStore."""
    import tempfile
    from pathlib import Path

    from trading_bot.data.sentiment import SentimentAnalyzer
    from trading_bot.data.store import DataStore

    tmpdir = tempfile.mkdtemp()
    store = DataStore(db_path=Path(tmpdir) / "test.db")
    return SentimentAnalyzer(store=store)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_score_headline_positive():
    """Gemini returning 'Positive' should give score == 1.0."""
    analyzer = _make_analyzer()
    with patch("requests.post", return_value=_mock_gemini_response("Positive")):
        result = analyzer.score_headline("Reliance beats earnings estimates")
    assert result["label"] == "Positive"
    assert result["score"] == pytest.approx(1.0)


def test_score_headline_negative():
    """Gemini returning 'Negative' should give score == -1.0."""
    analyzer = _make_analyzer()
    with patch("requests.post", return_value=_mock_gemini_response("Negative")):
        result = analyzer.score_headline("Sensex tanks 800 points on FII selling")
    assert result["label"] == "Negative"
    assert result["score"] == pytest.approx(-1.0)


def test_score_headline_neutral():
    """Gemini returning 'Neutral' should give score == 0.0."""
    analyzer = _make_analyzer()
    with patch("requests.post", return_value=_mock_gemini_response("Neutral")):
        result = analyzer.score_headline("Market opens flat ahead of RBI policy")
    assert result["label"] == "Neutral"
    assert result["score"] == pytest.approx(0.0)


def test_score_headline_api_error():
    """A RequestException from Gemini should return neutral default without crashing."""
    import requests as req

    analyzer = _make_analyzer()
    with patch("requests.post", side_effect=req.exceptions.RequestException("timeout")):
        result = analyzer.score_headline("Some headline")
    assert result["label"] == "Neutral"
    assert result["score"] == pytest.approx(0.0)


def test_batch_skips_cached():
    """Headlines already in the cache should not trigger additional Gemini API calls."""
    analyzer = _make_analyzer()
    entry_date = date(2024, 1, 15)
    ticker = "RELIANCE.NS"
    headline = "Reliance Q3 profit rises 12%"

    # Pre-populate the cache
    analyzer._store.save_sentiment(ticker, entry_date, headline, "Positive", 1.0)

    with patch("requests.post") as mock_post:
        results = analyzer.score_batch([headline], entry_date, ticker)
        mock_post.assert_not_called()

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_batch_rate_limited():
    """score_batch must call the rate limiter between API requests."""
    analyzer = _make_analyzer()
    entry_date = date(2024, 1, 15)
    ticker = "TCS.NS"

    headlines = ["Headline A", "Headline B"]

    with patch("requests.post", return_value=_mock_gemini_response("Neutral")):
        with patch.object(analyzer, "_rate_limit", wraps=analyzer._rate_limit) as mock_rl:
            analyzer.score_batch(headlines, entry_date, ticker)
            # Should be called once per uncached headline
            assert mock_rl.call_count == len(headlines)


def test_get_daily_sentiment_mean():
    """
    Three headlines with scores 1.0, -1.0, 0.0 should give daily_sentiment == 0.0.
    """
    analyzer = _make_analyzer()
    entry_date = date(2024, 1, 15)
    ticker = "INFY.NS"

    headlines = ["Positive news", "Negative news", "Neutral news"]
    labels = ["Positive", "Negative", "Neutral"]
    responses = [_mock_gemini_response(lbl) for lbl in labels]

    with patch("requests.post", side_effect=responses):
        daily_score = analyzer.get_daily_sentiment(ticker, entry_date, headlines)

    assert abs(daily_score - 0.0) < 1e-6, f"Expected 0.0, got {daily_score}"


def test_get_daily_sentiment_empty_headlines():
    """No headlines should return 0.0 without calling the API."""
    analyzer = _make_analyzer()
    with patch("requests.post") as mock_post:
        score = analyzer.get_daily_sentiment("RELIANCE.NS", date(2024, 1, 15), [])
        mock_post.assert_not_called()
    assert score == pytest.approx(0.0)


def _mock_newsapi_response(headlines: list[str]) -> MagicMock:
    """Build a mock requests.Response for a NewsAPI articles list."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "articles": [{"title": h} for h in headlines]
    }
    return mock_resp


def test_fetch_and_score_news_no_api_key():
    """Missing NEWS_API_KEY must return an empty DataFrame without crashing."""
    import trading_bot.data.sentiment as sentiment_mod

    analyzer = _make_analyzer()
    original = sentiment_mod.NEWS_API_KEY
    try:
        sentiment_mod.NEWS_API_KEY = None  # type: ignore[assignment]
        result = analyzer.fetch_and_score_news("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))
    finally:
        sentiment_mod.NEWS_API_KEY = original

    assert result.empty
    assert "sentiment_score" in result.columns
    assert "headline_count" in result.columns


def test_fetch_and_score_news_with_headlines():
    """fetch_and_score_news must return one row per day that has headlines."""
    import trading_bot.data.sentiment as sentiment_mod

    analyzer = _make_analyzer()
    original_key = sentiment_mod.NEWS_API_KEY

    try:
        sentiment_mod.NEWS_API_KEY = "fake-key"

        newsapi_resp = _mock_newsapi_response(["Reliance Q4 profit up 10%", "Reliance wins contract"])
        gemini_resp = _mock_gemini_response("Positive")

        with patch("requests.get", return_value=newsapi_resp):
            with patch("requests.post", return_value=gemini_resp):
                result = analyzer.fetch_and_score_news(
                    "RELIANCE.NS", date(2024, 1, 2), date(2024, 1, 2)
                )
    finally:
        sentiment_mod.NEWS_API_KEY = original_key

    assert not result.empty
    assert "sentiment_score" in result.columns
    assert "headline_count" in result.columns
    assert result["headline_count"].iloc[0] == 2


def test_fetch_headlines_newsapi_error_returns_empty():
    """A RequestException from NewsAPI must return an empty list without crashing."""
    import requests as req

    import trading_bot.data.sentiment as sentiment_mod

    analyzer = _make_analyzer()
    original_key = sentiment_mod.NEWS_API_KEY
    try:
        sentiment_mod.NEWS_API_KEY = "fake-key"
        with patch("requests.get", side_effect=req.exceptions.RequestException("timeout")):
            headlines = analyzer._fetch_headlines_for_date("RELIANCE", date(2024, 1, 2))
    finally:
        sentiment_mod.NEWS_API_KEY = original_key

    assert headlines == []
