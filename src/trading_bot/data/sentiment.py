"""
Gemini-powered sentiment analyser for financial news headlines.

No Streamlit dependencies — pure Python with SQLite caching.
Rate limited to 1 request/second to respect the Gemini free tier.
"""

import hashlib
import json
import logging
import time
from datetime import date

import pandas as pd
import requests

from trading_bot.config import GEMINI_API_KEY, NEWS_API_KEY
from trading_bot.data.store import DataStore

logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={api_key}"
)
_LABEL_TO_SCORE: dict[str, float] = {
    "Positive": 1.0,
    "Neutral": 0.0,
    "Negative": -1.0,
}
_REQUEST_INTERVAL = 1.0  # seconds between Gemini calls (free tier)


class SentimentAnalyzer:
    """Scores financial headlines using the Gemini API with SQLite caching."""

    def __init__(self, store: DataStore | None = None) -> None:
        """
        Args:
            store: DataStore instance. Creates a default one if not provided.
        """
        self._store = store or DataStore()
        self._last_request_time: float = 0.0

    def score_headline(self, headline: str) -> dict:
        """
        Send a single headline to Gemini and return a sentiment dict.

        Rate-limited to 1 request/second.

        Args:
            headline: Raw news headline text.

        Returns:
            Dict with keys: label (str), score (float), raw (str).
        """
        self._rate_limit()

        prompt = (
            "Analyze the sentiment of the following financial news headline. "
            "Respond with a single word: 'Positive', 'Negative', or 'Neutral'. "
            "Do not add any other text.\n\n"
            f"Headline: '{headline}'"
        )
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        url = _GEMINI_URL.format(api_key=GEMINI_API_KEY)

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            raw_text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "Neutral")
                .strip()
            )
        except requests.RequestException as exc:
            logger.warning("Gemini API error: %s. Defaulting to Neutral.", exc)
            raw_text = "Neutral"

        label = raw_text if raw_text in _LABEL_TO_SCORE else "Neutral"
        score = _LABEL_TO_SCORE[label]
        logger.debug("Scored headline → %s (%.1f)", label, score)
        return {"label": label, "score": score, "raw": raw_text}

    def score_batch(
        self,
        headlines: list[str],
        entry_date: date,
        ticker: str,
    ) -> list[dict]:
        """
        Score a list of headlines, skipping any already in the cache.

        Args:
            headlines: List of raw headline strings.
            entry_date: Publication date (used for cache keying).
            ticker: NSE ticker symbol (used for cache keying).

        Returns:
            List of dicts with keys: label, score, raw.
        """
        cached_entries = self._store.load_sentiment(ticker, entry_date)
        cached_hashes = {e["headline_hash"] for e in cached_entries}

        results: list[dict] = []
        for headline in headlines:
            h = hashlib.sha256(headline.encode()).hexdigest()
            if h in cached_hashes:
                # Retrieve from cache
                entry = next(e for e in cached_entries if e["headline_hash"] == h)
                results.append(
                    {
                        "label": entry["sentiment_label"],
                        "score": entry["sentiment_score"],
                        "raw": "",
                    }
                )
                logger.debug("Cache hit for headline hash %s", h[:8])
                continue

            scored = self.score_headline(headline)
            self._store.save_sentiment(
                ticker, entry_date, headline, scored["label"], scored["score"], scored["raw"]
            )
            results.append(scored)

        return results

    def get_daily_sentiment(
        self,
        ticker: str,
        entry_date: date,
        headlines: list[str],
    ) -> float:
        """
        Return the mean sentiment score for *ticker* on *entry_date*.

        Scores are fetched or retrieved from cache.

        Returns:
            Mean sentiment score in [-1.0, 1.0]. Returns 0.0 if no headlines.
        """
        if not headlines:
            return 0.0
        scored = self.score_batch(headlines, entry_date, ticker)
        scores = [s["score"] for s in scored]
        return float(sum(scores) / len(scores))

    def fetch_and_score_news(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch headlines from NewsAPI for a date range and score them with Gemini.

        Only fetches for the most recent date (end) to avoid API rate limits.
        Historical dates are filled with 0.0.

        Args:
            ticker: NSE ticker symbol (used as search query keyword).
            start: First date.
            end: Last date.

        Returns:
            DataFrame with columns [date, sentiment_score, headline_count].
            One row per day that has at least one headline.
        """
        if not NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set — returning empty sentiment DataFrame.")
            return pd.DataFrame(columns=["date", "sentiment_score", "headline_count"])

        # Strip NSE suffixes for a cleaner search query
        query = ticker.replace(".NS", "").replace("^", "")
        rows: list[dict] = []

        # Only fetch news for the most recent date (live inference)
        # Historical dates get 0.0 to avoid API rate limits
        try:
            headlines = self._fetch_headlines_for_date(query, end)
            if headlines:
                score = self.get_daily_sentiment(ticker, end, headlines)
                rows.append(
                    {
                        "date": end,
                        "sentiment_score": score,
                        "headline_count": len(headlines),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI error for %s on %s: %s — defaulting to 0.0", query, end, exc)

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "sentiment_score", "headline_count"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_headlines_for_date(self, query: str, for_date: date) -> list[str]:
        """Fetch up to 5 headlines from NewsAPI for *query* on *for_date*."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": str(for_date),
            "to": str(for_date),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a.get("title")]
        except requests.RequestException as exc:
            logger.warning("NewsAPI error for %s on %s: %s", query, for_date, exc)
            return []

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain at most 1 request/second."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()
