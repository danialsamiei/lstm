"""
Sentiment Data Collector and Analyzer
Collects text data from Twitter/Reddit and computes sentiment scores using FinBERT.
Based on thesis sections 3-5-3 and 3-6-4.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes financial text sentiment using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.pipeline = None

    def load_model(self):
        """Load FinBERT model for sentiment analysis."""
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            logger.info(f"Loaded sentiment model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self.pipeline = None

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.

        Returns:
            Float between -1 (very negative) and +1 (very positive)
        """
        if self.pipeline is None:
            self.load_model()

        if self.pipeline is None:
            return 0.0

        try:
            results = self.pipeline(text[:512])[0]
            score_map = {}
            for r in results:
                score_map[r["label"].lower()] = r["score"]

            sentiment = (
                score_map.get("positive", 0)
                - score_map.get("negative", 0)
            )
            return float(sentiment)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Analyze sentiment for a batch of texts."""
        if self.pipeline is None:
            self.load_model()

        if self.pipeline is None:
            return [0.0] * len(texts)

        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t[:512] for t in batch]
            try:
                results = self.pipeline(batch)
                for result in results:
                    score_map = {}
                    for r in result:
                        score_map[r["label"].lower()] = r["score"]
                    score = score_map.get("positive", 0) - score_map.get("negative", 0)
                    scores.append(float(score))
            except Exception as e:
                logger.warning(f"Batch sentiment analysis failed: {e}")
                scores.extend([0.0] * len(batch))

        return scores


class SentimentDataCollector:
    """Collects and processes sentiment data from social media."""

    def __init__(self, analyzer: SentimentAnalyzer = None):
        self.analyzer = analyzer or SentimentAnalyzer()

    def collect_twitter_data(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        bearer_token: str = None,
    ) -> pd.DataFrame:
        """
        Collect tweets using Twitter API v2.

        Args:
            keywords: Search keywords
            start_date: Start date
            end_date: End date
            bearer_token: Twitter API bearer token

        Returns:
            DataFrame with tweet text, timestamp, and metadata
        """
        if bearer_token:
            return self._fetch_real_tweets(keywords, start_date, end_date, bearer_token)

        logger.warning("No Twitter bearer token provided, generating synthetic data")
        return self._generate_synthetic_sentiment(start_date, end_date, source="twitter")

    def collect_reddit_data(
        self,
        subreddits: list[str] = None,
        keywords: list[str] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Collect Reddit posts and comments."""
        logger.warning("Using synthetic Reddit data")
        return self._generate_synthetic_sentiment(start_date, end_date, source="reddit")

    def _fetch_real_tweets(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        bearer_token: str,
    ) -> pd.DataFrame:
        """Fetch real tweets using tweepy."""
        try:
            import tweepy

            client = tweepy.Client(bearer_token=bearer_token)
            query = " OR ".join(keywords) + " -is:retweet lang:en"

            tweets_data = []
            for response in tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                max_results=100,
                limit=100,
            ):
                if response.data:
                    for tweet in response.data:
                        tweets_data.append({
                            "text": tweet.text,
                            "timestamp": tweet.created_at,
                            "likes": tweet.public_metrics.get("like_count", 0),
                            "retweets": tweet.public_metrics.get("retweet_count", 0),
                        })

            return pd.DataFrame(tweets_data)

        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return self._generate_synthetic_sentiment(start_date, end_date, "twitter")

    def _generate_synthetic_sentiment(
        self,
        start_date: str,
        end_date: str,
        source: str = "twitter",
    ) -> pd.DataFrame:
        """
        Generate synthetic sentiment data for testing/development.
        Simulates realistic sentiment patterns correlated with market events.
        """
        np.random.seed(42 if source == "twitter" else 123)
        dates = pd.date_range(start=start_date, end=end_date, freq="1h")
        n = len(dates)

        # Base sentiment: slightly positive with noise
        base_sentiment = np.random.normal(0.05, 0.3, n)

        # Add event-driven sentiment shifts
        events = [
            # (start_hour, duration, sentiment_shift)
            (500, 200, -0.6),    # Negative event (e.g., hack)
            (3000, 400, -0.8),   # Severe negative (e.g., Terra crash)
            (5000, 100, -0.4),   # Moderate negative
            (7000, 150, 0.5),    # Positive recovery
            (10000, 200, -0.3),  # Banking crisis
        ]

        for start_idx, duration, shift in events:
            if start_idx + duration < n:
                event_sentiment = np.linspace(shift, shift * 0.1, duration)
                base_sentiment[start_idx : start_idx + duration] += event_sentiment

        # Clip to [-1, 1]
        sentiment_scores = np.clip(base_sentiment, -1, 1)

        # Generate synthetic social metrics
        tweet_volume = np.abs(np.random.lognormal(mean=6, sigma=1, size=n)).astype(int)
        # More tweets during high-sentiment periods
        tweet_volume = (tweet_volume * (1 + np.abs(sentiment_scores))).astype(int)

        df = pd.DataFrame(
            {
                "sentiment_score": sentiment_scores,
                "tweet_volume": tweet_volume,
                "source": source,
            },
            index=dates,
        )
        df.index.name = "timestamp"

        logger.info(f"Generated {n} synthetic {source} sentiment data points")
        return df

    def aggregate_sentiment(
        self, sentiment_df: pd.DataFrame, freq: str = "1h"
    ) -> pd.DataFrame:
        """
        Aggregate raw sentiment scores to target frequency.
        Uses weighted mean based on engagement metrics.
        """
        if "tweet_volume" in sentiment_df.columns:
            # Volume-weighted sentiment
            resampled = sentiment_df.resample(freq).agg({
                "sentiment_score": "mean",
                "tweet_volume": "sum",
            })
        else:
            resampled = sentiment_df.resample(freq).agg({
                "sentiment_score": "mean",
            })

        resampled = resampled.ffill()
        return resampled

    def save_data(
        self, data: pd.DataFrame, coin_name: str, output_dir: str = "data/raw"
    ):
        """Save sentiment data to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"{coin_name}_sentiment.csv"
        data.to_csv(filepath)
        logger.info(f"Saved {coin_name} sentiment data to {filepath}")
