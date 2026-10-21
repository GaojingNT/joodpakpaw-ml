import numpy as np
import pandas as pd

from app.core.config import ScoringConfig
from app.services.sentiment import SentimentAnalyzer


class QualityScorer:
    """
    Turns ratings and review comments into a quality multiplier in
    [min_quality_mult, 1.0] for every property.

    Formula per property:
        bayesian_avg   = (bayes_c * bayes_m + sum_of_ratings) / (bayes_c + count)
        norm_rating    = (bayesian_avg - 1.0) / 4.0          # maps [1,5] → [0,1]
        norm_sentiment = (mean_compound_score + 1.0) / 2.0   # maps [-1,1] → [0,1]
        quality_score  = rating_weight * norm_rating + sentiment_weight * norm_sentiment
        multiplier     = min_quality_mult + (1 - min_quality_mult) * quality_score
    """

    def __init__(self, config: ScoringConfig, sentiment: SentimentAnalyzer):
        self._config = config
        self._sentiment = sentiment

    def multipliers(self, property_ids: pd.Series, ratings_df: pd.DataFrame) -> np.ndarray:
        """Returns one multiplier per entry of property_ids, in the same order."""
        c = self._config

        # --- Step 1: Bayesian Average Rating ---
        if ratings_df.empty:
            # No reviews yet — neutral multiplier for every property
            neutral = c.min_quality_mult + (1.0 - c.min_quality_mult) * ((c.bayes_m - 1.0) / 4.0)
            return np.full(len(property_ids), neutral)

        stats = ratings_df.groupby('property_id')['rating'].agg(n='count', sum_rating='sum')
        stats['bayesian_avg'] = (c.bayes_c * c.bayes_m + stats['sum_rating']) / (c.bayes_c + stats['n'])

        # --- Step 2: Sentiment Score (averaged per property) ---
        sentiment_by_prop = (
            ratings_df[ratings_df['comment'].notna()]
            .groupby('property_id')['comment']
            .apply(lambda comments: np.mean([self._sentiment.score(x) for x in comments]))
        )

        # --- Step 3: Combine into a single quality_score in [0, 1] ---
        quality_df = pd.DataFrame({'property_id': property_ids}).set_index('property_id')
        quality_df['bayesian_avg'] = stats['bayesian_avg']
        quality_df['sentiment'] = sentiment_by_prop

        # Unrated properties fall back to the neutral prior
        quality_df['bayesian_avg'] = quality_df['bayesian_avg'].fillna(c.bayes_m)
        quality_df['sentiment'] = quality_df['sentiment'].fillna(0.0)

        quality_df['norm_rating'] = (quality_df['bayesian_avg'] - 1.0) / 4.0
        quality_df['norm_sentiment'] = (quality_df['sentiment'] + 1.0) / 2.0

        quality_df['quality_score'] = (
            c.rating_weight * quality_df['norm_rating'] +
            c.sentiment_weight * quality_df['norm_sentiment']
        )
        quality_df['multiplier'] = c.min_quality_mult + (1.0 - c.min_quality_mult) * quality_df['quality_score']

        return property_ids.map(quality_df['multiplier']).values
