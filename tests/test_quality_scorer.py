import numpy as np
import pandas as pd
import pytest

from app.core.config import ScoringConfig
from app.services.quality_scorer import QualityScorer
from app.services.sentiment import NeutralSentimentAnalyzer


class FixedSentiment:
    def __init__(self, scores):
        self.scores = scores

    def score(self, comment):
        return self.scores[comment]


def test_no_ratings_gives_neutral_multiplier_for_everyone():
    scorer = QualityScorer(ScoringConfig(), NeutralSentimentAnalyzer())
    ids = pd.Series([1, 2, 3])

    result = scorer.multipliers(ids, pd.DataFrame(columns=['property_id', 'rating', 'comment']))

    # bayes_m = 3 → norm_rating 0.5 → 0.5 + 0.5 * 0.5
    assert result == pytest.approx([0.75, 0.75, 0.75])


def test_bayesian_rating_and_sentiment_are_combined():
    scorer = QualityScorer(ScoringConfig(), FixedSentiment({"great": 1.0, "bad": -0.5}))
    ids = pd.Series([10, 20, 30])
    ratings = pd.DataFrame(
        [[10, 5, "great"], [10, 5, None], [20, 1, "bad"]],
        columns=['property_id', 'rating', 'comment'],
    )

    result = scorer.multipliers(ids, ratings)

    # Property 10: bayes (5*3 + 10) / (5 + 2) = 25/7, sentiment 1.0
    q10 = 0.7 * ((25 / 7 - 1) / 4) + 0.3 * ((1.0 + 1) / 2)
    # Property 20: bayes (15 + 1) / 6 = 16/6, sentiment -0.5
    q20 = 0.7 * ((16 / 6 - 1) / 4) + 0.3 * ((-0.5 + 1) / 2)
    # Property 30: unrated → prior 3.0, sentiment 0
    q30 = 0.7 * 0.5 + 0.3 * 0.5
    expected = [0.5 + 0.5 * q for q in (q10, q20, q30)]

    assert result == pytest.approx(expected)
    assert isinstance(result, np.ndarray)
