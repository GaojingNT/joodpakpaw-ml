import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class SentimentAnalyzer(Protocol):
    """Scores one review comment in [-1.0, +1.0]."""

    def score(self, comment: str) -> float: ...


class NeutralSentimentAnalyzer:
    """Adds no signal. Used in tests or when no analyzer is available."""

    def score(self, comment: str) -> float:
        return 0.0


class VaderSentimentAnalyzer:
    """
    VADER (rule-based, no training needed). Loaded once and reused for every comment.

    IMPORTANT: VADER works best on English text. Thai reviews get near-zero scores
    (neutral), which is safe — it just won't add signal. To support Thai later,
    add another SentimentAnalyzer (e.g. pythainlp-based) and inject it instead.
    """

    def __init__(self):
        self._analyzer = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer

                try:
                    self._analyzer = SentimentIntensityAnalyzer()
                except LookupError:
                    # Lexicon not baked into the image — fetch it once
                    nltk.download('vader_lexicon', quiet=True)
                    self._analyzer = SentimentIntensityAnalyzer()
            except Exception:
                logger.warning("VADER is unavailable; sentiment scores fall back to neutral", exc_info=True)
                self._analyzer = None

    def score(self, comment: str) -> float:
        if not self._loaded:
            self._load()
        if self._analyzer is None:
            return 0.0
        try:
            return self._analyzer.polarity_scores(str(comment))['compound']
        except Exception:
            return 0.0
