from functools import lru_cache

from sqlalchemy import create_engine

from app.core.config import get_settings
from app.repositories.property_repository import PropertyRepository
from app.services.feature_builder import FeatureStore
from app.services.quality_scorer import QualityScorer
from app.services.recommender import RecommenderService
from app.services.sentiment import VaderSentimentAnalyzer


@lru_cache
def get_recommender() -> RecommenderService:
    """Wires the production object graph once per process."""
    settings = get_settings()

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    repository = PropertyRepository(engine)
    quality_scorer = QualityScorer(settings.scoring, VaderSentimentAnalyzer())
    feature_store = FeatureStore(repository, quality_scorer, ttl_seconds=settings.cache_ttl_seconds)

    return RecommenderService(repository, feature_store, settings.scoring)
