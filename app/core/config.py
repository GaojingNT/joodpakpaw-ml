import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class ScoringConfig:
    # Hybrid weighting: how much cosine similarity vs quality score contribute.
    # alpha + beta should equal 1.0.
    alpha: float = 0.6  # Weight for content-based cosine similarity score
    beta: float = 0.4   # Weight for quality score (ratings + sentiment)

    # The quality multiplier range. A property with a perfect score gets 1.0x,
    # a property with the worst score gets min_quality_mult x.
    # This prevents a single bad review from completely hiding a relevant result.
    min_quality_mult: float = 0.5

    # Bayesian prior: equivalent to every property "starting" with bayes_c phantom
    # ratings at bayes_m. Prevents a single 5-star review from dominating.
    bayes_c: int = 5      # Prior count (raise this to trust actual ratings less early on)
    bayes_m: float = 3.0  # Prior mean (neutral midpoint of 1–5 scale)

    # Share of star rating vs text sentiment inside the quality score
    rating_weight: float = 0.7
    sentiment_weight: float = 0.3

    # How much one interaction counts when building a user's taste profile
    booking_weight: float = 5.0
    view_weight: float = 1.0


@dataclass(frozen=True)
class Settings:
    database_url: str
    # Seconds to reuse property features between requests (0 = rebuild every request)
    cache_ttl_seconds: int = 60
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@lru_cache
def get_settings() -> Settings:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    return Settings(
        database_url=database_url,
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "60")),
    )
