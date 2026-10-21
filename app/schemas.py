from typing import List

from pydantic import BaseModel


class SessionRequest(BaseModel):
    # user_id lets us filter out the user's own properties
    user_id: int
    viewed_ids: List[int]


class RecommendationItem(BaseModel):
    property_id: int
    similarity_score: float
    quality_score: float
    hybrid_score: float
    matched_tags: List[str]
