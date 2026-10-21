from typing import List, Optional

from fastapi import APIRouter, Depends

from app.api.dependencies import get_recommender
from app.schemas import RecommendationItem, SessionRequest
from app.services.recommender import RecommenderService

router = APIRouter()


@router.get("/")
def home():
    return {"message": "JOODPAKPAW Recommendation Service is Running"}


@router.post("/recommend/session", response_model=List[RecommendationItem])
def recommend_session(request: SessionRequest, service: RecommenderService = Depends(get_recommender)):
    return service.recommend_for_session(request.user_id, request.viewed_ids)


@router.get("/recommend/user/{user_id}", response_model=List[RecommendationItem])
def recommend_user(user_id: int, province: Optional[str] = None, service: RecommenderService = Depends(get_recommender)):
    return service.recommend_for_user(user_id, province=province)


@router.get("/recommend/property/{property_id}", response_model=List[RecommendationItem])
def recommend_similar_property(property_id: int, limit: int = 10, service: RecommenderService = Depends(get_recommender)):
    return service.similar_properties(property_id, limit=limit)
