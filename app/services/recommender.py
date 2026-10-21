from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import ScoringConfig
from app.repositories.property_repository import PropertyDataSource
from app.services.feature_builder import FeatureStore, PropertyFeatures

# A history entry is either a (property_id, action) pair or a bare property_id (= a view)
HistoryItem = Union[int, Tuple[int, str], List]


class RecommenderService:
    """
    Hybrid recommender: content similarity (TF-IDF of property tags) blended with a
    quality score (Bayesian rating + review sentiment).
    """

    def __init__(self, data_source: PropertyDataSource, feature_store: FeatureStore, scoring: ScoringConfig):
        self._data_source = data_source
        self._features = feature_store
        self._scoring = scoring

    # --- Public use cases ---

    def recommend_for_user(self, user_id: int, province: Optional[str] = None) -> List[dict]:
        """Recommendations from the user's stored views/bookings."""
        history = self._data_source.get_user_history(user_id)
        if not history:
            return []
        return self.recommend_for_session(user_id, history, province=province)

    def recommend_for_session(self, user_id: int, history: Iterable[HistoryItem], province: Optional[str] = None) -> List[dict]:
        """
        Builds a taste profile from the given history (bookings weigh more than views),
        then ranks every other property. Skips properties already seen, the user's own
        properties, and — if given — properties outside the province.
        """
        features = self._features.get()
        if features.is_empty:
            return []

        weighted_vectors = []
        seen_ids = []
        for item in history:
            if isinstance(item, (tuple, list)):
                prop_id, action = item[0], item[1]
            else:
                prop_id, action = item, 'view_property'
            seen_ids.append(prop_id)

            row = features.row_by_property_id.get(prop_id)
            if row is None:
                continue
            weight = self._scoring.booking_weight if action == 'create_booking' else self._scoring.view_weight
            weighted_vectors.append(features.tfidf_matrix[row].toarray()[0] * weight)

        if not weighted_vectors:
            return []

        user_profile_vector = np.mean(weighted_vectors, axis=0).reshape(1, -1)
        cosine_scores = cosine_similarity(user_profile_vector, features.tfidf_matrix)[0]

        recommendations = []
        for i, hybrid in self._rank(cosine_scores, features):
            row = features.frame.iloc[i]
            rec_id = int(row['property_id'])

            # --- Hard filters ---
            if rec_id in seen_ids:
                continue
            if pd.notna(row['host_id']) and int(row['host_id']) == user_id:
                continue
            if province and row['province'] != province:
                continue

            recommendations.append(self._to_item(features, i, cosine_scores[i], hybrid))

        return recommendations

    def similar_properties(self, property_id: int, limit: int = 10) -> List[dict]:
        """Item-to-item: properties most similar to one property, ranked by hybrid score."""
        features = self._features.get()
        row = features.row_by_property_id.get(property_id)
        if row is None:
            return []  # Property not found or inactive

        target_vector = features.tfidf_matrix[row].toarray()
        cosine_scores = cosine_similarity(target_vector, features.tfidf_matrix)[0]

        recommendations = []
        for i, hybrid in self._rank(cosine_scores, features):
            rec_id = int(features.frame.iloc[i]['property_id'])
            if rec_id == property_id:
                continue  # Don't recommend the same property

            recommendations.append(self._to_item(features, i, cosine_scores[i], hybrid))
            if len(recommendations) >= limit:
                break

        return recommendations

    # --- Shared scoring ---

    def _rank(self, cosine_scores: np.ndarray, features: PropertyFeatures) -> Iterator[Tuple[int, float]]:
        """Yields (row index, hybrid score), best first. Ties keep row order (sorted is stable)."""
        # Normalize cosine scores to [0, 1] so they are on the same scale as quality
        max_cos = cosine_scores.max()
        norm_cosine = cosine_scores / max_cos if max_cos > 0 else cosine_scores

        hybrid_scores = (self._scoring.alpha * norm_cosine) + (self._scoring.beta * features.quality)
        yield from sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _to_item(features: PropertyFeatures, i: int, cosine: float, hybrid: float) -> dict:
        row = features.frame.iloc[i]
        return {
            "property_id": int(row['property_id']),
            "similarity_score": round(float(cosine), 3),  # Raw cosine for transparency
            "quality_score": round(float(features.quality[i]), 3),
            "hybrid_score": round(float(hybrid), 3),
            "matched_tags": row['tags'],
        }
