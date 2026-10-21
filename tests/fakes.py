import pandas as pd

from app.core.config import ScoringConfig
from app.services.feature_builder import FeatureStore
from app.services.quality_scorer import QualityScorer
from app.services.recommender import RecommenderService
from app.services.sentiment import NeutralSentimentAnalyzer

ROW_COLUMNS = ['property_id', 'title', 'province', 'district', 'host_id', 'facility_name', 'base_price', 'pet_size_category_id']


def property_rows(spec):
    """
    spec: {property_id: (host_id, province, [facilities], [(base_price, size_id), ...])}
    Returns rows shaped like the SQL join (one row per facility x room category).
    """
    rows = []
    for pid, (host_id, province, facilities, rooms) in spec.items():
        for facility in facilities:
            for price, size in rooms:
                rows.append([pid, f"Property {pid}", province, "district", host_id, facility, price, size])
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


class FakeDataSource:
    def __init__(self, rows, ratings=None, histories=None):
        self.rows = rows
        self.ratings = ratings if ratings is not None else pd.DataFrame(columns=['property_id', 'rating', 'comment'])
        self.histories = histories or {}
        self.row_calls = 0

    def get_property_rows(self):
        self.row_calls += 1
        return self.rows

    def get_property_ratings(self):
        return self.ratings

    def get_user_history(self, user_id):
        return self.histories.get(user_id, [])


def make_service(data_source, ttl_seconds=0, scoring=None, sentiment=None):
    scoring = scoring or ScoringConfig()
    scorer = QualityScorer(scoring, sentiment or NeutralSentimentAnalyzer())
    store = FeatureStore(data_source, scorer, ttl_seconds=ttl_seconds)
    return RecommenderService(data_source, store, scoring)


# Cat places (1, 2, 3) vs dog places (4, 5); host 100 owns 1 and 4
SAMPLE_SPEC = {
    1: (100, "กรุงเทพมหานคร", ["Cat Garden", "Air Conditioning"], [(300, 1)]),
    2: (200, "กรุงเทพมหานคร", ["Cat Garden", "Air Conditioning", "Pet Suites"], [(320, 1)]),
    3: (200, "เชียงใหม่", ["Cat Garden", "Pet Suites"], [(900, 2)]),
    4: (100, "กรุงเทพมหานคร", ["Dog Park", "Pet Swimming Pool"], [(1600, 3)]),
    5: (300, "เชียงใหม่", ["Dog Park", "Walking Trail"], [(1700, 3)]),
}
