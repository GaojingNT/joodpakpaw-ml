import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from app.repositories.property_repository import PropertyDataSource
from app.services.quality_scorer import QualityScorer


# --- Tag translators: turn raw columns into behavioural tags for the ML ---

def get_price_tag(price) -> str:
    """Turns a raw THB price into a behavioural tag."""
    if pd.isna(price):
        return "price_unknown"
    if price <= 699:
        return "price_tier_budget"
    elif price <= 1499:
        return "price_tier_premium"
    else:
        return "price_tier_luxury"


def get_size_tag(size_id) -> str:
    """Turns pet_size_category_id into a behavioural tag (ids follow pet_size_category_seeder)."""
    if size_id == 1:
        return "size_small"
    elif size_id == 2:
        return "size_medium"
    elif size_id == 3:
        return "size_large"
    else:
        return "size_all"


def build_property_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Squashes one-row-per-facility/room data into one row per property, sorted by
    property_id, with a combined 'tags' list (facilities + price tag + size tags).
    """
    grouped = rows.groupby('property_id').agg({
        'title': 'first',
        'province': 'first',
        'host_id': 'first',
        'base_price': 'min',
        # sorted() keeps tag order stable between restarts
        'pet_size_category_id': lambda x: sorted(set(x.dropna())),
        'facility_name': lambda x: sorted(set(x.dropna())),
    }).reset_index()

    grouped['price_tag'] = grouped['base_price'].apply(get_price_tag)
    grouped['size_tags'] = grouped['pet_size_category_id'].apply(lambda sizes: [get_size_tag(s) for s in sizes])
    grouped['tags'] = grouped.apply(
        lambda row: row['facility_name'] + [row['price_tag']] + row['size_tags'],
        axis=1,
    )
    return grouped


@dataclass(frozen=True)
class PropertyFeatures:
    frame: pd.DataFrame          # one row per active property
    tfidf_matrix: csr_matrix     # row i ↔ frame.iloc[i]
    quality: np.ndarray          # quality multiplier per row
    row_by_property_id: Dict[int, int]

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


class FeatureStore:
    """Builds PropertyFeatures from the data source and reuses them for ttl_seconds."""

    def __init__(self, data_source: PropertyDataSource, quality_scorer: QualityScorer, ttl_seconds: int = 60):
        self._data_source = data_source
        self._quality_scorer = quality_scorer
        self._ttl_seconds = ttl_seconds
        self._cached: Optional[PropertyFeatures] = None
        self._built_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> PropertyFeatures:
        with self._lock:
            is_fresh = self._cached is not None and (time.monotonic() - self._built_at) < self._ttl_seconds
            if not is_fresh:
                self._cached = self._build()
                self._built_at = time.monotonic()
            return self._cached

    def _build(self) -> PropertyFeatures:
        rows = self._data_source.get_property_rows()
        if rows.empty:
            return PropertyFeatures(rows, csr_matrix((0, 0)), np.array([]), {})

        frame = build_property_frame(rows)
        tags_strings = frame['tags'].apply(lambda tags: " ".join(tags) if isinstance(tags, list) else "")
        tfidf_matrix = TfidfVectorizer().fit_transform(tags_strings)

        ratings = self._data_source.get_property_ratings()
        quality = self._quality_scorer.multipliers(frame['property_id'], ratings)

        row_by_property_id = {int(pid): i for i, pid in enumerate(frame['property_id'])}
        return PropertyFeatures(frame, tfidf_matrix, quality, row_by_property_id)
