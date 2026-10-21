from typing import List, Protocol, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# (property_id, action name) — newest first
UserHistory = List[Tuple[int, str]]


class PropertyDataSource(Protocol):
    """Everything the recommender needs from storage. Tests can pass a fake instead of the DB."""

    def get_property_rows(self) -> pd.DataFrame: ...

    def get_property_ratings(self) -> pd.DataFrame: ...

    def get_user_history(self, user_id: int) -> UserHistory: ...


class PropertyRepository:
    """Reads property, rating and behaviour data from the JOODPAKPAW database."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def get_property_rows(self) -> pd.DataFrame:
        """
        One row per (active property, facility, room category).
        Columns: property_id, title, province, district, host_id, facility_name,
        base_price, pet_size_category_id.
        """
        query = """
        select  p.id as property_id,
            p.title,
            p.province,
            p.district,
            p.host_id,
            f.name as facility_name,
            rc.base_price,
            rc.pet_size_category_id
        from tbl_properties p
        left join  tbl_properties_has_facilities phf on p.id = phf.property_id
        left join tbl_facilities f on phf.facility_id = f.id
        left join tbl_room_categories rc on p.id = rc.property_id
        where p.is_active = true;
        """
        return pd.read_sql(query, self._engine)

    def get_property_ratings(self) -> pd.DataFrame:
        """
        Ratings and comments from bookings, resolved to property_id.
        Columns: property_id, rating, comment.
        """
        query = text("""
            SELECT
                bo.property_id,
                b.rating,
                b.comment
            FROM tbl_bookings b
            JOIN tbl_booking_orders bo ON b.id = bo.booking_id
            WHERE b.rating IS NOT NULL
        """)

        with self._engine.connect() as conn:
            rows = conn.execute(query).fetchall()

        if not rows:
            return pd.DataFrame(columns=['property_id', 'rating', 'comment'])

        return pd.DataFrame(rows, columns=['property_id', 'rating', 'comment'])

    def get_user_history(self, user_id: int) -> UserHistory:
        """Latest 50 property views/bookings of a user on active properties."""
        query = text("""
            SELECT b.reference_record, c.name as action
            FROM tbl_user_behaviors b
            JOIN tbl_user_behavior_categories c ON b.behavior_category_id = c.id
            JOIN tbl_properties p ON CAST(b.reference_record AS INTEGER) = p.id
            WHERE b.user_id = :user_id
              AND b.reference_table = 'tbl_properties'
              AND c.name IN ('view_property', 'create_booking')
              AND p.is_active = true
            ORDER BY b.created_at DESC
            LIMIT 50;
        """)

        with self._engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id})
            return [(int(row[0]), row[1]) for row in result]
