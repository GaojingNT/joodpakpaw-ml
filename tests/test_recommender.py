import pandas as pd

from app.services.feature_builder import build_property_frame, get_price_tag, get_size_tag
from tests.fakes import SAMPLE_SPEC, FakeDataSource, make_service, property_rows


def ids(items):
    return [item["property_id"] for item in items]


def test_tags_combine_facilities_price_and_sizes():
    rows = property_rows({7: (1, "x", ["Pet Suites", "Air Conditioning"], [(900, 1), (500, 3)])})

    frame = build_property_frame(rows)

    assert frame.loc[0, 'tags'] == ["Air Conditioning", "Pet Suites", "price_tier_budget", "size_small", "size_large"]


def test_price_and_size_tags():
    assert [get_price_tag(p) for p in (699, 700, 1499, 1500, float('nan'))] == [
        "price_tier_budget", "price_tier_premium", "price_tier_premium", "price_tier_luxury", "price_unknown"]
    assert [get_size_tag(s) for s in (1, 2, 3, 9)] == ["size_small", "size_medium", "size_large", "size_all"]


def test_similar_properties_ranks_same_kind_first_and_skips_itself():
    service = make_service(FakeDataSource(property_rows(SAMPLE_SPEC)))

    result = service.similar_properties(2, limit=10)

    assert 2 not in ids(result)
    assert ids(result)[:2] == [1, 3]  # cat places before dog places
    assert set(result[0]) == {"property_id", "similarity_score", "quality_score", "hybrid_score", "matched_tags"}


def test_similar_properties_respects_limit_and_unknown_id():
    service = make_service(FakeDataSource(property_rows(SAMPLE_SPEC)))

    assert len(service.similar_properties(1, limit=2)) == 2
    assert service.similar_properties(999) == []


def test_session_filters_seen_own_and_province():
    service = make_service(FakeDataSource(property_rows(SAMPLE_SPEC)))

    result = service.recommend_for_session(100, [2])
    assert 2 not in ids(result)              # already seen
    assert {1, 4}.isdisjoint(ids(result))    # owned by user 100
    assert ids(result)[0] == 3

    in_chiang_mai = service.recommend_for_session(100, [2], province="เชียงใหม่")
    assert ids(in_chiang_mai) == [3, 5]


def test_booking_weighs_more_than_view():
    service = make_service(FakeDataSource(property_rows(SAMPLE_SPEC)))

    # Viewed a cat place but booked a dog place → dog places should win
    result = service.recommend_for_session(999, [(2, 'view_property'), (4, 'create_booking')])

    assert ids(result)[0] == 5


def test_user_without_history_gets_empty_list():
    service = make_service(FakeDataSource(property_rows(SAMPLE_SPEC)))

    assert service.recommend_for_user(42) == []


def test_user_history_comes_from_data_source():
    data = FakeDataSource(property_rows(SAMPLE_SPEC), histories={42: [(5, 'create_booking')]})
    service = make_service(data)

    assert ids(service.recommend_for_user(42))[0] == 4


def test_empty_catalog_returns_nothing():
    empty_rows = pd.DataFrame(columns=property_rows(SAMPLE_SPEC).columns)
    service = make_service(FakeDataSource(empty_rows))

    assert service.similar_properties(1) == []
    assert service.recommend_for_session(1, [1]) == []


def test_ties_keep_property_order():
    # Identical properties → identical scores → order must follow property_id
    spec = {pid: (1, "x", ["Air Conditioning"], [(300, 1)]) for pid in (5, 3, 9, 1)}
    service = make_service(FakeDataSource(property_rows(spec)))

    assert ids(service.similar_properties(9)) == [1, 3, 5]


def test_features_are_cached_within_ttl():
    data = FakeDataSource(property_rows(SAMPLE_SPEC))
    service = make_service(data, ttl_seconds=60)

    service.similar_properties(1)
    service.similar_properties(2)

    assert data.row_calls == 1


def test_features_rebuild_when_ttl_is_zero():
    data = FakeDataSource(property_rows(SAMPLE_SPEC))
    service = make_service(data, ttl_seconds=0)

    service.similar_properties(1)
    service.similar_properties(2)

    assert data.row_calls == 2
