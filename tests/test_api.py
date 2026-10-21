import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_recommender
from app.main import create_app
from tests.fakes import SAMPLE_SPEC, FakeDataSource, make_service, property_rows


@pytest.fixture
def client():
    app = create_app()
    data = FakeDataSource(property_rows(SAMPLE_SPEC), histories={42: [(2, 'view_property')]})
    app.dependency_overrides[get_recommender] = lambda: make_service(data)
    return TestClient(app)


def test_home(client):
    assert client.get("/").json() == {"message": "JOODPAKPAW Recommendation Service is Running"}


def test_property_endpoint(client):
    response = client.get("/recommend/property/2?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert [item["property_id"] for item in body] == [1, 3]
    assert set(body[0]) == {"property_id", "similarity_score", "quality_score", "hybrid_score", "matched_tags"}


def test_user_endpoint_returns_list_even_without_history(client):
    assert client.get("/recommend/user/7").json() == []
    assert len(client.get("/recommend/user/42").json()) > 0


def test_user_endpoint_province_filter(client):
    body = client.get("/recommend/user/42", params={"province": "เชียงใหม่"}).json()

    assert [item["property_id"] for item in body] == [3, 5]


def test_session_endpoint(client):
    response = client.post("/recommend/session", json={"user_id": 100, "viewed_ids": [2]})

    assert response.status_code == 200
    assert [item["property_id"] for item in response.json()][0] == 3


def test_session_endpoint_validates_body(client):
    assert client.post("/recommend/session", json={"user_id": "x"}).status_code == 422
