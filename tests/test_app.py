"""
End-to-end tests for the FastAPI app, using the already-committed model
and data artifacts (app/model/*.joblib, data/*.csv) - no dependency on the
raw Transfermarkt dataset.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_predict_payload():
    return {
        "age_at_transfer": 24.0,
        "height_in_cm": 182.0,
        "position": "Attack",
        "foot": "right",
        "pre_apps": 30.0,
        "pre_minutes": 2500.0,
        "pre_goals_p90": 0.5,
        "pre_ga_p90": 0.7,
        "pre_mins_per_app": 83.0,
        "transfer_fee": 50_000_000.0,
        "value_before": 60_000_000.0,
        "from_domestic_competition_id": "L1",
        "to_domestic_competition_id": "GB1",
        "from_total_market_value": 400_000_000.0,
        "to_total_market_value": 900_000_000.0,
    }


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_examples_returns_known_transfers_with_breakdown():
    res = client.get("/api/examples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    for ex in data:
        assert 0 <= ex["success_score"] <= 100
        assert len(ex["breakdown"]) == 5
        for component in ex["breakdown"]:
            assert "description" in component and component["description"]


def test_predict_returns_score_range_and_explanation(sample_predict_payload):
    res = client.post("/api/predict", json=sample_predict_payload)
    assert res.status_code == 200
    data = res.json()
    assert 0 <= data["success_score"] <= 100
    lo, hi = data["score_range"]
    assert lo <= hi
    assert len(data["explanation"]) == 5
    for e in data["explanation"]:
        assert "detail" in e and e["detail"]
        assert isinstance(e["contribution"], (int, float))


def test_predict_rejects_invalid_payload():
    res = client.post("/api/predict", json={"age_at_transfer": 5})
    assert res.status_code == 422


def test_compare_returns_both_results_and_delta(sample_predict_payload):
    other = dict(sample_predict_payload, transfer_fee=10_000_000.0)
    res = client.post("/api/compare", json={
        "a": sample_predict_payload, "b": other,
        "label_a": "Scenario A", "label_b": "Scenario B",
    })
    assert res.status_code == 200
    data = res.json()
    assert "success_score" in data["a"]
    assert "success_score" in data["b"]
    assert data["delta"] == pytest.approx(data["a"]["success_score"] - data["b"]["success_score"], abs=0.05)


def test_players_search_is_accent_insensitive():
    res = client.get("/api/players/search", params={"q": "Dembele"})
    assert res.status_code == 200
    names = [p["name"] for p in res.json()]
    assert any("Dembélé" in n for n in names)


def test_players_search_response_has_no_internal_fields():
    res = client.get("/api/players/search", params={"q": "Haaland"})
    for p in res.json():
        assert "_name_fold" not in p


def test_clubs_search_is_accent_insensitive():
    res = client.get("/api/clubs/search", params={"q": "atletico"})
    assert res.status_code == 200
    names = [c["name"] for c in res.json()]
    assert any("Atlético" in n for n in names)


def test_clubs_search_response_has_no_internal_fields():
    res = client.get("/api/clubs/search", params={"q": "Real"})
    for c in res.json():
        assert "_name_fold" not in c


def test_filters_endpoint():
    res = client.get("/api/filters")
    assert res.status_code == 200
    data = res.json()
    assert "Attack" in data["positions"]
    assert len(data["leagues"]) > 0
    assert all("id" in l and "name" in l for l in data["leagues"])


def test_league_names_disambiguate_duplicates():
    res = client.get("/api/filters")
    leagues = {l["id"]: l["name"] for l in res.json()["leagues"]}
    assert leagues["A1"] != leagues["L1"]
    assert "Bundesliga" in leagues["A1"] and "Bundesliga" in leagues["L1"]


def test_transfers_list_pagination():
    res = client.get("/api/transfers", params={"limit": 10, "offset": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 1000
    assert len(data["results"]) == 10


def test_transfers_list_position_filter():
    res = client.get("/api/transfers", params={"position": "Goalkeeper", "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(r["position"] == "Goalkeeper" for r in data["results"])


def test_transfers_list_search_is_accent_insensitive():
    res = client.get("/api/transfers", params={"q": "Dembele"})
    data = res.json()
    assert data["total"] > 0
    assert any("Dembélé" in r["name"] for r in data["results"])


def test_transfers_list_sorted_descending_by_default():
    res = client.get("/api/transfers", params={"limit": 20})
    scores = [r["success_score"] for r in res.json()["results"]]
    assert scores == sorted(scores, reverse=True)
