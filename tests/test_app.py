"""
End-to-end tests for the FastAPI app, using the already-committed model
and data artifacts (app/model/*.joblib, data/*.csv) - no dependency on the
raw Transfermarkt dataset.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import (
    app, build_feature_row, explain_prediction, pipeline, predict_marginalized_recent_performance, PredictRequest,
)

client = TestClient(app)


def explain_all(payload_dict):
    """
    Call explain_prediction() directly (bypassing the HTTP layer's default
    top_k=5) so tests can check a feature's contribution/detail even when
    it isn't one of the top 5 shown in the UI - e.g. transfer fee often
    isn't, now that it's compared against a sensible paid-transfer
    reference instead of the free-transfer-skewed overall one.
    """
    req = PredictRequest(**payload_dict)
    feature_row, real_data_flags = build_feature_row(req)
    raw_score = float(pipeline.predict(feature_row)[0])
    return explain_prediction(feature_row, raw_score, real_data_flags, top_k=100)


@pytest.fixture
def sample_predict_payload():
    """A valid, realistic PredictRequest body (a Bundesliga attacker to a Premier League club) reused across the predict/compare tests."""
    return {
        "age_at_transfer": 24.0,
        "height_in_cm": 182.0,
        "position": "Attack",
        "sub_position": "Centre-Forward",
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
    """/api/health should respond 200 with status "ok" - basic liveness check."""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_examples_returns_known_transfers_with_breakdown():
    """Every curated homepage card should have a valid score and a fully-described breakdown."""
    res = client.get("/api/examples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    for ex in data:
        assert 0 <= ex["success_score"] <= 100
        # 5 components always; up to 4 FotMob-derived rows (rating/
        # attacking/defensive/possession), each shown only when its own
        # FotMob data is known; "Resale profit" only when the club later
        # resold the player for a known fee - so 5 to 10 total.
        assert 5 <= len(ex["breakdown"]) <= 10
        for component in ex["breakdown"]:
            assert "description" in component and component["description"]


def test_resale_profit_description_matches_actual_profit_or_loss():
    """
    Dembélé's Barcelona spell: bought for €148m, later resold to PSG for
    €50m - a real loss. The description must say "loss", not imply this
    was a good outcome just because a resale happened.
    """
    res = client.get("/api/examples")
    dembele = next(
        ex for ex in res.json()
        if ex["name"] == "Ousmane Dembélé" and ex["to_club"] == "Barcelona"
    )
    resale = next(c for c in dembele["breakdown"] if c["label"] == "Resale profit")
    assert "loss" in resale["description"].lower()
    assert "profitable" not in resale["description"].lower()


def test_sub_million_fee_does_not_round_to_free():
    """
    A €300k resale fee must not display as '€0m' (rounds to zero at 0
    decimal places), which reads as a contradiction next to "profitable".
    """
    res = client.get("/api/transfers/detail", params={"player_id": 106675, "transfer_date": "2020-08-06"})
    assert res.status_code == 200
    resale = next(c for c in res.json()["breakdown"] if c["label"] == "Resale profit")
    assert "€0m" not in resale["description"]
    assert "€0.3m" in resale["description"]


def test_predict_returns_score_range_and_explanation(sample_predict_payload):
    """A valid predict request should return a 0-100 score, a sane [lo, hi] range, and a fully-detailed top-5 explanation."""
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


def test_predict_explanation_shows_league_relative_context(sample_predict_payload):
    """
    'Recent goal contributions per 90' should show the player's output as a
    multiple of their current league's average, not just the raw number -
    consistent with how historical cards explain performance. Checked via
    explain_all() since this feature's contribution shrank (correctly, see
    test_predict_performance_explanation_is_position_conditional) and may
    no longer be in the default top-5 shown by the API.
    """
    all_features = {e["feature"]: e for e in explain_all(sample_predict_payload)}
    ga_row = all_features["pre_ga_p90"]
    assert "league's average" in ga_row["actual_value"]
    assert "x" in ga_row["actual_value"]


def test_predict_performance_explanation_is_position_conditional(sample_predict_payload):
    """
    An attacker's typical goal contributions (~0.47/90) look nothing like
    the whole population's (~0.20/90, dragged down by defenders and
    goalkeepers) - the "typical" comparison should reflect that, not a
    flat cross-position number.
    """
    all_features = {e["feature"]: e for e in explain_all(sample_predict_payload)}
    ga_row = all_features["pre_ga_p90"]
    assert "attacker" in ga_row["detail"]
    assert "0.2" not in ga_row["typical_value"]  # the misleading flat cross-position median


def test_predict_missing_recent_performance_is_imputed_not_zero(sample_predict_payload):
    """
    A player currently at a club outside LEAGUE_MAP (e.g. Messi at Inter
    Miami, Son at LAFC - both MLS) has no real recent-performance data at
    all - pre_apps/pre_minutes/pre_goals_p90/pre_ga_p90/pre_mins_per_app
    arrive as None (see build_lookups.py/search_players()), not a
    fabricated 0. build_feature_row should median-impute a plausible
    position-typical value instead of feeding the model a literal 0 (a
    real, terrible "played 0 minutes recently" signal for a player it
    actually has no data on), and explain_prediction should say plainly
    that the data is missing with an honest 0.0 contribution, rather than
    a backwards-looking swing that happens to fall out of comparing 0
    against a real reference.
    """
    payload = {**sample_predict_payload, "pre_apps": None, "pre_minutes": None,
               "pre_goals_p90": None, "pre_ga_p90": None, "pre_mins_per_app": None}
    req = PredictRequest(**payload)
    feature_row, real_data_flags = build_feature_row(req)
    assert feature_row["pre_apps"].iloc[0] > 0  # imputed with a real position median, not 0
    assert feature_row["pre_minutes"].iloc[0] > 0
    assert real_data_flags["pre_apps"] is False

    raw_score = float(pipeline.predict(feature_row)[0])
    explanation = explain_prediction(feature_row, raw_score, real_data_flags, top_k=100)
    all_features = {e["feature"]: e for e in explanation}
    # pre_apps/pre_minutes/pre_mins_per_app are explained together as one
    # "pre_playing_time" entry (see PLAYING_TIME_FEATURES) - pre_goals_p90/
    # pre_ga_p90 remain independent entries.
    assert all_features["pre_playing_time"]["contribution"] == 0.0
    assert "No recent performance data available" in all_features["pre_playing_time"]["detail"]
    for feat in ["pre_goals_p90", "pre_ga_p90"]:
        assert all_features[feat]["contribution"] == 0.0
        assert "No recent performance data available" in all_features[feat]["detail"]


def test_predict_missing_recent_performance_marginalizes_instead_of_one_guess(sample_predict_payload):
    """
    The *displayed* success_score for a missing-data player should come
    from predict_marginalized_recent_performance (averaging the prediction
    over every real same-position profile), not the plain pipeline.predict()
    on feature_row's single median-imputed guess used for everything else
    (comparables, other features' swap baseline) - a single median point
    isn't neutral, since a tree ensemble's response to a feature isn't
    linear (E[f(X)] != f(E[X]), checked directly with a real case where the
    two differed by ~1.4 points - the gap's size depends on the specific
    combination of other features, so isn't asserted as a fixed margin here).
    """
    payload = {**sample_predict_payload, "pre_apps": None, "pre_minutes": None,
               "pre_goals_p90": None, "pre_ga_p90": None, "pre_mins_per_app": None}
    req = PredictRequest(**payload)
    feature_row, real_data_flags = build_feature_row(req)
    expected_score = round(max(0.0, min(100.0, predict_marginalized_recent_performance(feature_row, "Attack"))), 1)

    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    assert res.json()["success_score"] == pytest.approx(expected_score, abs=0.05)

    # Deterministic - marginalizing over a fixed, precomputed sample set
    # (not live random sampling), so repeating the exact same request must
    # give the exact same score.
    res2 = client.post("/api/predict", json=payload)
    assert res2.json()["success_score"] == res.json()["success_score"]


def test_predict_playing_time_explained_as_one_group_not_backwards():
    """
    pre_apps/pre_minutes/pre_mins_per_app are structurally dependent
    (minutes roughly equals apps times mins_per_app) - explaining
    pre_minutes alone (holding pre_apps/pre_mins_per_app fixed at their
    real 0 values) used to produce a physically impossible synthetic row
    ("0 apps, ~2000 minutes, 0 min/app") that the model extrapolated at
    unpredictably, showing a backwards *positive* contribution for having
    no recent playing time. Swapping all three together should show a
    real, honest *negative* contribution instead - checked directly for a
    real player with real 0 recent apps/minutes/mins_per_app.
    """
    payload = {
        "age_at_transfer": 22.0, "height_in_cm": 178.0, "position": "Midfield",
        "sub_position": "Attacking Midfield", "foot": "left",
        "pre_apps": 0.0, "pre_minutes": 0.0, "pre_goals_p90": 0.0, "pre_ga_p90": 0.0, "pre_mins_per_app": 0.0,
        "transfer_fee": 30_000_000.0, "value_before": 25_000_000.0,
        "from_domestic_competition_id": "L1", "to_domestic_competition_id": "GB1",
        "from_total_market_value": 80_000_000.0, "to_total_market_value": 900_000_000.0,
    }
    all_features = {e["feature"]: e for e in explain_all(payload)}
    assert all_features["pre_playing_time"]["contribution"] < 0
    assert "lowering" in all_features["pre_playing_time"]["detail"]
    assert all_features["pre_playing_time"]["stats"] is not None  # real data - bulleted breakdown, not the "no data" message


def test_predict_fee_explanation_compares_against_value_expectation_not_flat_average(sample_predict_payload):
    """
    A €50m fee for a player worth €60m isn't remarkable - it should be
    compared against what's typically paid for a similarly-valued player
    (fee_regression, fit on paid transfers), not a single flat "typical
    paid fee" that ignores the player's own value entirely.
    """
    all_features = {e["feature"]: e for e in explain_all(sample_predict_payload)}
    fee_row = all_features["log_transfer_fee"]
    assert "similarly-valued player" in fee_row["detail"]
    assert "€0.0m" not in fee_row["detail"]
    ratio_row = all_features["fee_to_value_ratio"]
    assert "paid transfer" in ratio_row["detail"]


def test_predict_fee_explanation_for_a_free_transfer_uses_overall_reference():
    """A genuinely free transfer (fee=0) should display as "free", compared against the overall reference (which is itself mostly free transfers) rather than a nonsensical "paid" comparison."""
    payload = {
        "age_at_transfer": 24.0, "height_in_cm": 182.0, "position": "Attack", "sub_position": "Centre-Forward", "foot": "right",
        "pre_apps": 30.0, "pre_minutes": 2500.0, "pre_goals_p90": 0.5, "pre_ga_p90": 0.7, "pre_mins_per_app": 83.0,
        "transfer_fee": 0.0, "value_before": 60_000_000.0,
        "from_domestic_competition_id": "L1", "to_domestic_competition_id": "GB1",
        "from_total_market_value": 400_000_000.0, "to_total_market_value": 900_000_000.0,
    }
    all_features = {e["feature"]: e for e in explain_all(payload)}
    fee_row = all_features["log_transfer_fee"]
    assert fee_row["actual_value"] == "free"
    assert "similarly-valued player" not in fee_row["detail"]


def test_predict_fee_explanation_reflects_players_own_value():
    """
    The exact scenario that motivated this fix: a €100m fee for a player
    already worth €70m is a modest premium (~1.4x), not a wild outlier -
    the regression-based reference should land close to the actual value
    (~€73m expected), not a flat low number.
    """
    payload = {
        "age_at_transfer": 25.0, "height_in_cm": 182.0, "position": "Attack", "sub_position": "Centre-Forward", "foot": "right",
        "pre_apps": 30.0, "pre_minutes": 2500.0, "pre_goals_p90": 0.5, "pre_ga_p90": 0.7, "pre_mins_per_app": 83.0,
        "transfer_fee": 100_000_000.0, "value_before": 70_000_000.0,
        "from_domestic_competition_id": "ES1", "to_domestic_competition_id": "GB1",
        "from_total_market_value": 400_000_000.0, "to_total_market_value": 900_000_000.0,
    }
    all_features = {e["feature"]: e for e in explain_all(payload)}
    fee_row = all_features["log_transfer_fee"]
    typical_fee_m = float(fee_row["typical_value"].strip("€m"))
    assert 50 < typical_fee_m < 100  # in the same ballpark as the player's own €70m value


def test_predict_rejects_invalid_payload():
    """Pydantic validation should reject an out-of-range/incomplete request (age=5 violates the ge=15 constraint) with a 422, not a 500."""
    res = client.post("/api/predict", json={"age_at_transfer": 5})
    assert res.status_code == 422


def test_compare_returns_both_results_and_delta(sample_predict_payload):
    """/api/compare should score both scenarios and report a delta consistent with their individual scores."""
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
    """Searching the plain-ASCII 'Dembele' should still find the accented 'Dembélé'."""
    res = client.get("/api/players/search", params={"q": "Dembele"})
    assert res.status_code == 200
    names = [p["name"] for p in res.json()]
    assert any("Dembélé" in n for n in names)


def test_players_search_response_has_no_internal_fields():
    """The internal _name_fold helper column must never leak into the API response."""
    res = client.get("/api/players/search", params={"q": "Haaland"})
    for p in res.json():
        assert "_name_fold" not in p


def test_players_search_sends_null_not_zero_for_uncovered_league_players():
    """
    A player currently at a club outside LEAGUE_MAP (e.g. Messi at Inter
    Miami - MLS isn't one of the 23 tracked leagues) has no real recent-
    performance data - the search response must send JSON null so the
    frontend/build_feature_row can tell "no data" apart from a real 0,
    not a fabricated 0.0 that looks like this player genuinely played
    zero minutes recently (see build_lookups.py).
    """
    res = client.get("/api/players/search", params={"q": "Lionel Messi"})
    matches = [p for p in res.json() if p["name"] == "Lionel Messi"]
    assert len(matches) == 1
    assert matches[0]["recent_apps"] is None
    assert matches[0]["recent_minutes"] is None


def test_players_search_limit_is_clamped():
    """
    limit is a raw client-supplied query param with no FastAPI-level bound
    (unlike /api/transfers'/loans' limit, which clamp to [1, 100]) - an
    oversized value must still cap at 50 rather than dumping the whole
    lookup table, and a non-positive one must still return at least 1 row
    rather than pandas' .head(0)/.head(-n) surprise.
    """
    res = client.get("/api/players/search", params={"q": "an", "limit": 999_999})
    assert res.status_code == 200
    assert len(res.json()) <= 50

    res = client.get("/api/players/search", params={"q": "an", "limit": -5})
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_clubs_search_limit_is_clamped():
    """Same clamp as test_players_search_limit_is_clamped, for the other unbounded search endpoint."""
    res = client.get("/api/clubs/search", params={"q": "an", "limit": 999_999})
    assert res.status_code == 200
    assert len(res.json()) <= 50

    res = client.get("/api/clubs/search", params={"q": "an", "limit": -5})
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_clubs_search_is_accent_insensitive():
    """Searching 'atletico' should still find the accented 'Atlético'."""
    res = client.get("/api/clubs/search", params={"q": "atletico"})
    assert res.status_code == 200
    names = [c["name"] for c in res.json()]
    assert any("Atlético" in n for n in names)


def test_clubs_search_response_has_no_internal_fields():
    """The internal _name_fold helper column must never leak into the API response."""
    res = client.get("/api/clubs/search", params={"q": "Real"})
    for c in res.json():
        assert "_name_fold" not in c


def test_filters_endpoint():
    """/api/filters should list at least one position and one league, each with an id and a display name."""
    res = client.get("/api/filters")
    assert res.status_code == 200
    data = res.json()
    assert "Attack" in data["positions"]
    assert len(data["leagues"]) > 0
    assert all("id" in l and "name" in l for l in data["leagues"])


def test_league_names_disambiguate_duplicates():
    """Austria's and Germany's Bundesliga must not display identically (see competitions_df's display_name logic)."""
    res = client.get("/api/filters")
    leagues = {l["id"]: l["name"] for l in res.json()["leagues"]}
    assert leagues["A1"] != leagues["L1"]
    assert "Bundesliga" in leagues["A1"] and "Bundesliga" in leagues["L1"]


def test_transfers_list_pagination():
    """A limit=10 request should return exactly 10 results, with the true total count reported separately."""
    res = client.get("/api/transfers", params={"limit": 10, "offset": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 1000
    assert len(data["results"]) == 10


def test_transfers_list_position_filter():
    """Filtering by position=Goalkeeper should return only goalkeepers."""
    res = client.get("/api/transfers", params={"position": "Goalkeeper", "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(r["position"] == "Goalkeeper" for r in data["results"])


def test_transfers_list_search_is_accent_insensitive():
    """The browse page's search box should match accented names via the plain-ASCII query too."""
    res = client.get("/api/transfers", params={"q": "Dembele"})
    data = res.json()
    assert data["total"] > 0
    assert any("Dembélé" in r["name"] for r in data["results"])


def test_transfers_list_sorted_descending_by_default():
    """With no explicit sort params, results should default to success_score descending."""
    res = client.get("/api/transfers", params={"limit": 20})
    scores = [r["success_score"] for r in res.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_transfers_list_includes_player_id_for_detail_lookup():
    """Each row needs player_id so the browse page's click-to-view-card feature can call /api/transfers/detail."""
    res = client.get("/api/transfers", params={"q": "Haaland"})
    row = res.json()["results"][0]
    assert "player_id" in row
    detail = client.get("/api/transfers/detail", params={
        "player_id": row["player_id"], "transfer_date": row["transfer_date"],
    })
    assert detail.status_code == 200
    assert detail.json()["name"] == row["name"]


def test_transfer_detail_matches_examples_card_shape():
    """The detail endpoint and /api/examples share build_transfer_card(), so the same transfer must produce byte-identical cards from either route."""
    examples_res = client.get("/api/examples")
    haaland = next(e for e in examples_res.json() if e["name"] == "Erling Haaland")
    detail_res = client.get("/api/transfers/detail", params={
        "player_id": haaland["player_id"], "transfer_date": haaland["transfer_date"],
    })
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["success_score"] == haaland["success_score"]
    assert detail["breakdown"] == haaland["breakdown"]


def test_transfer_detail_404_for_unknown_transfer():
    """A (player_id, transfer_date) pair that doesn't exist should 404, not 500 or return an empty/malformed card."""
    res = client.get("/api/transfers/detail", params={"player_id": 999999999, "transfer_date": "2020-01-01"})
    assert res.status_code == 404


def test_loans_filters_endpoint():
    """/api/loans/filters should list at least one position and one league, each with an id and a display name."""
    res = client.get("/api/loans/filters")
    assert res.status_code == 200
    data = res.json()
    assert len(data["positions"]) > 0
    assert len(data["leagues"]) > 0
    assert all("id" in l and "name" in l for l in data["leagues"])


def test_loans_list_pagination():
    """A limit=10 request should return exactly 10 results, with the true total count reported separately."""
    res = client.get("/api/loans", params={"limit": 10, "offset": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 50
    assert len(data["results"]) == 10


def test_loans_list_sorted_descending_by_default():
    """With no explicit sort params, results should default to loan_success_score descending."""
    res = client.get("/api/loans", params={"limit": 20})
    scores = [r["loan_success_score"] for r in res.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_loans_list_position_filter():
    """Filtering by position=Goalkeeper should return only goalkeepers."""
    res = client.get("/api/loans", params={"position": "Goalkeeper", "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(r["position"] == "Goalkeeper" for r in data["results"])


def test_loans_list_search_is_accent_insensitive():
    """The loans page's search box should match accented names via the plain-ASCII query too."""
    res = client.get("/api/loans", params={"q": "Lossl"})
    data = res.json()
    assert data["total"] > 0
    assert any("Lössl" in r["name"] for r in data["results"])


def test_loan_detail_matches_list_card_shape():
    """/api/loans/detail should return a 4-8 component card (no value-for-money/resale-profit rows, but up to 4 FotMob rows) matching the list row it came from."""
    list_res = client.get("/api/loans", params={"limit": 1})
    row = list_res.json()["results"][0]
    detail_res = client.get("/api/loans/detail", params={
        "player_id": row["player_id"], "transfer_date": row["transfer_date"],
    })
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["name"] == row["name"]
    assert detail["loan_success_score"] == row["loan_success_score"]
    assert 4 <= len(detail["breakdown"]) <= 8
    for component in detail["breakdown"]:
        assert "description" in component and component["description"]


def test_loan_detail_404_for_unknown_loan():
    """A (player_id, transfer_date) pair that doesn't exist in loans_processed.csv should 404, not 500."""
    res = client.get("/api/loans/detail", params={"player_id": 999999999, "transfer_date": "2020-01-01"})
    assert res.status_code == 404


def test_surprises_list_pagination():
    """A limit=10 request should return exactly 10 results, with the true total count reported separately."""
    res = client.get("/api/surprises", params={"limit": 10, "offset": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 1000
    assert len(data["results"]) == 10


def test_surprises_list_sorted_descending_by_default():
    """With no explicit sort params, results should default to surprise_delta descending - the biggest overachievers first."""
    res = client.get("/api/surprises", params={"limit": 20})
    deltas = [r["surprise_delta"] for r in res.json()["results"]]
    assert deltas == sorted(deltas, reverse=True)
    for r in res.json()["results"]:
        assert r["surprise_delta"] == pytest.approx(r["success_score"] - r["predicted_score"], abs=0.05)


def test_surprises_list_ascending_surfaces_busts():
    """order=asc should surface the biggest busts - a large *negative* surprise_delta - not just reverse into small positives."""
    res = client.get("/api/surprises", params={"sort": "surprise_delta", "order": "asc", "limit": 5})
    deltas = [r["surprise_delta"] for r in res.json()["results"]]
    assert deltas == sorted(deltas)
    assert deltas[0] < -20


def test_surprises_list_position_filter():
    """Filtering by position=Goalkeeper should return only goalkeepers."""
    res = client.get("/api/surprises", params={"position": "Goalkeeper", "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(r["position"] == "Goalkeeper" for r in data["results"])


def test_surprises_list_search_is_accent_insensitive():
    """The search box should match accented names via the plain-ASCII query too."""
    res = client.get("/api/surprises", params={"q": "Dembele"})
    data = res.json()
    assert data["total"] > 0
    assert any("Dembélé" in r["name"] for r in data["results"])


def test_surprises_list_excludes_transfers_with_no_prediction():
    """
    A transfer missing a required model feature (fee, height, origin league)
    never got a held-out prediction (see scripts/compute_prediction_surprises.py)
    and must not appear here with a fabricated score.
    """
    res = client.get("/api/surprises", params={"limit": 100})
    for r in res.json()["results"]:
        assert r["predicted_score"] is not None


def test_surprises_detail_reuses_transfer_detail_card():
    """The surprises page's modal reuses /api/transfers/detail for the breakdown - the same (player_id, transfer_date) key must resolve there too."""
    res = client.get("/api/surprises", params={"limit": 1})
    row = res.json()["results"][0]
    detail = client.get("/api/transfers/detail", params={
        "player_id": row["player_id"], "transfer_date": row["transfer_date"],
    })
    assert detail.status_code == 200
    assert detail.json()["success_score"] == row["success_score"]
