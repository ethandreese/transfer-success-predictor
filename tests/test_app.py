"""
End-to-end tests for the FastAPI app, using the already-committed model
and data artifacts (app/model/*.joblib, data/*.csv) - no dependency on the
raw Transfermarkt dataset.
"""
import csv
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import (
    CLUB_NAME_ALIASES, MIN_LEAGUE_TRANSFERS, app, build_feature_row, eur_m, explain_prediction, league_trends_df,
    numeric_column, pipeline, predict_marginalized_recent_performance, PredictRequest, transfers_df,
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
        if ex["name"] == "Ousmane Dembélé" and ex["to_club"] == "FC Barcelona"
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


def test_predict_handles_composite_with_one_missing_raw_substat(sample_predict_payload):
    """
    A composite (e.g. "defensive") can have real data overall from some of
    its raw sub-stats while one specific one is still missing - e.g. an
    attacker with real FotMob tackle/interception/recovery numbers but no
    recorded clearances at all, since attackers rarely attempt any.
    format_feature_value crashed trying to format that individual raw
    stat's None, turning this completely normal partial-coverage case
    (not a rare edge case - the README documents FotMob bucket coverage
    at only ~54-67% even among transfers with *some* FotMob data) into a
    400 on the whole prediction. Caught testing 3-way Compare by hand.
    """
    payload = dict(
        sample_predict_payload,
        pre_fotmob_total_tackle=0.5, pre_fotmob_interception=0.3,
        pre_fotmob_ball_recovery=2.0, pre_fotmob_effective_clearance=None,
    )
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200

    defensive = next(e for e in explain_all(payload) if e["stats"] and any("Tackles" in s for s in e["stats"]))
    assert any("Interceptions" in s for s in defensive["stats"])
    assert not any("Clearances" in s for s in defensive["stats"])


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


def test_compare_returns_one_result_per_scenario_in_order(sample_predict_payload):
    """/api/compare should score every scenario and return them in the same order, each carrying its own label."""
    other = dict(sample_predict_payload, transfer_fee=10_000_000.0)
    res = client.post("/api/compare", json={
        "scenarios": [
            {"request": sample_predict_payload, "label": "Scenario A"},
            {"request": other, "label": "Scenario B"},
        ],
    })
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 2
    assert [r["label"] for r in results] == ["Scenario A", "Scenario B"]
    assert all("success_score" in r for r in results)


def test_compare_supports_three_and_four_scenarios(sample_predict_payload):
    """/api/compare's whole point past the original two-scenario version is supporting more than a pair - both 3-way and 4-way must work."""
    for n in (3, 4):
        scenarios = [
            {"request": dict(sample_predict_payload, transfer_fee=float(i) * 1_000_000), "label": f"Option {i}"}
            for i in range(n)
        ]
        res = client.post("/api/compare", json={"scenarios": scenarios})
        assert res.status_code == 200, f"{n}-way compare failed: {res.text}"
        assert len(res.json()["results"]) == n


def test_compare_rejects_fewer_than_two_or_more_than_four_scenarios(sample_predict_payload):
    """A single scenario isn't a comparison, and the compare form only ever shows up to 4 columns - both ends should 422, not silently truncate or score just one side."""
    one = {"scenarios": [{"request": sample_predict_payload, "label": "Solo"}]}
    assert client.post("/api/compare", json=one).status_code == 422

    five = {"scenarios": [
        {"request": sample_predict_payload, "label": f"Option {i}"} for i in range(5)
    ]}
    assert client.post("/api/compare", json=five).status_code == 422


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


def test_transfers_list_fee_and_age_range_filters():
    """min_fee/max_fee/min_age/max_age should each narrow the results to exactly that range, and a free/undisclosed-fee transfer should never appear once a fee bound is set (NaN can't be judged inside or outside a range it doesn't have a number for)."""
    res = client.get("/api/transfers", params={"min_fee": 50_000_000, "max_fee": 100_000_000, "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(50_000_000 <= r["transfer_fee"] <= 100_000_000 for r in data["results"])

    res2 = client.get("/api/transfers", params={"min_age": 30, "max_age": 32, "limit": 50})
    data2 = res2.json()
    assert data2["total"] > 0
    assert all(30 <= r["age_at_transfer"] <= 32 for r in data2["results"])


def test_transfers_export_matches_the_current_filters_as_csv():
    """/api/transfers/export must return every transfer matching the filters (not paginated like /api/transfers) as a real CSV, so a filtered Browse view can be exported wholesale."""
    res = client.get("/api/transfers/export", params={"position": "Goalkeeper"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert {"player_id", "name", "position", "success_score"} <= set(rows[0].keys())
    assert all(row["position"] == "Goalkeeper" for row in rows)

    list_total = client.get("/api/transfers", params={"position": "Goalkeeper", "limit": 1}).json()["total"]
    assert len(rows) == list_total  # every matching row, not just one page


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


def test_loans_list_age_and_duration_range_filters():
    """min_age/max_age/min_duration/max_duration should each narrow the results to exactly that range."""
    res = client.get("/api/loans", params={"min_age": 25, "max_age": 28, "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(25 <= r["age_at_transfer"] <= 28 for r in data["results"])

    res2 = client.get("/api/loans", params={"min_duration": 150, "max_duration": 200, "limit": 50})
    data2 = res2.json()
    assert data2["total"] > 0
    assert all(150 <= r["tenure_days"] <= 200 for r in data2["results"])


def test_loans_export_matches_the_current_filters_as_csv():
    """/api/loans/export must return every loan matching the filters (not paginated like /api/loans) as a real CSV."""
    res = client.get("/api/loans/export", params={"position": "Defender"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert {"player_id", "name", "position", "loan_success_score", "converted_to_permanent"} <= set(rows[0].keys())
    assert all(row["position"] == "Defender" for row in rows)

    list_total = client.get("/api/loans", params={"position": "Defender", "limit": 1}).json()["total"]
    assert len(rows) == list_total


def test_loan_conversion_detected_for_a_real_loan_to_buy():
    """
    Timur Suleymanov's 2023-09-14 loan from Pari NN to Loko Moscow was
    followed by a genuine permanent transfer between the same two clubs on
    2024-07-01 - a real loan-to-buy in the committed dataset, not a
    fabricated example. converted_to_permanent must flag it and surface
    that later transfer's date/score.
    """
    res = client.get("/api/loans", params={"q": "Suleymanov", "limit": 5})
    loan = next(r for r in res.json()["results"] if r["transfer_date"] == "2023-09-14")
    assert loan["converted_to_permanent"] is True
    assert loan["conversion_transfer_date"] == "2024-07-01"
    assert loan["conversion_success_score"] is not None

    detail = client.get("/api/loans/detail", params={
        "player_id": loan["player_id"], "transfer_date": loan["transfer_date"],
    })
    assert detail.json()["converted_to_permanent"] is True


def test_loan_conversion_none_for_a_synthetic_loan_with_no_match():
    """A loan whose player/from-club/to-club combination has no later transfers_df row at all must return None, not a stale/wrong match against an unrelated transfer."""
    from app.main import find_loan_conversion
    fake_loan = pd.Series({
        "player_id": -1, "from_club_name": "Nowhere FC", "to_club_name": "Nowhere Else FC",
        "transfer_date": "2020-01-01",
    })
    assert find_loan_conversion(fake_loan) is None


def test_loan_conversion_requires_the_same_from_and_to_club_not_just_the_player():
    """A player permanently transferring to a *third* club after a loan isn't a conversion of that loan - find_loan_conversion must require both from_club_name and to_club_name to match, not just player_id and a later date."""
    from app.main import find_loan_conversion, transfers_df
    real_transfer = transfers_df.iloc[0]
    fake_loan = pd.Series({
        "player_id": real_transfer["player_id"],
        "from_club_name": real_transfer["from_club_name"],
        "to_club_name": "A Club This Player Was Never Loaned To",
        "transfer_date": "2000-01-01",  # earlier than virtually every real transfer_date
    })
    assert find_loan_conversion(fake_loan) is None


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


def test_surprises_scatter_returns_every_prediction_unfiltered_and_unpaginated():
    """/api/surprises/scatter backs the Model vs Reality chart - it should return the full set of predicted transfers (matching /api/surprises' total with no filters), not a paginated page of it."""
    listing = client.get("/api/surprises", params={"limit": 1})
    total = listing.json()["total"]
    res = client.get("/api/surprises/scatter")
    assert res.status_code == 200
    data = res.json()
    assert len(data["name"]) == total
    for key in ("player_id", "transfer_date", "predicted_score", "success_score", "surprise_delta"):
        assert len(data[key]) == total


def test_surprises_scatter_delta_matches_actual_minus_predicted():
    """surprise_delta in the scatter payload should agree with success_score - predicted_score, same invariant /api/surprises' own results satisfy."""
    res = client.get("/api/surprises/scatter")
    data = res.json()
    for i in range(0, len(data["name"]), 500):
        assert data["surprise_delta"][i] == pytest.approx(data["success_score"][i] - data["predicted_score"][i], abs=0.05)


def test_clubs_leaderboard_pagination():
    """A limit=10 request should return exactly 10 results, with the true total count reported separately."""
    res = client.get("/api/clubs/leaderboard", params={"limit": 10, "offset": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 100
    assert len(data["results"]) == 10


def test_clubs_leaderboard_default_sort_respects_minimum_sample():
    """Ranking by avg_incoming_score (the default) must exclude any club below MIN_CLUB_TRANSFERS incoming transfers - a club with one lucky signing shouldn't top the list."""
    res = client.get("/api/clubs/leaderboard", params={"limit": 50})
    data = res.json()
    scores = [r["avg_incoming_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)
    for r in data["results"]:
        assert r["transfers_in"] >= 5


def test_clubs_leaderboard_resale_profit_sort_respects_its_own_minimum():
    """Ranking by avg_resale_profit_pct must exclude clubs below MIN_CLUB_RESALES resold transfers, independently of the incoming-transfer threshold."""
    res = client.get("/api/clubs/leaderboard", params={"sort": "avg_resale_profit_pct", "order": "desc", "limit": 50})
    data = res.json()
    pcts = [r["avg_resale_profit_pct"] for r in data["results"]]
    assert pcts == sorted(pcts, reverse=True)
    for r in data["results"]:
        assert r["resales_count"] >= 3


def test_clubs_leaderboard_resale_profit_direction_is_the_buying_club():
    """
    Resale profit belongs to the club that bought the player (to_club_name)
    and later resold him on, not the club that originally sold him to that
    buyer (from_club_name) - next_transfer_fee is what a third club paid
    the *buyer*, not anything about the original seller (see
    build_club_report_cards). Real Madrid bought Ronaldo from Man Utd in
    2009 and sold him to Juventus in 2018 for a real, large fee - his
    transfer to Juventus must show up as one of *Juventus's* incoming
    transfers, not attributed to Real Madrid's resale record.
    """
    res = client.get("/api/clubs/leaderboard", params={"q": "Real Madrid", "sort": "total_spent"})
    real_madrid = next(r for r in res.json()["results"] if r["club_name"] == "Real Madrid")
    flip_names = {real_madrid["best_flip"]["name"] if real_madrid["best_flip"] else None,
                  real_madrid["worst_flip"]["name"] if real_madrid["worst_flip"] else None}
    assert "Cristiano Ronaldo" not in flip_names


def test_clubs_leaderboard_best_and_worst_flip_are_the_real_extremes():
    """best_flip/worst_flip and total_resale_profit are all derived from the same per-club profit Series (computed once - see build_club_report_cards) - best_flip's own profit must be >= worst_flip's, and both must be real resales this club actually made."""
    res = client.get("/api/clubs/leaderboard", params={"q": "Real Madrid", "sort": "total_spent"})
    real_madrid = next(r for r in res.json()["results"] if r["club_name"] == "Real Madrid")
    assert real_madrid["resales_count"] >= 2
    best, worst = real_madrid["best_flip"], real_madrid["worst_flip"]
    best_profit = best["fee_received"] - best["fee_paid"]
    worst_profit = worst["fee_received"] - worst["fee_paid"]
    assert best_profit >= worst_profit


def test_clubs_leaderboard_search_filters_by_name():
    """The search box should filter to clubs whose name contains the query."""
    res = client.get("/api/clubs/leaderboard", params={"q": "Real Madrid"})
    data = res.json()
    assert data["total"] >= 1
    assert all("real madrid" in r["club_name"].lower() for r in data["results"])


def test_clubs_leaderboard_league_filter():
    """Filtering by league=GB1 should return only clubs whose primary league is the Premier League."""
    res = client.get("/api/clubs/leaderboard", params={"league": "GB1", "limit": 50})
    data = res.json()
    assert data["total"] > 0
    assert all(r["league"] == "Premier League" for r in data["results"])


def test_clubs_leaderboard_row_has_no_second_request_needed_fields():
    """Every row must already carry its own best/worst signing so the frontend's click-to-view-card modal needs no second request."""
    res = client.get("/api/clubs/leaderboard", params={"limit": 5})
    for r in res.json()["results"]:
        assert "name" in r["best_signing"] and "success_score" in r["best_signing"]
        assert "name" in r["worst_signing"] and "success_score" in r["worst_signing"]


def test_clubs_leaderboard_highlights_are_clickable_into_transfer_detail():
    """Every highlight (signing/flip/departure) must carry a player_id, and that (player_id, transfer_date) pair must resolve on /api/transfers/detail - the report card's highlights link into the same real transfer record the rest of the site uses."""
    res = client.get("/api/clubs/leaderboard", params={"q": "Real Madrid", "sort": "total_spent"})
    real_madrid = next(r for r in res.json()["results"] if r["club_name"] == "Real Madrid")
    for key in ["best_signing", "worst_signing", "best_flip", "worst_flip", "best_departure", "worst_departure"]:
        highlight = real_madrid[key]
        assert highlight is not None, key
        assert "player_id" in highlight, key
        detail = client.get("/api/transfers/detail", params={
            "player_id": highlight["player_id"], "transfer_date": highlight["transfer_date"],
        })
        assert detail.status_code == 200, key
        assert detail.json()["name"] == highlight["name"], key


def test_club_report_card_matches_the_leaderboard_row_for_the_same_club():
    """/api/clubs/report-card?name=X should agree with /api/clubs/leaderboard's own row for that club on every shared field - both come from club_row_dict, so they must never drift apart."""
    leaderboard = client.get("/api/clubs/leaderboard", params={"q": "Real Madrid", "sort": "total_spent"})
    row = next(r for r in leaderboard.json()["results"] if r["club_name"] == "Real Madrid")
    res = client.get("/api/clubs/report-card", params={"name": "Real Madrid"})
    assert res.status_code == 200
    card = res.json()
    for key in row:
        assert card[key] == row[key], key


def test_club_report_card_404_for_unknown_club():
    res = client.get("/api/clubs/report-card", params={"name": "Definitely Not A Real Club FC"})
    assert res.status_code == 404


def test_club_report_card_by_year_excludes_the_current_in_progress_year():
    """The by_year spend-vs-quality series should never include the dataset's own most recent (in-progress) year - same convention as League Trends' by_year."""
    current_year = int(pd.to_datetime(transfers_df["transfer_date"]).dt.year.max())
    res = client.get("/api/clubs/report-card", params={"name": "Real Madrid"})
    by_year = res.json()["by_year"]
    assert by_year, "Real Madrid should have real year-by-year data"
    assert all(entry["year"] < current_year for entry in by_year)
    for entry in by_year:
        assert entry["transfers"] > 0
        assert entry["avg_score"] is not None


def test_club_report_card_position_breakdown_respects_minimum_sample():
    """Every position bucket in position_breakdown should meet MIN_CLUB_POSITION_SAMPLE - a position with only 1-2 transfers is too noisy to show."""
    res = client.get("/api/clubs/report-card", params={"name": "Real Madrid"})
    breakdown = res.json()["position_breakdown"]
    assert breakdown
    for entry in breakdown:
        assert entry["transfers"] >= 3


def test_club_report_card_search_is_not_limited_by_sample_size():
    """/api/clubs/report-card-search must not apply /api/clubs/leaderboard's minimum-sample sort filters - a small club should still be findable for the head-to-head comparison."""
    res = client.get("/api/clubs/report-card-search", params={"q": "Real Madrid"})
    assert res.status_code == 200
    names = [r["club_name"] for r in res.json()]
    assert "Real Madrid" in names


def test_club_report_card_search_short_query_returns_empty():
    res = client.get("/api/clubs/report-card-search", params={"q": "a"})
    assert res.json() == []


def test_eur_m_formats_billions_above_the_threshold():
    """No individual transfer fee reaches a billion, but a club's aggregate spend (see build_club_report_cards) can - eur_m() should switch to 'b' at that point instead of an unwieldy 4-digit million count."""
    assert eur_m(2_049_250_000) == "€2.05b"
    assert eur_m(999_999_999) == "€1000m"
    assert eur_m(1_000_000_000) == "€1.00b"


def test_club_name_aliases_merge_known_legal_suffix_variants():
    """Barcelona's transfer history is split between 'Barcelona' and 'FC Barcelona' in the raw data - both must resolve to the same canonical club so its report card isn't missing half its transfers."""
    assert CLUB_NAME_ALIASES.get("FC Barcelona", "FC Barcelona") == CLUB_NAME_ALIASES.get("Barcelona", "Barcelona")
    res = client.get("/api/clubs/leaderboard", params={"q": "Barcelona"})
    names = {r["club_name"] for r in res.json()["results"]}
    assert "Barcelona" not in names  # merged into "FC Barcelona", not its own separate row
    assert "FC Barcelona" in names


def test_club_name_aliases_prefer_the_longer_spelling():
    """Given two spellings of the same club, the canonical one should be whichever is longer ('Arsenal FC' over 'Arsenal', 'Tottenham Hotspur' over 'Tottenham') - the fuller name a reader unfamiliar with the shorthand is more likely to recognize."""
    assert CLUB_NAME_ALIASES["Arsenal"] == "Arsenal FC"
    assert CLUB_NAME_ALIASES["Tottenham"] == "Tottenham Hotspur"
    assert CLUB_NAME_ALIASES["Man City"] == "Manchester City"
    assert CLUB_NAME_ALIASES["Man Utd"] == "Manchester United"


def test_club_name_aliases_does_not_merge_distinct_clubs():
    """'SC Dnipro-1' is a real, distinct club from the dissolved 'Dnipro Dnipropetrovsk' - the alias builder must not merge them just because they share a city name."""
    assert CLUB_NAME_ALIASES.get("SC Dnipro-1", "SC Dnipro-1") != CLUB_NAME_ALIASES.get("Dnipro", "Dnipro")


@pytest.mark.parametrize("a,b", [
    ("Man City", "Manchester City"),
    ("Man Utd", "Manchester United"),
    ("PSG", "Paris Saint-Germain"),
    ("Paris SG", "Paris Saint-Germain"),
    ("Bor. Dortmund", "Dortmund"),
    ("Borussia Dortmund", "Dortmund"),
    ("Tottenham", "Tottenham Hotspur"),
    ("Newcastle", "Newcastle United"),
    ("West Ham", "West Ham United"),
    ("West Brom", "West Bromwich Albion"),
    ("Brighton", "Brighton & Hove Albion"),
    ("Leeds", "Leeds United"),
    ("Leicester", "Leicester City"),
    ("AS Monaco", "Monaco"),
    ("Lyon", "Olympique Lyon"),
    ("Marseille", "Olympique Marseille"),
    ("LOSC Lille", "Lille"),
    ("Nice", "OGC Nice"),
    ("Real Betis", "Real Betis Balompié"),
    ("Athletic Bilbao", "Athletic Club"),
    ("Ajax", "Ajax Amsterdam"),
    ("Feyenoord", "Feyenoord Rotterdam"),
    ("PSV", "PSV Eindhoven"),
    ("Benfica", "SL Benfica"),
    ("Espanyol", "RCD Espanyol Barcelona"),
    ("Hamburg", "Hamburger SV"),
    ("Zenit S-Pb", "AO FK Zenit Sankt-Peterburg"),
    ("Shakhtar D.", "FC Shakhtar Donetsk"),
    ("Sporting", "Sporting CP"),
    ("Inter", "Inter Milan"),
    # Second sweep, triggered by "Swansea"/"Swansea City" still showing as
    # two rows on /clubs.html - see the comment above CLUB_NICKNAME_GROUPS.
    ("Swansea", "Swansea City"),
    ("Cardiff", "Cardiff City"),
    ("Norwich", "Norwich City"),
    ("Wigan", "Wigan Athletic"),
    ("Huddersfield", "Huddersfield Town"),
    ("Wolves", "Wolverhampton Wanderers"),
    ("QPR", "Queens Park Rangers"),
    ("Nottingham Forest", "Nott'm Forest"),
    ("Frankfurt", "Eintracht Frankfurt"),
    ("Mönchengladbach", "Borussia Mönchengladbach"),
    ("Hoffenheim", "TSG 1899 Hoffenheim"),
    ("Atlético", "Atlético Madrid"),
    ("Lazio", "Società Sportiva Lazio S.p.A."),
    ("Panathinaikos", "Panathinaikos Athlitikos Omilos"),
    ("Willem II", "Willem II Tilburg"),
    ("Karabükspor", "Kardemir Karabükspor"),
    ("Ankaragücü", "MKE Ankaragücü"),
    ("Leipzig", "RB Leipzig"),
    ("Salzburg", "RB Salzburg"),
    ("Montpellier", "Montpellier HSC"),
    ("FC Twente", "FC Twente Enschede"),
    ("Estoril", "GD Estoril Praia"),
    ("Vitesse", "Vitesse Arnhem"),
    ("Excelsior", "Excelsior Rotterdam"),
    ("Panionios", "Panionios Athens"),
    ("SönderjyskE", "Sönderjyske Fodbold"),
    ("Roda JC", "Roda JC Kerkrade"),
    ("Dnipro", "Dnipro Dnipropetrovsk (-2020)"),
    ("Kryvbas", "Kryvbas Kryvyi Rig"),
    ("Belenenses", "CF Os Belenenses"),
    ("Chornomorets", "Chornomorets Odesa"),
    ("Panetolikos", "Panetolikos Agrinio"),
    ("Guingamp", "EA Guingamp"),
    ("Marítimo", "CS Marítimo"),
    ("Iraklis", "Iraklis Thessaloniki"),
    ("De Graafschap", "De Graafschap Doetinchem"),
    ("Dinamo Zagreb", "GNK Dinamo Zagreb"),
    ("Xanthi", "AO Xanthi"),
    ("Karagümrük", "Fatih Karagümrük"),
    ("Coimbra", "Académica Coimbra"),
    ("Basel", "FC Basel 1893"),
    ("Red Star", "Red Star Belgrade"),
    ("Malmö", "Malmö FF"),
    ("Young Boys", "BSC Young Boys"),
    ("Greuther Fürth", "SpVgg Greuther Fürth"),
    ("Rosenborg", "Rosenborg BK"),
    ("Sarpsborg 08", "Sarpsborg 08 Fotballforening"),
    ("Ergotelis", "GS Ergotelis"),
    ("Ingulets", "Ingulets Petrove"),
    ("Clermont Foot", "Clermont Foot 63"),
    ("Desna", "Desna Chernigiv"),
    ("Karpaty Lviv", "Karpaty Lviv (-2021)"),
    ("Anzhi", "Anzhi Makhachkala ( -2022)"),
    ("Mordovia", "Mordovia Saransk (-2020)"),
    ("Mouscron", "Royal Excel Mouscron (-2022)"),
    ("SC Paderborn", "SC Paderborn 07"),
    ("Metalist Kharkiv", "Metalist Kharkiv (- 2016)"),
    ("Roma", "Associazione Sportiva Roma"),
    ("Leverkusen", "Bayer 04 Leverkusen"),
    ("Parma", "Parma Calcio 1913"),
    ("Atromitos", "APS Atromitos Athinon"),
    ("PAOK", "PAOK Salonika"),
    ("Akhmat Grozny", "RFK Akhmat Grozny"),
    ("Alavés", "Deportivo Alavés"),
    ("Stade Brestois", "Stade Brestois 29"),
    ("Troyes", "ESTAC Troyes"),
    ("Ural", "Ural Yekaterinburg"),
    ("Salernitana", "US Salernitana 1919"),
    ("Levadiakos", "APO Levadiakos Football Club"),
    ("Kuban Krasnodar", "Kuban Krasnodar (-2018)"),
    ("Dijon", "Dijon FCO"),
    ("Veres Rivne", "NK Veres Rivne"),
    ("SC Cambuur", "SC Cambuur Leeuwarden"),
    ("Athletic", "Athletic Bilbao"),
    ("Évian", "Thonon Évian Grand Genève FC"),
    ("Partizan", "FK Partizan Belgrade"),
])
def test_club_name_aliases_merge_verified_nickname_pairs(a, b):
    """
    Each of these pairs was individually verified (same domestic
    competition on both sides, non-contradictory transfer_date range - see
    CLUB_NICKNAME_GROUPS in app/main.py) to be the same real club under a
    nickname/official-name split that CLUB_NAME_STRIP_TOKENS' mechanical
    pass can't catch on its own (no shared token). Both spellings must
    resolve to the same canonical club.
    """
    assert CLUB_NAME_ALIASES.get(a, a) == CLUB_NAME_ALIASES.get(b, b)


@pytest.mark.parametrize("a,b", [
    ("Genoa", "Genoa CFC"),
    ("Fiorentina", "ACF Fiorentina"),
    ("Besiktas", "Beşiktaş Jimnastik Kulübü"),
    ("Napoli", "SSC Napoli"),
    ("Atalanta", "Atalanta BC"),
    ("Sampdoria", "UC Sampdoria"),
    ("Club Brugge", "Club Brugge KV"),
    ("Udinese", "Udinese Calcio"),
    ("Genk", "KRC Genk"),
    ("Rostov", "FK Rostov"),
    ("Krasnodar", "FK Krasnodar"),
    ("Cagliari", "Cagliari Calcio"),
    ("Basaksehir", "Basaksehir FK"),
    ("Genclerbirligi", "Gençlerbirliği Spor Kulübü"),
    ("Sochi", "FK Sochi"),
    ("Lens", "RC Lens"),
    ("FC Oleksandriya", "FK Oleksandriya"),
    ("Ufa", "FK Ufa"),
    ("Santa Clara", "CD Santa Clara"),
    ("Chaves", "GD Chaves"),
    ("Nacional", "CD Nacional"),
    ("Frosinone", "Frosinone Calcio"),
    ("Benevento", "Benevento Calcio"),
    ("Tondela", "CD Tondela"),
    ("FC Minaj", "FK Minaj"),
    ("Feirense", "CD Feirense"),
    ("Brescia", "Brescia Calcio"),
    ("FC Mariupol", "FK Mariupol"),
    ("Nizhny Novgorod", "FK Nizhny Novgorod"),
    ("Osmanlispor", "Osmanlispor FK"),
    ("Qarabağ", "Qarabag FK"),
])
def test_club_name_aliases_merge_mechanical_suffix_variants(a, b):
    """
    Unlike the nickname pairs above, these merge purely from
    CLUB_NAME_STRIP_TOKENS (generic legal-entity markers like "CFC", "AS",
    "Calcio", "Spor Kulübü") stripping down to an identical remainder - no
    CLUB_NICKNAME_GROUPS entry needed. Still individually league/date
    verified during the same sweep to rule out same-city-different-club
    collisions (see the comment above CLUB_NICKNAME_GROUPS).
    """
    assert CLUB_NAME_ALIASES.get(a, a) == CLUB_NAME_ALIASES.get(b, b)


def test_club_name_aliases_keeps_distinct_sporting_clubs_separate():
    """'Sporting' alone is only merged into Sporting CP (verified via league PO1) - Sporting Gijón (Spain) and Royal Charleroi Sporting Club (Belgium) are different real clubs and must stay their own rows."""
    sporting_cp_canonical = CLUB_NAME_ALIASES.get("Sporting CP", "Sporting CP")
    assert CLUB_NAME_ALIASES.get("Sporting Gijón", "Sporting Gijón") != sporting_cp_canonical
    assert CLUB_NAME_ALIASES.get("Royal Charleroi Sporting Club", "Royal Charleroi Sporting Club") != sporting_cp_canonical


@pytest.mark.parametrize("a,b", [
    ("Arsenal", "Arsenal Tula"),
    ("Arsenal", "Arsenal Kyiv"),
    ("Barcelona", "RCD Espanyol Barcelona"),
    ("Rangers", "Queens Park Rangers"),
    ("Nacional", "Atl. Nacional"),
    ("Athletic", "Wigan Athletic"),
    ("Athletic", "Forfar Athletic"),
    ("Krasnodar", "Kuban Krasnodar"),
    ("Krasnodar", "Kuban Krasnodar (-2018)"),
])
def test_club_name_aliases_rejects_same_name_different_club_traps(a, b):
    """
    Each pair here shares a word (sometimes a whole name) but is a genuinely
    different real club - the same trap as Manchester City/United or Dundee
    FC/United, caught in the second sweep by checking domestic_competition_id
    (Arsenal Tula/Kyiv are a different league from Arsenal FC) or football
    domain knowledge (Rangers is Glasgow Rangers, not QPR; FC Krasnodar and
    the dissolved Kuban Krasnodar are two different Krasnodar clubs). Must
    stay separate rows.
    """
    assert CLUB_NAME_ALIASES.get(a, a) != CLUB_NAME_ALIASES.get(b, b)


@pytest.mark.parametrize("first_team,reserve_or_youth", [
    ("Benfica", "Benfica B"),
    ("Barcelona", "Barcelona B"),
    ("Tottenham", "Tottenham U21"),
    ("Villarreal", "FC Villarreal C"),
    ("Krasnodar", "Krasnodar 2"),
    ("Utrecht", "Utrecht U21"),
    ("FC Cartagena", "FC Cartagena B"),
])
def test_club_name_aliases_keeps_reserve_and_youth_sides_separate(first_team, reserve_or_youth):
    """A club's B/reserve/youth side runs its own transfer history, separate from the first team's - merging them would misattribute one squad's transfers to the other."""
    assert CLUB_NAME_ALIASES.get(first_team, first_team) != CLUB_NAME_ALIASES.get(reserve_or_youth, reserve_or_youth)


def test_club_name_aliases_metalist_cluster_merges_only_the_confirmed_pair():
    """
    Ukrainian "Metalist Kharkiv" went bankrupt around 2016, and two
    separately-run organizations have since both laid claim to the
    name/legacy - a real, contested identity dispute. Per explicit user
    confirmation, only "Metalist Kharkiv" and "Metalist Kharkiv (- 2016)"
    (the same name, just marking when that spelling stopped appearing) are
    merged; bare "Metalist" and "Metalist 1925" stay their own separate rows.
    """
    kharkiv_canonical = CLUB_NAME_ALIASES.get("Metalist Kharkiv (- 2016)", "Metalist Kharkiv (- 2016)")
    assert CLUB_NAME_ALIASES.get("Metalist Kharkiv", "Metalist Kharkiv") == kharkiv_canonical
    assert CLUB_NAME_ALIASES.get("Metalist", "Metalist") != kharkiv_canonical
    assert CLUB_NAME_ALIASES.get("Metalist 1925", "Metalist 1925") != kharkiv_canonical
    assert CLUB_NAME_ALIASES.get("Metalist", "Metalist") != CLUB_NAME_ALIASES.get("Metalist 1925", "Metalist 1925")


@pytest.mark.parametrize("a,b", [
    ("Apollon Smyrnis", "Apollon Smyrni"),
    ("Ionikos Nikeas", "Ionikos Nikea"),
    ("Nott'm Forest", "Nottm Forest"),
    ("FC Mariupol", "FSC Mariupol"),
    ("Dynamo Moscow", "Dinamo Moscow"),
    ("Beerschot VA", "Beerschot V.A."),
    ("FC Helsingör", "FC Helsingør"),
    ("Niki Volou", "Niki Volos"),
    ("PFC Lviv", "PFK Lviv"),
    ("Yeni Malatyaspor", "Y. Malatyaspor"),
    ("GFC Ajaccio", "G. Ajaccio"),
    ("Aris Saloniki", "Aris Thessalonikis"),
    ("Arm. Bielefeld", "Arminia Bielefeld"),
    ("Sint-Truiden", "Sint-Truidense VV"),
    ("A.G.S Asteras Tripolis", "Asteras Tripoli"),
    ("Aalesund", "Aalesunds FK"),
])
def test_club_name_aliases_merge_fuzzy_spelling_variants(a, b):
    """
    Found by a third pass matching on string similarity rather than shared
    whole words (the second sweep's method would miss an abbreviation/typo
    pair like "Man City"/"Manchester City" or "Y. Malatyaspor"/"Yeni
    Malatyaspor" if it hadn't already been caught another way) - same
    per-pair league/date verification as every other entry in
    CLUB_NICKNAME_GROUPS.
    """
    assert CLUB_NAME_ALIASES.get(a, a) == CLUB_NAME_ALIASES.get(b, b)


@pytest.mark.parametrize("a,b", [
    ("Atalanta", "Atlanta"),
    ("Metalurg D.", "Metalurg Z."),
    ("Al-Wehda", "Al-Wahda"),
    ("Beerschot AC", "Beerschot VA"),
])
def test_club_name_aliases_rejects_fuzzy_look_alike_traps(a, b):
    """
    More same-trap-as-Racing rejections from the string-similarity pass:
    "Atalanta"/"Atlanta" is Italy's Atalanta BC vs. MLS's Atlanta United
    (different leagues, confirmed); "Metalurg D."/"Metalurg Z." are two
    different Ukrainian clubs (Donetsk and Zaporizhzhia) that just happen to
    share a country with a single top flight, so the usual league-match
    check can't discriminate them; "Al-Wehda"/"Al-Wahda" is likely a Saudi
    club and a UAE club respectively; and "Beerschot AC" (bankrupt 2013) is
    left split from the later-reformed "Beerschot VA" per the same kind of
    user-confirmed call as the Metalist cluster.
    """
    assert CLUB_NAME_ALIASES.get(a, a) != CLUB_NAME_ALIASES.get(b, b)


def test_clubs_leaderboard_filters_endpoint():
    """/api/clubs/leaderboard/filters should list at least one league, each with an id and a display name."""
    res = client.get("/api/clubs/leaderboard/filters")
    assert res.status_code == 200
    data = res.json()
    assert len(data["leagues"]) > 0
    assert all("id" in l and "name" in l for l in data["leagues"])


def test_career_search_finds_players_missing_from_the_predict_autocomplete():
    """
    /api/players/career-search must cover players below players_lookup.csv's
    market-value threshold (see search_players_for_career) - a real gap
    that would otherwise hide most of a player's career history from this
    page specifically. "Mert Yılmaz" is a real, verified case: has scored
    transfer/loan history but is entirely missing from players_lookup.csv,
    so /api/players/search never finds him at all.
    """
    res = client.get("/api/players/career-search", params={"q": "Mert Yılmaz"})
    assert res.status_code == 200
    names = [p["name"] for p in res.json()]
    assert "Mert Yılmaz" in names

    narrow_res = client.get("/api/players/search", params={"q": "Mert Yılmaz"})
    assert not any(p["name"] == "Mert Yılmaz" for p in narrow_res.json())


def test_career_search_is_accent_insensitive():
    """The search box should match accented names via the plain-ASCII query too."""
    res = client.get("/api/players/career-search", params={"q": "Dembele"})
    names = [p["name"] for p in res.json()]
    assert any("Dembélé" in n for n in names)


def test_career_search_deduplicates_by_player():
    """A player with several transfers/loans must appear once in search results, not once per row."""
    res = client.get("/api/players/career-search", params={"q": "Joselu"})
    player_ids = [p["player_id"] for p in res.json()]
    assert len(player_ids) == len(set(player_ids))


def test_career_search_uses_the_precomputed_table():
    """search_players_for_career must filter the startup-precomputed PLAYER_CAREER_SEARCH_DF, not rebuild it per request - a stale/wrong table would still return results, just not the real ones."""
    from app.main import PLAYER_CAREER_SEARCH_DF
    assert len(PLAYER_CAREER_SEARCH_DF) > 0
    assert PLAYER_CAREER_SEARCH_DF["player_id"].duplicated().sum() == 0


def test_players_search_and_career_search_not_shadowed_by_player_id_route():
    """
    /api/players/{player_id} is registered after /api/players/search and
    /api/players/career-search specifically so it can't swallow requests
    meant for them - a bare {player_id} path segment matches any string,
    including literally "search" or "career-search", and only fails (422,
    not 200) once FastAPI tries to parse that string as the int player_id.
    Registered in the wrong order, both search endpoints would 422 on
    every request instead of ever running their own handler.
    """
    assert client.get("/api/players/search", params={"q": "Joselu"}).status_code == 200
    assert client.get("/api/players/career-search", params={"q": "Joselu"}).status_code == 200


def test_get_player_by_id_matches_the_search_result_exactly():
    """/api/players/{player_id} exists so Compare's shareable-link restore can reconstruct a scenario's player selection from just an id - it must return byte-for-byte the same record /api/players/search already does, or a restored comparison would silently differ from the one that was shared."""
    search_res = client.get("/api/players/search", params={"q": "Joselu"})
    player = search_res.json()[0]
    by_id_res = client.get(f"/api/players/{player['player_id']}")
    assert by_id_res.status_code == 200
    assert by_id_res.json() == player


def test_get_player_404_for_unknown_id():
    """A player_id with no players_lookup.csv row (e.g. below the market-value threshold that table is filtered to) should 404, not 500 or return an empty/garbage record."""
    res = client.get("/api/players/999999999")
    assert res.status_code == 404


@pytest.mark.parametrize("endpoint,extra_params", [
    ("/api/players/career-search", {}),
    ("/api/clubs/leaderboard", {}),
    ("/api/surprises", {}),
    ("/api/transfers", {}),
    ("/api/loans", {}),
    ("/api/players/search", {}),
    ("/api/clubs/search", {}),
])
def test_search_endpoints_do_not_500_on_regex_metacharacters(endpoint, extra_params):
    """A query box that treats the query as a plain substring (not a regex) must not 500 when the query happens to contain an unbalanced regex metacharacter - a real risk for a search-as-you-type box firing on every keystroke."""
    for bad_query in ["(", "[", "a{2,", "*abc", "a)b"]:
        res = client.get(endpoint, params={"q": bad_query, **extra_params})
        assert res.status_code == 200, f"{endpoint}?q={bad_query!r} returned {res.status_code}"


def test_player_career_combines_transfers_and_loans_chronologically():
    """A player with both permanent transfers and loans should get one merged, date-sorted timeline, each stop tagged with which table it came from."""
    res = client.get("/api/players/81999/career")  # Joselu
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "Joselu"
    stops = data["stops"]
    assert len(stops) >= 5
    assert {"permanent", "loan"} <= {s["type"] for s in stops}
    dates = [s["transfer_date"] for s in stops]
    assert dates == sorted(dates)
    for s in stops:
        assert 0 <= s["score"] <= 100


def test_player_career_404_for_unknown_player():
    """A player_id with no scored transfer or loan at all should 404, not return an empty timeline."""
    res = client.get("/api/players/999999999/career")
    assert res.status_code == 404


def test_player_career_degrades_gracefully_without_players_lookup_entry():
    """A player missing from players_lookup.csv entirely (see search_players_for_career) should still return a full timeline, just with position/current_club as null instead of a 500."""
    res = client.get("/api/players/393217/career")  # Mert Yılmaz - confirmed missing from players_lookup.csv
    assert res.status_code == 200
    data = res.json()
    assert len(data["stops"]) > 0
    assert data["position"] is None
    assert data["current_club"] is None


def test_player_career_includes_similar_careers_field():
    """/api/players/{id}/career should carry its own similar_careers list (see nearest_similar_careers) so the frontend needs no second request for the 'similar career shape' suggestions."""
    res = client.get("/api/players/81999/career")  # Joselu - 5+ stops
    data = res.json()
    assert "similar_careers" in data
    assert 0 < len(data["similar_careers"]) <= 5
    for entry in data["similar_careers"]:
        assert "player_id" in entry and "name" in entry
        assert entry["player_id"] != 81999


def test_similar_careers_empty_for_a_single_stop_career():
    """A career with only one stop has no real 'shape' to match against - similar_careers should be empty, not a handful of arbitrary players."""
    res = client.get("/api/players/418560/career")  # Erling Haaland - exactly one scored permanent transfer
    data = res.json()
    assert len(data["stops"]) == 1
    assert data["similar_careers"] == []


def test_similar_careers_matches_normalized_euclidean_nearest_neighbor():
    """nearest_similar_careers' own ranking should agree with an independent, direct recomputation of normalized Euclidean distance over PLAYER_SHAPE_NORMALIZED - not just internally consistent, but actually the nearest points."""
    from app.main import PLAYER_SHAPE_DF, PLAYER_SHAPE_NORMALIZED, nearest_similar_careers

    query_id = PLAYER_SHAPE_DF[PLAYER_SHAPE_DF["n_stops"] >= 2].index[0]

    query_vec = PLAYER_SHAPE_NORMALIZED.loc[query_id]
    diffs = PLAYER_SHAPE_NORMALIZED.drop(query_id) - query_vec
    dists = (diffs ** 2).sum(axis=1) ** 0.5
    expected_top5 = set(dists.sort_values().head(5).index)

    actual = {entry["player_id"] for entry in nearest_similar_careers(query_id)}
    assert actual == expected_top5


def test_league_trends_excludes_thin_leagues():
    """Every league in league_trends_df must have at least MIN_LEAGUE_TRANSFERS - a league with a handful of transfers shouldn't get a report card at all, let alone a noisy year-by-year trend."""
    assert len(league_trends_df) > 0
    assert (league_trends_df["transfers"] >= MIN_LEAGUE_TRANSFERS).all()


def test_league_trends_current_partial_year_excluded_from_by_year():
    """The dataset's own most recent (still in-progress) year must never appear in a league's by_year series - it runs far below a full season's transfer count and would show up as a misleading collapse."""
    current_year = pd.read_csv("data/transfers_processed.csv")["transfer_date"].str[:4].astype(int).max()
    for _, row in league_trends_df.iterrows():
        for point in row["by_year"]:
            assert point["year"] < current_year


def test_league_trends_early_and_recent_windows_are_real_and_disjoint():
    """A league with trend data must compare two genuinely different, non-overlapping multi-year windows, not the same years against themselves."""
    with_trend = league_trends_df[league_trends_df["early_years"].notna()]
    assert len(with_trend) > 0
    for _, row in with_trend.iterrows():
        early_start, early_end = (int(y) for y in row["early_years"].split("–"))
        recent_start, recent_end = (int(y) for y in row["recent_years"].split("–"))
        assert early_end < recent_start


def test_leagues_trends_endpoint_returns_every_qualifying_league():
    """/api/leagues/trends has few enough rows to return all of them unpaginated."""
    res = client.get("/api/leagues/trends")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == len(data["results"]) == len(league_trends_df)
    for r in data["results"]:
        assert r["transfers"] >= MIN_LEAGUE_TRANSFERS


def test_leagues_trends_sort_by_fee_growth_drops_leagues_with_no_trend():
    """Sorting by fee_growth_pct must never return a league whose fee_growth_pct is null (not enough history for the early/recent comparison)."""
    res = client.get("/api/leagues/trends", params={"sort": "fee_growth_pct", "order": "desc"})
    data = res.json()
    assert len(data["results"]) > 0
    values = [r["fee_growth_pct"] for r in data["results"]]
    assert all(v is not None for v in values)
    assert values == sorted(values, reverse=True)


def test_leagues_trends_endpoint_never_leaks_raw_nan(monkeypatch):
    """
    A league with no trend-eligible history has early_avg_fee/recent_avg_fee
    as None in league_trends_df (see build_league_trends) - once collected
    into a float64 DataFrame column alongside real leagues' real values,
    None becomes np.nan. The endpoint must serialize that as JSON null like
    its sibling fields already do, not leak the raw NaN token: Python's own
    json module silently accepts NaN as a non-standard extension, but a
    browser's strict JSON.parse does not, and would fail the whole page's
    fetch over one league's missing trend data.
    """
    import app.main as main
    synthetic_row = {
        "league_id": "ZZ9", "transfers": 20, "avg_score": 50.0, "avg_fee": 1_000_000.0,
        "early_years": None, "recent_years": None,
        "early_avg_fee": None, "recent_avg_fee": None, "fee_growth_pct": None,
        "early_avg_score": None, "recent_avg_score": None, "score_change": None,
        "by_year": [], "buys_from": [], "sells_to": [],
    }
    monkeypatch.setattr(main, "league_trends_df", pd.DataFrame([synthetic_row]))

    res = client.get("/api/leagues/trends")
    assert res.status_code == 200
    assert "NaN" not in res.text
    row = res.json()["results"][0]
    assert row["early_avg_fee"] is None
    assert row["recent_avg_fee"] is None


def test_leagues_trends_row_has_no_second_request_needed_fields():
    """Every row must already carry its own by_year series so the frontend's trend-chart modal needs no second request."""
    res = client.get("/api/leagues/trends")
    for r in res.json()["results"]:
        assert "by_year" in r
        if r["fee_growth_pct"] is not None:
            assert len(r["by_year"]) > 0
            assert all({"year", "avg_score", "avg_fee", "count"} <= set(point.keys()) for point in r["by_year"])


def test_league_trends_cross_league_flow_excludes_transfers_within_the_league():
    """buys_from/sells_to are cross-league flow - a league's own top entry must never be itself (an intra-league transfer isn't cross-league flow)."""
    res = client.get("/api/leagues/trends")
    for r in res.json()["results"]:
        for entry in r["buys_from"]:
            assert entry["league_id"] != r["league_id"]
        for entry in r["sells_to"]:
            assert entry["league_id"] != r["league_id"]


def test_league_trends_cross_league_flow_is_ranked_by_transfer_count_descending():
    res = client.get("/api/leagues/trends", params={"sort": "transfers", "order": "desc"})
    prem = next(r for r in res.json()["results"] if "Premier League" == r["league"])
    buys_counts = [e["transfers"] for e in prem["buys_from"]]
    sells_counts = [e["transfers"] for e in prem["sells_to"]]
    assert buys_counts == sorted(buys_counts, reverse=True)
    assert sells_counts == sorted(sells_counts, reverse=True)
    assert len(prem["buys_from"]) <= 8
    assert len(prem["sells_to"]) <= 8


def test_league_trends_cross_league_flow_counts_are_real_cross_checked_transfers():
    """The Premier League's top buys_from entry's transfer count should match a direct filter of the raw processed data - not just internally self-consistent, but actually correct."""
    df = pd.read_csv("data/transfers_processed.csv")
    res = client.get("/api/leagues/trends")
    prem = next(r for r in res.json()["results"] if r["league"] == "Premier League")
    top_source = prem["buys_from"][0]
    real_count = len(df[
        (df["to_domestic_competition_id"] == "GB1")
        & (df["from_domestic_competition_id"] == top_source["league_id"])
    ])
    assert top_source["transfers"] == real_count


def test_analytics_scatter_columns_are_all_the_same_length():
    """scatter is columnar (one array per field) - every array must have exactly one entry per transfer, in the same order, or the frontend would zip mismatched rows together."""
    res = client.get("/api/analytics")
    assert res.status_code == 200
    scatter = res.json()["scatter"]
    lengths = {len(v) for v in scatter.values()}
    assert len(lengths) == 1
    assert lengths.pop() == len(pd.read_csv("data/transfers_processed.csv"))


def test_analytics_scatter_player_id_and_date_resolve_via_transfers_detail():
    """scatter's player_id/transfer_date exist so a clicked chart point can open its full card - each pair must actually resolve through /api/transfers/detail, the same lookup every other list page's click-through uses."""
    res = client.get("/api/analytics")
    scatter = res.json()["scatter"]
    for i in (0, len(scatter["player_id"]) // 2, -1):
        detail_res = client.get("/api/transfers/detail", params={
            "player_id": scatter["player_id"][i], "transfer_date": scatter["transfer_date"][i],
        })
        assert detail_res.status_code == 200
        assert detail_res.json()["name"] == scatter["name"][i]


def test_analytics_scatter_nulls_are_json_null_not_nan():
    """transfer_fee/market_value_in_eur are missing for some transfers - the response must serialize those as JSON null, not leak a raw NaN token that a browser's strict JSON.parse would choke on."""
    res = client.get("/api/analytics")
    assert "NaN" not in res.text
    scatter = res.json()["scatter"]
    assert None in scatter["transfer_fee"]
    assert None in scatter["market_value_in_eur"]
    assert None not in scatter["age_at_transfer"]
    assert None not in scatter["success_score"]


def test_numeric_column_never_rounds_a_real_positive_value_down_to_zero():
    """
    transfer_fee/market_value_in_eur use 0 as a real "free transfer"/no-value
    sentinel, and the Analytics page's frontend filters scatter points on
    `> 0` to decide whether a transfer has a usable fee. Rounding to the
    nearest €1k (as /api/analytics does) must never turn a small real fee
    like €200 into exactly 0 - that would make it indistinguishable from a
    genuinely free transfer and silently drop it from the fee charts. A
    real free transfer (0) and a missing one (NaN) must still come through
    as 0 and None respectively - this isn't about pretending every value is
    positive, only about never manufacturing a false zero from rounding.
    """
    series = pd.Series([200.0, 0.0, None, 50_000.0])
    result = numeric_column(series, ndigits=-3)
    assert result[0] > 0, "a real €200 fee must not round down to 0"
    assert result[1] == 0
    assert result[2] is None
    assert result[3] == 50_000.0


def test_analytics_fee_trend_excludes_free_transfers_and_is_sorted():
    """fee_trend buckets transfer_fee on a log scale, so a free (0/null-fee) transfer must never pull a bucket down to x=0; buckets should come back sorted ascending for the trend line to draw correctly."""
    res = client.get("/api/analytics")
    fee_trend = res.json()["fee_trend"]
    assert len(fee_trend) > 0
    xs = [p["x"] for p in fee_trend]
    assert all(x > 0 for x in xs)
    assert xs == sorted(xs)


def test_analytics_trends_endpoint_matches_the_full_analytics_endpoint():
    """/api/analytics/trends exists purely to skip /api/analytics' much larger scatter payload - its fee_trend/age_trend must still be exactly the same series."""
    full = client.get("/api/analytics").json()
    trends = client.get("/api/analytics/trends").json()
    assert set(trends.keys()) == {"fee_trend", "age_trend"}
    assert trends["fee_trend"] == full["fee_trend"]
    assert trends["age_trend"] == full["age_trend"]


def test_analytics_age_trend_shows_younger_transfers_score_higher():
    """Sanity-checks the actual shape of the data, not just the plumbing: the youngest age bucket's average score should beat the oldest's, matching every other page's framing that younger transfers tend to pay off more."""
    res = client.get("/api/analytics")
    age_trend = res.json()["age_trend"]
    assert len(age_trend) >= 2
    assert age_trend[0]["avg_score"] > age_trend[-1]["avg_score"]


def test_analytics_by_year_excludes_current_partial_year():
    """Same reasoning as league_trends_df's by_year: the dataset's own most recent (still in-progress) year runs far below a full year's transfer count and must not appear, or it reads as a sudden market collapse."""
    current_year = pd.read_csv("data/transfers_processed.csv")["transfer_date"].str[:4].astype(int).max()
    res = client.get("/api/analytics")
    years = [r["year"] for r in res.json()["by_year"]]
    assert len(years) > 0
    assert current_year not in years
    assert years == sorted(years)
