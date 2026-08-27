"""
Tests for the success-score formula and the data it produces, using only
the small, committed CSVs (data/*.csv) - no dependency on the 730MB raw
Transfermarkt dataset, so these run anywhere without a kagglehub pull.
"""
import os

import pandas as pd
import pytest

from scripts.build_dataset import POSITION_WEIGHTS, percentile_rank

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

PCT_COLUMNS = [
    "perf_level_pct", "perf_delta_pct", "value_growth_pct",
    "playing_time_pct", "value_for_money_pct",
]


@pytest.fixture(scope="module")
def transfers():
    return pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))


def test_percentile_rank_is_0_to_100():
    s = pd.Series([10, 20, 30, 40, 50])
    ranked = percentile_rank(s)
    assert ranked.min() == pytest.approx(20.0)
    assert ranked.max() == pytest.approx(100.0)


def test_position_weights_sum_to_one():
    for position, weights in POSITION_WEIGHTS.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"{position} weights sum to {total}, not 1.0"


def test_goalkeepers_have_no_goal_contribution_weight():
    # The whole point of position-weighting: goal contributions are
    # meaningless for keepers (see data/score_weights.json), so their
    # weight must be zero, not just small.
    gk = POSITION_WEIGHTS["Goalkeeper"]
    assert gk["perf_level"] == 0
    assert gk["perf_delta"] == 0


def test_attackers_weight_performance_more_than_defenders():
    attack = POSITION_WEIGHTS["Attack"]
    defender = POSITION_WEIGHTS["Defender"]
    attack_perf = attack["perf_level"] + attack["perf_delta"]
    defender_perf = defender["perf_level"] + defender["perf_delta"]
    assert attack_perf > defender_perf


def test_success_score_within_bounds(transfers):
    assert transfers["success_score"].between(0, 100).all()


def test_league_baselines_reflect_real_scoring_difficulty(transfers):
    """
    Bundesliga attackers genuinely average more goal contributions/90 than
    Premier League attackers across the full appearances dataset (~0.54 vs
    ~0.48) - a Bundesliga -> Premier League move should be read as stepping
    into a harder-to-score-in league, not judged on raw output alone.
    """
    bundesliga = transfers[
        (transfers["to_domestic_competition_id"] == "L1") & (transfers["position"] == "Attack")
    ]["to_league_ga_baseline"]
    premier_league = transfers[
        (transfers["to_domestic_competition_id"] == "GB1") & (transfers["position"] == "Attack")
    ]["to_league_ga_baseline"]
    assert bundesliga.nunique() == 1
    assert premier_league.nunique() == 1
    assert bundesliga.iloc[0] > premier_league.iloc[0]


def test_league_adjusted_ratio_matches_raw_over_baseline(transfers):
    sample = transfers.sample(n=min(200, len(transfers)), random_state=7)
    expected = sample["post_ga_p90"] / sample["to_league_ga_baseline"].clip(lower=0.05)
    assert (sample["post_ga_p90_vs_league"] - expected).abs().max() < 1e-6


def test_performance_change_measures_vs_expectation_not_raw_delta():
    """
    A player who was extremely far above their league's average before the
    move (e.g. 2.6x) has more statistical room to regress toward the mean
    than someone who started closer to average - "performance change"
    should credit them for beating that regressed expectation, not punish
    them for a raw decline that's largely just regression to the mean.
    Concretely: Haaland's Dortmund -> Man City move (raw G+A/90 fell from
    1.39 to 1.07) should score well above the middle of the pack on
    "performance change", not below it.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    haaland = df[(df["name"] == "Erling Haaland") & (df["to_club_name"] == "Man City")].iloc[0]
    assert haaland["post_ga_p90_vs_league"] > haaland["pre_ga_p90_vs_league"] - 0.5  # real decline, but modest
    assert haaland["post_ga_p90_vs_league"] > haaland["expected_post_ga_p90_vs_league"]  # beat expectation
    assert haaland["perf_delta_pct"] > 75  # scores well, not punished for a raw decline


def test_genuine_collapse_still_scores_badly_on_performance_change():
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    sancho = df[(df["name"] == "Jadon Sancho") & (df["to_club_name"] == "Man Utd")].iloc[0]
    assert sancho["post_ga_p90_vs_league"] < sancho["expected_post_ga_p90_vs_league"]  # fell short too
    assert sancho["perf_delta_pct"] < 25


def test_percentile_components_within_bounds(transfers):
    for col in PCT_COLUMNS:
        assert transfers[col].between(0, 100).all(), f"{col} has values outside [0, 100]"


def test_no_nulls_in_key_columns(transfers):
    key_columns = [
        "name", "position", "transfer_date", "from_club_name", "to_club_name",
        "success_score", "tenure_days",
    ]
    for col in key_columns:
        assert transfers[col].notna().all(), f"{col} has null values"


def test_tenure_days_non_negative(transfers):
    assert (transfers["tenure_days"] >= 0).all()


def test_success_score_matches_weighted_components(transfers):
    """
    Recompute success_score from the stored per-component percentiles and
    each row's position weights, and check it matches the stored score.
    Catches a formula/weights drift (e.g. someone edits score_weights.json
    without rerunning build_dataset.py) before it ships.
    """
    sample = transfers.sample(n=min(300, len(transfers)), random_state=42)
    for _, row in sample.iterrows():
        w = POSITION_WEIGHTS[row["position"]]
        recomputed = (
            w["perf_level"] * row["perf_level_pct"]
            + w["perf_delta"] * row["perf_delta_pct"]
            + w["value_growth"] * row["value_growth_pct"]
            + w["playing_time"] * row["playing_time_pct"]
            + w["value_for_money"] * row["value_for_money_pct"]
        )
        assert recomputed == pytest.approx(row["success_score"], abs=0.15), (
            f"{row['name']} ({row['position']}): recomputed {recomputed:.2f} "
            f"!= stored {row['success_score']}"
        )
