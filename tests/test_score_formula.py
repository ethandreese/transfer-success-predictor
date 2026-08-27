"""
Tests for the success-score formula and the data it produces, using only
the small, committed CSVs (data/*.csv) - no dependency on the 730MB raw
Transfermarkt dataset, so these run anywhere without a kagglehub pull.
"""
import os

import pandas as pd
import pytest

from scripts.build_dataset import POSITION_WEIGHTS, RESALE_WEIGHT_CURVE, compute_resale_weight, percentile_rank

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
    without rerunning build_dataset.py) before it ships. Handles both cases:
    resale_profit included at its tenure-scaled weight (when has_resale_data)
    or dropped entirely and the rest renormalized (when it's unknown).
    """
    sample = transfers.sample(n=min(300, len(transfers)), random_state=42)
    for _, row in sample.iterrows():
        w = POSITION_WEIGHTS[row["position"]]
        base = (
            w["perf_level"] * row["perf_level_pct"]
            + w["perf_delta"] * row["perf_delta_pct"]
            + w["value_growth"] * row["value_growth_pct"]
            + w["playing_time"] * row["playing_time_pct"]
            + w["value_for_money"] * row["value_for_money_pct"]
        )
        other_weight_sum = 1 - w["resale_profit"]
        if row["has_resale_data"]:
            rescale = (1 - row["resale_weight"]) / other_weight_sum
            recomputed = base * rescale + row["resale_weight"] * row["resale_profit_pct"]
        else:
            recomputed = base / other_weight_sum
        assert recomputed == pytest.approx(row["success_score"], abs=0.15), (
            f"{row['name']} ({row['position']}): recomputed {recomputed:.2f} "
            f"!= stored {row['success_score']}"
        )


def test_resale_profit_only_counted_for_genuine_positive_fee_sales():
    """
    ~83% of transfers have no known resale (still at the club, or the
    dataset doesn't distinguish a loan-shaped fee=0 "next transfer" from a
    genuine free exit) - has_resale_data should be false, and
    resale_profit_pct null, for all of them.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    assert df["has_resale_data"].mean() == pytest.approx(0.169, abs=0.02)
    assert df.loc[~df["has_resale_data"], "resale_profit_pct"].isna().all()
    assert df.loc[df["has_resale_data"], "resale_profit_pct"].notna().all()
    assert df.loc[df["has_resale_data"], "next_transfer_fee"].gt(0).all()


def test_resale_profit_rewards_a_profitable_flip():
    """Moisés Caicedo joined Brighton for free and was later sold to Chelsea for €116m."""
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    row = df[(df["name"] == "Moisés Caicedo") & (df["to_club_name"] == "Brighton")]
    assert not row.empty
    r = row.iloc[0]
    assert r["has_resale_data"]
    assert r["next_transfer_fee"] > 100_000_000
    assert r["resale_profit_pct"] > 95


def test_pct_team_games_played_is_bounded(transfers):
    assert transfers["pct_team_games_played"].between(0, 1).all()
    assert transfers["team_games_in_tenure"].ge(1).all()


def test_pct_team_games_played_matches_apps_over_team_games(transfers):
    sample = transfers.sample(n=min(200, len(transfers)), random_state=11)
    expected = (sample["post_apps"] / sample["team_games_in_tenure"].clip(lower=1)).clip(upper=1.0)
    assert (sample["pct_team_games_played"] - expected).abs().max() < 1e-9


def test_playing_time_surfaces_injury_hit_tenures_raw_count_hides():
    """
    Dembélé made 185 appearances for Barcelona over 6 years - a big raw
    number - but that's only 57% of the games Barcelona actually played in
    that span (long-term injury problems). Playing time should reflect
    that poor availability, not just the large raw count.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    r = df[(df["name"] == "Ousmane Dembélé") & (df["to_club_name"] == "Barcelona")].iloc[0]
    assert r["post_apps"] > 150  # a large raw number on its own
    assert r["pct_team_games_played"] < 0.65  # but well under two-thirds of games available
    assert r["playing_time_pct"] < 60  # so playing_time should NOT read as elite


def test_value_growth_credits_peak_not_just_end_of_tenure_value():
    """
    Heung-min Son joined Tottenham valued at ~16-25m, peaked at ~90m
    mid-tenure, and was worth only ~20m a decade later when he finally
    left (natural age-related decline after a long, valuable career).
    Measuring value growth by the end-of-tenure snapshot alone would read
    this as a flat/unremarkable outcome; the peak value he reached is what
    actually reflects the asset the club held.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    son = df[(df["name"] == "Heung-min Son") & (df["to_club_name"] == "Tottenham")].iloc[0]
    assert son["value_peak"] >= son["value_before"] * 3  # a real, large peak appreciation
    assert son["value_after"] < son["value_before"] * 1.5  # end-of-tenure snapshot alone looks unremarkable
    assert son["value_growth_pct"] > 75  # but the score should credit the peak he reached


def test_value_peak_never_below_value_after(transfers):
    # value_peak is a max over the tenure window, so by construction it
    # can never be lower than the (also-in-window) end-of-tenure value.
    assert (transfers["value_peak"] >= transfers["value_after"]).all()


def test_resale_weight_decays_with_tenure_length():
    short = compute_resale_weight(0.25)   # ~3 months
    medium = compute_resale_weight(2.0)   # ~2 years
    long = compute_resale_weight(10.0)    # ~a decade
    assert short > medium > long
    assert short == pytest.approx(RESALE_WEIGHT_CURVE["max"], abs=0.03)
    assert long == pytest.approx(RESALE_WEIGHT_CURVE["min"], abs=0.01)


def test_resale_weight_column_matches_curve(transfers):
    resale = transfers[transfers["has_resale_data"]]
    expected = compute_resale_weight(resale["tenure_days"] / 365.25)
    assert (resale["resale_weight"] - expected).abs().max() < 1e-9


def test_long_tenure_resale_loss_barely_dents_the_score():
    """
    Heung-min Son: bought for ~30m, sold ~a decade later for ~22m (a loss
    on paper), but after a hugely valuable long career the resale outcome
    should barely move the needle - a low resale_weight, and a score that
    stays high despite the loss.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    son = df[(df["name"] == "Heung-min Son") & (df["to_club_name"] == "Tottenham")].iloc[0]
    assert son["next_transfer_fee"] < son["transfer_fee"]  # a genuine resale loss
    assert son["resale_weight"] < 0.03  # counts for almost nothing given the tenure length
    assert son["success_score"] > 70  # so the score stays high regardless
