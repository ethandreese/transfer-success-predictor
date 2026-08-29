"""
Tests for the success-score formula and the data it produces, using only
the small, committed CSVs (data/*.csv) - no dependency on the 730MB raw
Transfermarkt dataset, so these run anywhere without a kagglehub pull.
"""
import os

import pandas as pd
import pytest

from scripts.build_dataset import (
    LOAN_POSITION_WEIGHTS, LOAN_SUB_POSITION_WEIGHTS, MIN_LOAN_TENURE_DAYS,
    POSITION_WEIGHTS, RESALE_WEIGHT_CURVE, SUB_POSITION_WEIGHTS,
    compute_resale_weight, percentile_rank,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

PCT_COLUMNS = [
    "perf_level_pct", "perf_delta_pct", "value_growth_pct",
    "playing_time_pct", "value_for_money_pct",
]


@pytest.fixture(scope="module")
def transfers():
    """The committed processed dataset, loaded once and shared read-only across this module's tests."""
    return pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))


def test_percentile_rank_is_0_to_100():
    """percentile_rank() should map the lowest value to its rank/count (not 0) and the highest to exactly 100."""
    s = pd.Series([10, 20, 30, 40, 50])
    ranked = percentile_rank(s)
    assert ranked.min() == pytest.approx(20.0)
    assert ranked.max() == pytest.approx(100.0)


def test_position_weights_sum_to_one():
    """Every position's component weights (including resale_profit's reference weight) must sum to 1.0, or the score computation silently drifts off a 0-100 scale."""
    for position, weights in POSITION_WEIGHTS.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"{position} weights sum to {total}, not 1.0"


def test_goalkeepers_have_no_goal_contribution_weight():
    """
    The whole point of position-weighting: goal contributions are
    meaningless for keepers (see data/score_weights.json), so their
    weight must be zero, not just small.
    """
    gk = POSITION_WEIGHTS["Goalkeeper"]
    assert gk["perf_level"] == 0
    assert gk["perf_delta"] == 0


def test_attackers_weight_performance_more_than_defenders():
    """Sanity check on the position-weighting direction: goal contributions should matter more for attackers than defenders."""
    attack = POSITION_WEIGHTS["Attack"]
    defender = POSITION_WEIGHTS["Defender"]
    attack_perf = attack["perf_level"] + attack["perf_delta"]
    defender_perf = defender["perf_level"] + defender["perf_delta"]
    assert attack_perf > defender_perf


def test_sub_position_weights_sum_to_one():
    """Same invariant as the broad position rows - every sub_position override must still sum to 1.0."""
    for sub_position, weights in SUB_POSITION_WEIGHTS.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"{sub_position} weights sum to {total}, not 1.0"
    for sub_position, weights in LOAN_SUB_POSITION_WEIGHTS.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"loan {sub_position} weights sum to {total}, not 1.0"


def test_defensive_midfield_weights_attacking_output_less_than_central_midfield():
    """
    The motivating case for sub-position weighting (see README): a
    Defensive Midfielder shouldn't be judged on attacking output as
    heavily as a Central Midfielder, since goals/xG/xA/chance creation are
    much less central to their job. Central Midfield has no override (it
    IS the broad Midfield default), so this compares Defensive Midfield's
    override directly against POSITION_WEIGHTS["Midfield"].
    """
    dm = SUB_POSITION_WEIGHTS["Defensive Midfield"]
    cm = POSITION_WEIGHTS["Midfield"]
    dm_attacking_bloc = dm["perf_level"] + dm["perf_delta"] + dm["attacking"]
    cm_attacking_bloc = cm["perf_level"] + cm["perf_delta"] + cm["attacking"]
    assert dm_attacking_bloc < cm_attacking_bloc
    assert dm["defensive"] > cm["defensive"]


def test_centre_back_weights_attacking_output_less_than_full_backs():
    """Mirrors the midfield case: a Centre-Back's game has even less to do with attacking output than a full-back's, who's expected to contribute going forward."""
    cb = SUB_POSITION_WEIGHTS["Centre-Back"]
    rb = SUB_POSITION_WEIGHTS["Right-Back"]
    cb_attacking_bloc = cb["perf_level"] + cb["perf_delta"] + cb["attacking"]
    rb_attacking_bloc = rb["perf_level"] + rb["perf_delta"] + rb["attacking"]
    assert cb_attacking_bloc < rb_attacking_bloc
    assert cb["defensive"] > rb["defensive"]


def test_sub_position_lookup_falls_back_to_broad_position(transfers):
    """
    A sub_position with no override (Central Midfield, Centre-Forward,
    Second Striker, Goalkeeper, or missing/NaN) must use exactly its broad
    position's weights - the whole point of lookup_weights falling back
    rather than guessing at an unlisted sub-position's profile.
    """
    unoverridden = transfers[~transfers["sub_position"].isin(SUB_POSITION_WEIGHTS.keys())]
    assert len(unoverridden) > 0
    for position in unoverridden["position"].unique():
        assert position in POSITION_WEIGHTS


def test_success_score_within_bounds(transfers):
    """success_score is meant to be a 0-100 scale end to end - no row should fall outside it."""
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
    """post_ga_p90_vs_league should always equal post_ga_p90 divided by the stored to_league_ga_baseline (within the clip floor) - catches the formula and the stored column drifting apart."""
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
    """The regression-to-expectation fix shouldn't rescue a real bust: Sancho's Man Utd collapse must still fall well short of even the regressed expectation, and score low."""
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    sancho = df[(df["name"] == "Jadon Sancho") & (df["to_club_name"] == "Man Utd")].iloc[0]
    assert sancho["post_ga_p90_vs_league"] < sancho["expected_post_ga_p90_vs_league"]  # fell short too
    assert sancho["perf_delta_pct"] < 25


def test_percentile_components_within_bounds(transfers):
    """Every sub-score percentile that feeds into success_score should itself be a valid 0-100 percentile."""
    for col in PCT_COLUMNS:
        assert transfers[col].between(0, 100).all(), f"{col} has values outside [0, 100]"


def test_no_nulls_in_key_columns(transfers):
    """These identity/label columns should never be null in the shipped dataset - a null here would break the UI or a downstream join."""
    key_columns = [
        "name", "position", "transfer_date", "from_club_name", "to_club_name",
        "success_score", "tenure_days",
    ]
    for col in key_columns:
        assert transfers[col].notna().all(), f"{col} has null values"


def test_tenure_days_non_negative(transfers):
    """tenure_end is always >= transfer_date by construction; this would only fail if that invariant broke upstream."""
    assert (transfers["tenure_days"] >= 0).all()


FOTMOB_COMPONENTS = ["rating", "attacking", "defensive", "possession"]


def test_success_score_matches_weighted_components(transfers):
    """
    Recompute success_score from the stored per-component percentiles and
    each row's position weights, and check it matches the stored score.
    Catches a formula/weights drift (e.g. someone edits score_weights.json
    without rerunning build_dataset.py) before it ships. Handles every
    combination of the components that can independently be missing: each
    of the 4 FotMob components (dropped/renormalized first, per its own
    has_*_data flag) and resale_profit (layered on top at its tenure-scaled
    weight, or dropped/renormalized, when has_resale_data is false) - see
    build_dataset.py's main() for why the FotMob components have to be
    folded in before the resale-profit renormalization runs. perf_level and
    attacking are folded together first (see fold_perf_level_into_attacking)
    since they both measure attacking output. Weights come from
    SUB_POSITION_WEIGHTS when the row's actual sub_position has a distinct
    profile there, else fall back to the broad POSITION_WEIGHTS row (see
    lookup_weights) - percentile ranking is unaffected either way, it's
    still grouped by the broad position alone.
    """
    sample = transfers.sample(n=min(300, len(transfers)), random_state=42)
    for _, row in sample.iterrows():
        w = SUB_POSITION_WEIGHTS.get(row["sub_position"], POSITION_WEIGHTS[row["position"]])
        other_weight_sum = 1 - w["resale_profit"]

        has_attacking = bool(row["has_attacking_data"])
        perf_level_weight = 0 if has_attacking else w["perf_level"]
        attacking_weight = (w["attacking"] + w["perf_level"]) if has_attacking else w["attacking"]
        # attacking_pct is already stored post-fold (the average of
        # perf_level_pct and the raw FotMob attacking percentile) whenever
        # has_attacking_data is true - see fold_perf_level_into_attacking.
        component_weight = {"rating": w["rating"], "attacking": attacking_weight, "defensive": w["defensive"], "possession": w["possession"]}

        known_score = (
            perf_level_weight * row["perf_level_pct"]
            + w["perf_delta"] * row["perf_delta_pct"]
            + w["value_growth"] * row["value_growth_pct"]
            + w["playing_time"] * row["playing_time_pct"]
            + w["value_for_money"] * row["value_for_money_pct"]
        )
        known_weight = perf_level_weight + w["perf_delta"] + w["value_growth"] + w["playing_time"] + w["value_for_money"]
        for c in FOTMOB_COMPONENTS:
            if row[f"has_{c}_data"]:
                known_score += component_weight[c] * row[f"{c}_pct"]
                known_weight += component_weight[c]

        base = known_score / known_weight * other_weight_sum

        if row["has_resale_data"]:
            rescale = (1 - row["resale_weight"]) / other_weight_sum
            recomputed = base * rescale + row["resale_weight"] * row["resale_profit_pct"]
        else:
            recomputed = base / other_weight_sum
        assert recomputed == pytest.approx(row["success_score"], abs=0.15), (
            f"{row['name']} ({row['position']}): recomputed {recomputed:.2f} "
            f"!= stored {row['success_score']}"
        )


@pytest.mark.parametrize("component", FOTMOB_COMPONENTS)
def test_fotmob_component_only_counted_when_its_data_known(transfers, component):
    """has_{component}_data should exactly gate whether {component}_pct is present - never null-but-true or non-null-but-false, or its weight would multiply into a NaN or silently vanish."""
    flag = transfers[f"has_{component}_data"]
    assert transfers.loc[flag, f"{component}_pct"].notna().all()
    assert transfers.loc[~flag, f"{component}_pct"].isna().all()


@pytest.mark.parametrize("component", FOTMOB_COMPONENTS)
def test_fotmob_component_pct_within_bounds(transfers, component):
    """Like every other sub-score, each FotMob component's _pct should itself be a valid 0-100 percentile wherever it's known."""
    known = transfers.loc[transfers[f"has_{component}_data"], f"{component}_pct"]
    assert known.between(0, 100).all()


def test_goalkeepers_have_no_attacking_weight():
    """Goals/xG/xA/dribbles are as meaningless for a goalkeeper's *attacking* FotMob bucket as perf_level/perf_delta already are - the weight must be zero, not just small."""
    assert POSITION_WEIGHTS["Goalkeeper"]["attacking"] == 0


def test_defenders_weight_defensive_more_than_attackers():
    """Sanity check on the position-weighting direction: the defensive FotMob bucket should matter more for defenders than attackers, mirroring the existing perf_level check in reverse."""
    assert POSITION_WEIGHTS["Defender"]["defensive"] > POSITION_WEIGHTS["Attack"]["defensive"]


def test_fotmob_data_covers_most_transfers_but_not_all():
    """
    FotMob coverage is real but not universal (nothing before a league-
    specific season, and Ukraine has no usable stats at all - see
    scripts/fetch_fotmob_stats.py) - has_fotmob_data (true if ANY of the 4
    components has data) should cover most transfers, but a meaningful
    minority shouldn't, or the renormalization path would never actually
    be exercised.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    assert 0.6 < df["has_fotmob_data"].mean() < 0.95


def test_resale_profit_only_counted_for_genuine_positive_fee_sales():
    """
    ~31% of transfers have a known resale (a genuine subsequent sale for a
    recorded fee) - the rest (still at the club, exited for free, or the
    next move's fee just isn't recorded) should have has_resale_data false
    and resale_profit_pct null. This rate rose from ~17% once loans were
    excluded from the transfer chain (see load_transfers/load_loan_spells)
    - previously, a loan-out sitting between a permanent signing and its
    eventual resale would make next_transfer_fee land on the loan's
    (unrecorded) fee instead of skipping through to the real sale.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    assert df["has_resale_data"].mean() == pytest.approx(0.308, abs=0.02)
    assert df.loc[~df["has_resale_data"], "resale_profit_pct"].isna().all()
    assert df.loc[df["has_resale_data"], "resale_profit_pct"].notna().all()
    assert df.loc[df["has_resale_data"], "next_transfer_fee"].gt(0).all()


def test_resale_profit_rewards_a_profitable_flip():
    """Randal Kolo Muani joined Frankfurt from Nantes for free and was sold on to PSG for €95m about 14 months later."""
    df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
    row = df[(df["name"] == "Randal Kolo Muani") & (df["to_club_name"] == "Frankfurt")]
    assert not row.empty
    r = row.iloc[0]
    assert r["has_resale_data"]
    assert r["transfer_fee"] == 0
    assert r["next_transfer_fee"] > 90_000_000
    assert r["resale_profit_pct"] > 95


def test_pct_team_games_played_is_bounded(transfers):
    """A player can't appear in more games than the team played, so pct_team_games_played must be a clean fraction in [0, 1]."""
    assert transfers["pct_team_games_played"].between(0, 1).all()
    assert transfers["team_games_in_tenure"].ge(1).all()


def test_pct_team_games_played_matches_apps_over_team_games(transfers):
    """pct_team_games_played should always equal post_apps / team_games_in_tenure (clipped) - catches drift between the formula and the stored column."""
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
    """
    value_peak is a max over the tenure window, so by construction it
    can never be lower than the (also-in-window) end-of-tenure value -
    guards against the window-mismatch bug this was built to fix (see
    build_dataset.py's value_peak computation).
    """
    assert (transfers["value_peak"] >= transfers["value_after"]).all()


def test_resale_weight_decays_with_tenure_length():
    """compute_resale_weight should decrease monotonically with tenure, bounded by the curve's configured min/max."""
    short = compute_resale_weight(0.25)   # ~3 months
    medium = compute_resale_weight(2.0)   # ~2 years
    long = compute_resale_weight(10.0)    # ~a decade
    assert short > medium > long
    assert short == pytest.approx(RESALE_WEIGHT_CURVE["max"], abs=0.03)
    assert long == pytest.approx(RESALE_WEIGHT_CURVE["min"], abs=0.01)


def test_resale_weight_column_matches_curve(transfers):
    """The stored resale_weight column should exactly match compute_resale_weight() applied to each row's own tenure_days."""
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


def test_league_baselines_csv_has_real_and_fallback_rows():
    """
    league_baselines.csv is a shared artifact: build_dataset.py uses it to
    build the success-score label, and app/main.py separately loads it to
    add league context to live prediction explanations. Both must be able
    to find a value for any (league, position) via a "_default" fallback.
    """
    baselines = pd.read_csv(os.path.join(DATA_DIR, "league_baselines.csv"))
    assert set(baselines.columns) == {"competition_id", "position", "ga_p90_baseline"}
    assert (baselines["competition_id"] == "_default").sum() >= 4  # one per position
    real = baselines[baselines["competition_id"] != "_default"]
    assert real["competition_id"].nunique() > 20  # broad league coverage
    assert (baselines["ga_p90_baseline"] >= 0).all()

    bundesliga = baselines[(baselines["competition_id"] == "L1") & (baselines["position"] == "Attack")]
    premier_league = baselines[(baselines["competition_id"] == "GB1") & (baselines["position"] == "Attack")]
    assert bundesliga["ga_p90_baseline"].iloc[0] > premier_league["ga_p90_baseline"].iloc[0]


@pytest.fixture(scope="module")
def loans():
    """The committed processed loans dataset, loaded once and shared read-only across this module's tests."""
    return pd.read_csv(os.path.join(DATA_DIR, "loans_processed.csv"))


def test_loan_position_weights_sum_to_one():
    """Every position's loan-score weights (data/loan_score_weights.json) must sum to 1.0, or the blended score wouldn't be on a 0-100 scale."""
    for position, w in LOAN_POSITION_WEIGHTS.items():
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6), position


def test_loan_position_weights_have_no_fee_based_components():
    """Loan weights must not include value_for_money or resale_profit - neither maps to a loan spell (see data/loan_score_weights.json)."""
    for w in LOAN_POSITION_WEIGHTS.values():
        assert "value_for_money" not in w
        assert "resale_profit" not in w


def test_loan_goalkeepers_have_no_goal_contribution_weight():
    """Same rationale as the permanent-transfer weights: goal contributions are meaningless for goalkeepers."""
    gk = LOAN_POSITION_WEIGHTS["Goalkeeper"]
    assert gk["perf_level"] == 0
    assert gk["perf_delta"] == 0


def test_loan_success_score_within_bounds(loans):
    """loan_success_score is a blend of 0-100 percentiles, so it must itself land in [0, 100]."""
    assert loans["loan_success_score"].between(0, 100).all()


def test_loan_success_score_matches_weighted_components(loans):
    """
    Recompute loan_success_score from the stored per-component percentiles
    and each row's position weights, and check it matches the stored score -
    catches a formula/weights drift before it ships. Unlike the permanent
    transfer score, there's no resale/value-for-money renormalization
    layer since loans never have those components in the first place - just
    the single-stage renormalization for the 4 FotMob components, each
    independently droppable per its own has_*_data flag. perf_level and
    attacking are folded together first, same as the permanent-transfer
    formula - see fold_perf_level_into_attacking. Weights come from
    LOAN_SUB_POSITION_WEIGHTS when the row's actual sub_position has a
    distinct profile there, else fall back to the broad
    LOAN_POSITION_WEIGHTS row, same pattern as the permanent-transfer test.
    """
    sample = loans.sample(n=min(200, len(loans)), random_state=42)
    for _, row in sample.iterrows():
        w = LOAN_SUB_POSITION_WEIGHTS.get(row["sub_position"], LOAN_POSITION_WEIGHTS[row["position"]])

        has_attacking = bool(row["has_attacking_data"])
        perf_level_weight = 0 if has_attacking else w["perf_level"]
        attacking_weight = (w["attacking"] + w["perf_level"]) if has_attacking else w["attacking"]
        component_weight = {"rating": w["rating"], "attacking": attacking_weight, "defensive": w["defensive"], "possession": w["possession"]}

        known_score = (
            perf_level_weight * row["perf_level_pct"]
            + w["perf_delta"] * row["perf_delta_pct"]
            + w["value_growth"] * row["value_growth_pct"]
            + w["playing_time"] * row["playing_time_pct"]
        )
        known_weight = perf_level_weight + w["perf_delta"] + w["value_growth"] + w["playing_time"]
        for c in FOTMOB_COMPONENTS:
            if row[f"has_{c}_data"]:
                known_score += component_weight[c] * row[f"{c}_pct"]
                known_weight += component_weight[c]

        recomputed = known_score / known_weight
        assert recomputed == pytest.approx(row["loan_success_score"], abs=0.15), (
            f"{row['name']} ({row['position']}): recomputed {recomputed:.2f} != stored {row['loan_success_score']}"
        )


@pytest.mark.parametrize("component", FOTMOB_COMPONENTS)
def test_loan_fotmob_component_only_counted_when_its_data_known(loans, component):
    """Same gating check as the permanent-transfer components - has_{component}_data should exactly match {component}_pct's nullness."""
    flag = loans[f"has_{component}_data"]
    assert loans.loc[flag, f"{component}_pct"].notna().all()
    assert loans.loc[~flag, f"{component}_pct"].isna().all()


def test_loan_tenure_at_least_minimum_days(loans):
    """Every loan spell should be at least MIN_LOAN_TENURE_DAYS long - shorter spells are dropped as likely data artifacts (see load_loan_spells)."""
    assert (loans["tenure_days"] >= MIN_LOAN_TENURE_DAYS).all()


def test_loan_pre_apps_meets_minimum_but_zero_post_apps_allowed(loans):
    """
    Unlike permanent transfers (which require >= MIN_APPS_PER_WINDOW
    appearances on *both* sides), a loan spell only needs the pre-loan bar
    met - a loan where the player barely or never played is a real outcome
    the Loans tab exists to surface, not missing data (see
    prepare_loans's docstring). At least one such spell should exist
    in the real dataset, or the distinction isn't actually doing anything.
    """
    assert (loans["post_apps"] == 0).any()


def test_loan_no_nulls_in_key_columns(loans):
    """The score and its four input percentiles should never be null for a row that made it into loans_processed.csv."""
    key_cols = ["perf_level_pct", "perf_delta_pct", "value_growth_pct", "playing_time_pct", "loan_success_score"]
    assert not loans[key_cols].isna().any().any()
