"""
Train a model that predicts transfer success_score from pre-transfer-only
features (data/transfers_processed.csv), using a temporal train/test split
so evaluation reflects predicting *future* transfers from past ones.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "model")
SPLIT_DATE = "2023-06-01"

# How much more a recent transfer counts than an old one when fitting the
# model - football's finances and tactics genuinely shift over a decade
# (fee inflation, FFP-style rules, more data-driven recruitment), and the
# model's actual job is predicting *future* transfers, so a training row
# from 2023 plausibly generalizes to that better than one from 2013. Ridge
# fits every row unweighted otherwise. Checked directly across the same 5
# temporal splits used for the Ridge-vs-GBR comparison above: exponential
# recency weighting with a 3-year half-life improved MAE by ~0.1 and R^2 by
# ~0.005-0.011 on 4 of 5 splits, and was never worse (a ~0.001 R^2 wash on
# the 5th, the most recent/smallest-test-set split) - a small but
# consistent, low-risk win, not a knife-edge tuning result. Half-lives from
# 1-8 years were tried; 3 and 5 were both consistently good, 1 was clearly
# too aggressive (discards too much of the training set's effective size).
RECENCY_HALF_LIFE_YEARS = 3.0


def compute_recency_weight(transfer_date, reference_date, half_life_years=RECENCY_HALF_LIFE_YEARS):
    """
    Exponential decay sample weight: 1.0 for a transfer on `reference_date`
    itself, halving every `half_life_years` years further back. `reference_
    date` is the fitting population's own most recent transfer_date (not
    today's date), so the weighting is relative to "how old is this row
    compared to the newest thing the model is being fit on" - the same
    fit-on-train discipline every other fitted reference in this file
    follows (e.g. position_height_means).
    """
    age_years = (reference_date - transfer_date).dt.days / 365.25
    return 0.5 ** (age_years / half_life_years)

# Raw pre-transfer FotMob per-90 stats (see
# scripts/fetch_pretransfer_fotmob_stats.py) - the same rating/attacking/
# possession/defensive signal the historical score's post-transfer
# components already use, now also available as a pre-transfer input.
# Tested empirically (see git history): every bucket individually beat
# the without-FotMob baseline, and using all of them together was the
# single biggest accuracy improvement found this project - MAE 12.93 ->
# 12.34, R^2 0.114 -> 0.175 on the same temporal holdout, stable across
# 10 random seeds (R^2 0.161-0.175). pre_fotmob_total_att_assist (a
# season-total count on FotMob's own leaderboard, not a rate) isn't used
# directly - pre_fotmob_chances_created_p90 (computed in build_dataset.py,
# same normalization compute_fotmob_component_pcts uses for the
# post-transfer side) is used instead.
PRETRANSFER_FOTMOB_FEATURES = [
    "pre_fotmob_rating",
    "pre_fotmob_expected_goals_per_90", "pre_fotmob_expected_assists_per_90", "pre_fotmob_chances_created_p90",
    "pre_fotmob_accurate_pass", "pre_fotmob_won_contest",
    "pre_fotmob_total_tackle", "pre_fotmob_interception", "pre_fotmob_effective_clearance", "pre_fotmob_ball_recovery",
    "pre_fotmob_saves", "pre_fotmob__save_percentage", "pre_fotmob_goals_conceded",
]
# These 12 raw stats are the fetched/autofilled *inputs* (see
# scripts/fetch_pretransfer_fotmob_stats.py, app/main.py:PredictRequest) -
# not what the model trains on directly. See PRETRANSFER_FOTMOB_COMPOSITES
# below for why.
PRETRANSFER_RATING_STAT = "pre_fotmob_rating"
PRETRANSFER_ATTACKING_STATS = ["pre_fotmob_expected_goals_per_90", "pre_fotmob_expected_assists_per_90", "pre_fotmob_chances_created_p90"]
PRETRANSFER_DEFENSIVE_OUTFIELD_STATS = ["pre_fotmob_total_tackle", "pre_fotmob_interception", "pre_fotmob_effective_clearance", "pre_fotmob_ball_recovery"]
PRETRANSFER_DEFENSIVE_GK_STATS = ["pre_fotmob_saves", "pre_fotmob__save_percentage", "pre_fotmob_goals_conceded"]
PRETRANSFER_POSSESSION_STATS = ["pre_fotmob_accurate_pass", "pre_fotmob_won_contest"]
# Fewer conceded is better - percentile-ranked on the negated value so a
# higher percentile always means "better", same as fotmob_goals_conceded_inv
# in build_dataset.py.
PRETRANSFER_INVERTED_STATS = {"pre_fotmob_goals_conceded"}

# The model trains on these 4 position-relative percentiles (mirroring
# compute_fotmob_component_pcts in build_dataset.py - the same rating/
# attacking/defensive/possession split the historical score's own
# components use), not the 12 raw per-90 stats above. Checked empirically
# across 5 random seeds: replacing the raw stats with these beat feeding
# them in raw on every seed (MAE 12.68 -> 12.63, R^2 0.152 -> 0.161), and
# fixed a real problem the raw version had, not just a style preference -
# a raw stat's absolute scale means something completely different by
# position (1.5 tackles/90 is unremarkable for a striker, below average
# for a centre-back), so the model had no reliable way to learn that
# defensive stats should matter more for a defender than an attacker with
# a training set this small. Confirmed directly: swinging
# pre_fotmob_total_tackle from a low to a high value for a synthetic row
# of each position moved the raw-feature model's prediction by nearly the
# same amount regardless of position (+1.6 for an attacker, +2.0 for a
# defender), and for pre_fotmob_expected_goals_per_90 the *smallest* swing
# was for attackers - backwards from what the feature is supposed to
# capture. A percentile-within-position feature doesn't have this problem
# by construction: the number itself already means "how good is this for
# their position", so the model doesn't have to somehow relearn that
# per position from a training set this size. Adding the composites
# *alongside* the raw stats (rather than replacing them) tested no better
# than the raw-only baseline - redundant, correlated inputs don't help a
# tree ensemble, they just add noise to split on.
PRETRANSFER_FOTMOB_COMPOSITES = ["pre_fotmob_rating_pct", "pre_fotmob_attacking_pct", "pre_fotmob_defensive_pct", "pre_fotmob_possession_pct"]
# Per-bucket, not one blanket flag - a player can have real attacking data
# and no defensive data, and the model should be able to discount each
# composite independently rather than treating "has any FotMob data at
# all" as one signal.
PRETRANSFER_FOTMOB_HAS_DATA_FLAGS = ["has_pre_rating_data", "has_pre_attacking_data", "has_pre_defensive_data", "has_pre_possession_data"]
PRETRANSFER_PERCENTILE_POINTS = list(range(101))


def build_pretransfer_percentile_tables(df):
    """
    Per (raw stat, position), percentile breakpoints (0th..100th, at every
    integer point) fit from real - non-imputed, this must be called before
    any imputation - values only, for interpolation-based lookup later (see
    pretransfer_percentile). This is compute_fotmob_component_pcts's
    within-position percentile rank in build_dataset.py, but as a fitted,
    persistable transform rather than a one-off batch rank: a live
    prediction needs "what percentile would THIS value fall at, relative to
    the training distribution", which a batch-only rank can't answer for a
    value that wasn't already in the batch. `df` should be the train split
    for the holdout eval, the full dataset for the deployed model - same
    fit-on-train/apply-to-test discipline as every other fitted reference
    here. A position with under 20 real values for a stat (e.g. saves for
    an outfield position) gets no table entry - lookup then treats the
    value as missing, same as zero coverage; 20 is an arbitrary but
    reasonable floor for a percentile curve to mean anything.
    """
    tables = {}
    for stat in PRETRANSFER_FOTMOB_FEATURES:
        sign = -1 if stat in PRETRANSFER_INVERTED_STATS else 1
        tables[stat] = {}
        for position, sub in df.groupby("position"):
            values = sub[stat].dropna()
            if len(values) < 20:
                continue
            tables[stat][position] = (sign * values).quantile([p / 100 for p in PRETRANSFER_PERCENTILE_POINTS]).tolist()
    return tables


def compute_pretransfer_fotmob_composites(df, tables):
    """
    Turn the 12 raw pre_fotmob_* stats into the 4 position-relative
    composites the model actually trains on (PRETRANSFER_FOTMOB_COMPOSITES) -
    same bucket definitions, and the same "average whichever sub-stats are
    actually available, NaN only if none are" tolerance, as
    compute_fotmob_component_pcts - applied to a fitted percentile lookup
    (`tables`, see build_pretransfer_percentile_tables) instead of a batch
    rank. "Defensive" uses a genuinely different stat set by position
    (saves/save%/goals-conceded for goalkeepers, tackles/interceptions/
    clearances/recoveries for everyone else), same reasoning as the
    historical score's own defensive bucket.
    """
    df = df.copy()
    pct = {}
    for stat in PRETRANSFER_FOTMOB_FEATURES:
        sign = -1 if stat in PRETRANSFER_INVERTED_STATS else 1
        col = pd.Series(np.nan, index=df.index)
        for position, sub in df.groupby("position"):
            breakpoints = tables.get(stat, {}).get(position)
            if breakpoints is None:
                continue
            looked_up = pd.Series(np.interp(sign * sub[stat], breakpoints, PRETRANSFER_PERCENTILE_POINTS), index=sub.index)
            looked_up[sub[stat].isna()] = np.nan  # np.interp has no real notion of a missing input - mask it back in explicitly
            col.loc[sub.index] = looked_up
        pct[stat] = col

    def avg_pct(stats):
        return pd.concat([pct[s] for s in stats], axis=1).mean(axis=1, skipna=True)

    df["pre_fotmob_rating_pct"] = pct[PRETRANSFER_RATING_STAT]
    df["pre_fotmob_attacking_pct"] = avg_pct(PRETRANSFER_ATTACKING_STATS)
    df["pre_fotmob_possession_pct"] = avg_pct(PRETRANSFER_POSSESSION_STATS)

    is_gk = df["position"] == "Goalkeeper"
    defensive = pd.Series(np.nan, index=df.index)
    defensive.loc[~is_gk] = avg_pct(PRETRANSFER_DEFENSIVE_OUTFIELD_STATS).loc[~is_gk]
    defensive.loc[is_gk] = avg_pct(PRETRANSFER_DEFENSIVE_GK_STATS).loc[is_gk]
    df["pre_fotmob_defensive_pct"] = defensive

    for comp, flag in zip(["rating", "attacking", "defensive", "possession"], PRETRANSFER_FOTMOB_HAS_DATA_FLAGS):
        df[flag] = df[f"pre_fotmob_{comp}_pct"].notna().astype(int)
    return df


def impute_pretransfer_fotmob_composites(df, medians):
    """~35-45% of transfers have no pre-transfer FotMob match for a given bucket - filled with that bucket's population median (`medians`, fit on whichever population df was drawn from) rather than dropped, alongside the has_pre_*_data flags so the model can learn to discount an imputed placeholder."""
    df = df.copy()
    for feat in PRETRANSFER_FOTMOB_COMPOSITES:
        df[feat] = df[feat].fillna(medians[feat])
    return df

NUMERIC_FEATURES = [
    "age_at_transfer", "height_vs_position",
    "pre_apps", "pre_minutes", "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app",
    "log_transfer_fee", "log_value_before", "fee_to_value_ratio", "club_quality_ratio",
    "log_from_club_value", "log_to_club_value",
] + PRETRANSFER_FOTMOB_COMPOSITES + PRETRANSFER_FOTMOB_HAS_DATA_FLAGS
# Tried adding pre_ga_p90_vs_league / log_from_league_baseline /
# log_to_league_baseline (the same league-adjustment used in the success
# score label) as model inputs too - tested empirically against several
# combinations and model configs (see git history / README), and none
# beat this feature set's R^2 on the temporal holdout. The destination
# league category already captures most of that signal for leagues with
# enough training data, and the ~4,400-row training set is too small to
# reliably learn the added continuous relationships on top of that. Kept
# the simpler, empirically-better feature set rather than adding
# complexity that doesn't pay off; league_baselines.csv is still used to
# add league context to the live prediction explanation (app/main.py),
# just not as a trained feature.
CATEGORICAL_FEATURES = [
    "position", "sub_position", "foot", "from_domestic_competition_id", "to_domestic_competition_id",
]
# sub_position (e.g. Centre-Back vs. Winger, not just the 4 broad positions -
# see load_actual_sub_positions in build_dataset.py) tested as a small but
# real accuracy improvement alongside the broad position category, not a
# replacement for it - kept both rather than picking one (see git history
# for the comparison).
TARGET = "success_score"

# Features whose "typical" value varies a lot by position (a striker's
# goal contributions, a goalkeeper's minutes/appearances) - comparing them
# against the whole population regardless of position produces misleading
# prediction explanations (see reference_values_by_position below).
# height_in_cm isn't here - it's not even a model feature anymore, see
# add_height_vs_position.
POSITION_CONDITIONAL_FEATURES = [
    "age_at_transfer", "pre_apps", "pre_minutes",
    "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app",
    "log_value_before", "log_from_club_value", "log_to_club_value", "club_quality_ratio",
]
# PRETRANSFER_FOTMOB_COMPOSITES is deliberately NOT here, unlike the raw
# stats it replaced: a percentile-within-position feature is already
# position-relative by construction (its distribution has the same ~50
# median regardless of which position's rows it came from), so the flat
# reference_values median below is already the right "typical" comparison -
# no position-conditional lookup needed for these specifically.

# The 5 real Transfermarkt pre-transfer performance stats a live prediction
# can be missing entirely for (a player currently at a club outside
# LEAGUE_MAP - see app/main.py:impute_recent_performance) - never missing
# in the training data itself (every historical transfer has real
# appearance history), so this only matters at serve time. Structurally
# dependent on each other (pre_minutes roughly equals pre_apps times
# pre_mins_per_app) - see recent_performance_samples below and
# app/main.py:PLAYING_TIME_FEATURES.
RECENT_PERFORMANCE_FEATURES = ["pre_apps", "pre_minutes", "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app"]


def add_derived_features(df):
    """Add the log-transformed and NaN-filled columns NUMERIC_FEATURES/CATEGORICAL_FEATURES expect but transfers_processed.csv doesn't already have."""
    df = df.copy()
    df["log_transfer_fee"] = np.log1p(df["transfer_fee"].fillna(0))
    df["log_value_before"] = np.log1p(df["value_before"])
    df["log_from_club_value"] = np.log1p(df["from_total_market_value"])
    df["log_to_club_value"] = np.log1p(df["to_total_market_value"])
    df["foot"] = df["foot"].fillna("unknown")
    df["from_domestic_competition_id"] = df["from_domestic_competition_id"].fillna("unknown")
    df["to_domestic_competition_id"] = df["to_domestic_competition_id"].fillna("unknown")
    return df


def add_height_vs_position(df, position_means):
    """
    height_in_cm on its own tested as a bigger model input than the
    position category itself (3.4% feature importance vs. 1.7% for
    position's one-hot columns combined) despite being far less
    informative on its own - it was mostly acting as a silent proxy for
    position/body-type (tall -> CB/striker/keeper) rather than earning its
    weight for the positions height genuinely matters for (aerial duels).
    Centering height on its own position's average tested as a small but
    consistent MAE/R^2 improvement over both the raw value and dropping
    it outright (see git history for the comparison). `position_means`
    is fit on whichever population df was drawn from (train-only for the
    holdout eval below, full dataset for the deployed refit) - same
    fit-on-train/apply-to-test discipline as any other fitted transform.
    """
    df = df.copy()
    default = position_means.get("_default", 0.0)
    df["height_vs_position"] = df["height_in_cm"] - df["position"].map(position_means).fillna(default)
    return df


def main():
    """
    Load the processed transfers, fit a GradientBoostingRegressor on
    pre-transfer-only features, evaluate it on a temporal holdout (train on
    transfers before SPLIT_DATE, test on transfers since - so the reported
    accuracy reflects predicting *future* transfers, not just interpolating
    within the training period), then refit on the full dataset and save
    three artifacts to app/model/: the trained pipeline (model.joblib), a
    nearest-neighbors index over the same features for finding comparable
    historical transfers (comparables.joblib), and metadata.json (feature
    lists, test metrics, and per-feature reference/typical values used to
    explain predictions).
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["transfer_date"])
    df = add_derived_features(df)
    # height_vs_position isn't computed yet (needs a fitted position-mean -
    # see add_height_vs_position), so dropna against its source column
    # instead. The FotMob composites/flags are excluded entirely - missing
    # rows get imputed (see impute_pretransfer_fotmob_composites), not
    # dropped, and aren't computed yet at this point anyway (that needs a
    # fitted percentile table - see build_pretransfer_percentile_tables).
    required = [
        f for f in NUMERIC_FEATURES
        if f != "height_vs_position" and f not in PRETRANSFER_FOTMOB_COMPOSITES and f not in PRETRANSFER_FOTMOB_HAS_DATA_FLAGS
    ] + ["height_in_cm"]
    df = df.dropna(subset=required + CATEGORICAL_FEATURES + [TARGET])

    train = df[df["transfer_date"] < SPLIT_DATE].copy()
    test = df[df["transfer_date"] >= SPLIT_DATE].copy()
    print(f"Train: {len(train):,} transfers before {SPLIT_DATE}")
    print(f"Test:  {len(test):,} transfers on/after {SPLIT_DATE}")

    # Fit the position-height baseline on the train split only, same
    # discipline as fitting the model itself - test rows get the train-fit
    # mean applied to them, not their own.
    position_height_means = train.groupby("position")["height_in_cm"].mean().to_dict()
    position_height_means["_default"] = float(train["height_in_cm"].mean())
    train = add_height_vs_position(train, position_height_means)
    test = add_height_vs_position(test, position_height_means)

    # Same fit-on-train discipline for the pre-transfer FotMob percentile
    # tables and composite medians.
    pretransfer_tables = build_pretransfer_percentile_tables(train)
    train = compute_pretransfer_fotmob_composites(train, pretransfer_tables)
    test = compute_pretransfer_fotmob_composites(test, pretransfer_tables)
    pretransfer_composite_medians = {feat: float(train[feat].median()) for feat in PRETRANSFER_FOTMOB_COMPOSITES}
    train = impute_pretransfer_fotmob_composites(train, pretransfer_composite_medians)
    test = impute_pretransfer_fotmob_composites(test, pretransfer_composite_medians)

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    # min_frequency=30 folds any league with under 30 training rows (as
    # either origin or destination - Norway, Serbia, Austria, Sweden,
    # Croatia, Romania, Poland's DESTINATION column specifically, in
    # practice) into one shared "infrequent" bucket instead of giving it its
    # own one-hot column - see the Ridge-vs-GBR comment below for why this
    # matters specifically for a linear model. handle_unknown does the same
    # for a league that's genuinely unseen at fit time (rather than the
    # GBR-era "ignore", which zeroed it out - a live prediction for a truly
    # novel league now gets the "other rare league" coefficient instead of
    # no signal at all).
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), CATEGORICAL_FEATURES),
    ])

    # Ridge, not GradientBoostingRegressor - checked directly (see git
    # history / README) after a routine "any other backend gaps?" pass
    # turned up nothing new in the scoring pipeline and moved on to the
    # model itself. Every tree-based alternative tried (HistGradientBoosting,
    # RandomForest, ExtraTrees, and GBR variants shallower/deeper/more-or-
    # less regularized than the config this replaced) either lost to that
    # config or barely moved it; a plain Ridge beat all of them, including
    # unregularized OLS landing at essentially the same R^2 - the signal in
    # this feature set is close enough to linear, and this training set
    # small enough (~5,900-7,800 rows), that a tree ensemble's extra
    # flexibility was fitting noise rather than real structure. Checked
    # across 5 different temporal split dates (2022-01-01 through
    # 2024-01-01), not just the one SPLIT_DATE below, since Ridge has no
    # random seed of its own to check stability across the way the old
    # GBR config was checked over 5-10 seeds - Ridge won every single split,
    # by 0.017-0.041 R^2 each time, never once losing. alpha=1.0 sits in the
    # middle of a wide, flat 0.3-30 plateau, not a knife-edge optimum.
    #
    # Blending Ridge with the old GBR was tried too, in case the two had
    # complementary signal - looked promising on the one SPLIT_DATE holdout
    # (peeking at the exact best blend weight there is itself a mild form of
    # tuning-on-the-test-set, which is why this was re-checked across the
    # same 5 splits instead of trusted on one) but didn't hold up: a fixed
    # 50/50 blend beat plain Ridge on 4 of 5 splits by a small margin and
    # lost on the 5th - not a reliable win, and not worth permanently
    # shipping two models (double the serialization/prediction cost, and a
    # muddier "why this score" swap-based explanation) for that.
    #
    # OneHotEncoder's default (no min_frequency) also beat the GBR on every
    # split, by about the same margin - but its coefficients for the
    # thinnest leagues were unusable for a live model: a league with 2-9
    # training rows (Norway, Serbia, Romania) got coefficients as large as
    # +21.96, essentially memorizing those specific rows' targets rather
    # than learning a real per-league effect. min_frequency=30 costs at most
    # ~0.002 R^2 on any of the 5 splits (often nothing) while keeping every
    # coefficient in a plausible range - the same "don't trust a baseline
    # from too few examples" bar MIN_LEAGUE_SAMPLE/MIN_FOTMOB_LEAGUE_BASELINE_ROWS
    # already apply elsewhere in this codebase, just applied to the model's
    # own league dummies instead of a hand-computed baseline.
    model = Ridge(alpha=1.0)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    # See RECENCY_HALF_LIFE_YEARS above for why training rows are weighted
    # by recency - reference_date is train's own max date, not test's,
    # since test rows (and how "recent" they are relative to serving a live
    # prediction) must never influence how the model is fit.
    train_weight = compute_recency_weight(train["transfer_date"], train["transfer_date"].max())
    pipeline.fit(X_train, y_train, model__sample_weight=train_weight)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, np.full_like(preds, y_train.mean()))
    print(f"Test MAE: {mae:.2f} (baseline / predict-the-mean MAE: {baseline_mae:.2f})")
    print(f"Test R^2: {r2:.3f}")

    print("\nRefitting on full dataset for the deployed model...")
    # Refit the position-height baseline on the full dataset too, same as
    # the model itself - the train-only version above is only for an
    # honest holdout eval, not what actually ships.
    position_height_means_full = df.groupby("position")["height_in_cm"].mean().to_dict()
    position_height_means_full["_default"] = float(df["height_in_cm"].mean())
    df = add_height_vs_position(df, position_height_means_full)
    pretransfer_tables_full = build_pretransfer_percentile_tables(df)
    df = compute_pretransfer_fotmob_composites(df, pretransfer_tables_full)
    pretransfer_composite_medians_full = {feat: float(df[feat].median()) for feat in PRETRANSFER_FOTMOB_COMPOSITES}
    df = impute_pretransfer_fotmob_composites(df, pretransfer_composite_medians_full)
    X_all, y_all = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
    full_weight = compute_recency_weight(df["transfer_date"], df["transfer_date"].max())
    pipeline.fit(X_all, y_all, model__sample_weight=full_weight)

    print("Building comparable-transfers index...")
    comp_scaler = StandardScaler()
    comp_matrix = comp_scaler.fit_transform(df[NUMERIC_FEATURES])
    comp_index = NearestNeighbors(n_neighbors=6).fit(comp_matrix)
    comp_meta = df[[
        "name", "transfer_date", "from_club_name", "to_club_name", "success_score",
    ]].reset_index(drop=True)

    # "Typical transfer" reference values, used at prediction time to explain
    # a score by comparing each feature's actual value to this baseline (see
    # app/main.py:explain_prediction).
    reference_values = {f: float(df[f].median()) for f in NUMERIC_FEATURES}
    reference_values.update({f: df[f].mode().iloc[0] for f in CATEGORICAL_FEATURES})

    # log_transfer_fee and fee_to_value_ratio are both 0 for free transfers,
    # which are >50% of the dataset (out-of-contract moves, academy
    # graduates - a fundamentally different circumstance from an active
    # paid deal). That drags their overall median to 0, so "vs. a typical
    # transfer's €0m" is a misleading comparison for any transfer that DID
    # involve a fee. Separate "typical paid transfer" references, used
    # instead of the overall ones whenever the transfer being explained
    # itself has a nonzero fee (see app/main.py:explain_prediction).
    paid = df[df["transfer_fee"].fillna(0) > 0]
    reference_values_paid = {
        "log_transfer_fee": float(paid["log_transfer_fee"].median()),
        "fee_to_value_ratio": float(paid["fee_to_value_ratio"].median()),
    }
    pct_free_transfers = float((df["transfer_fee"].fillna(0) == 0).mean())

    # Same idea, more general: a striker's typical goal contributions are
    # nothing like the whole population's - attackers average 0.46 goal
    # contributions/90 pre-transfer, defenders 0.09, so comparing either
    # against an overall median of 0.20 is misleading in both directions.
    # Median per position, used instead of the flat reference for
    # POSITION_CONDITIONAL_FEATURES. height_vs_position doesn't need this -
    # it's already centered on its own position's average by construction
    # (see add_height_vs_position), so the flat reference_values median
    # below (naturally close to 0) is the right comparison for it too. Not
    # computed from a pre-imputation snapshot here (unlike the FotMob
    # composites, which needed one) - none of POSITION_CONDITIONAL_FEATURES
    # get touched by any imputation step, so `df` at this point already has
    # only real values for all of them. None (JSON null), not NaN, for a
    # position with no real values at all for a given feature - float("nan")
    # isn't valid JSON and breaks serialization; app/main.py's
    # explain_prediction treats a missing/None entry the same way (falls
    # back to the flat reference).
    reference_values_by_position = {
        position: {
            f: (None if pd.isna(sub[f].median()) else float(sub[f].median()))
            for f in POSITION_CONDITIONAL_FEATURES
        }
        for position, sub in df.groupby("position")
    }

    # sub_position needs the same position-conditional treatment as
    # POSITION_CONDITIONAL_FEATURES above, but as a MODE, not a median -
    # it's categorical. reference_values["sub_position"] (the flat,
    # dataset-wide mode) resolves to the single most common sub-position
    # across ALL positions combined (Centre-Forward, the single largest
    # sub-position group) - comparing e.g. a Left-Back against "a typical
    # Centre-Forward" in app/main.py's explain_prediction swaps sub_position
    # alone while every other feature (goal output, defensive stats, ...)
    # stays at the real Defender's values, producing a synthetic row that
    # never exists in real data and an outsized, misleading swap
    # contribution - checked directly on a real Left-Back-to-big-club
    # prediction: -13.7 pts using the flat mode vs. -0.2 pts using the
    # position-conditional mode (Centre-Back, the modal Defender sub-role).
    sub_position_reference_by_position = {
        position: sub["sub_position"].mode().iloc[0]
        for position, sub in df.groupby("position")
    }

    # For a live prediction with no real recent-performance data at all
    # (see app/main.py:impute_recent_performance - a player currently at a
    # club outside LEAGUE_MAP, e.g. Messi at Inter Miami), a single median
    # point is one arbitrary guess at an unknown 5-feature combination, and
    # a tree ensemble's response to a feature isn't linear - E[f(X)] !=
    # f(E[X]). Checked directly: for a real missing-data case, predicting
    # with the median point gave 55.6, but averaging the prediction over
    # every real attacker's actual (self-consistent) profile gave a mean of
    # 57.0 with real spread (52.9-63.1 across individual attackers) - the
    # median point isn't neutral, it's just one specific (and here,
    # somewhat pessimistic) guess. Every real position-matched row of
    # RECENT_PERFORMANCE_FEATURES is stored here (not a random subsample -
    # each position has at most ~1,600 rows, cheap to store and to average
    # over in one batched pipeline.predict() call at serve time, ~35ms for
    # 2,585 rows, tested directly) so app/main.py can marginalize over the
    # real empirical distribution instead of committing to one point guess.
    # Whole rows, not independently-sampled columns - these 5 features are
    # structurally dependent (see PLAYING_TIME_FEATURES in app/main.py), so
    # sampling each column on its own would recreate the exact "impossible
    # combination" bug already fixed for the leave-one-out explanation.
    recent_performance_samples = {
        position: sub[RECENT_PERFORMANCE_FEATURES].to_dict(orient="records")
        for position, sub in df.groupby("position")
    }

    # A €100m fee for a player already valued at €70m isn't remarkable -
    # it's a ~1.4x premium, in line with what similarly-valued players go
    # for. But comparing the raw €100m against a flat "typical paid fee"
    # (~€6m, dragged down by many cheaper deals) makes it look like a huge
    # outlier regardless of the player's own value. Fit fee ~ value (in log
    # space, paid transfers only) so the reference for a given prediction
    # is "what's typically paid for a player valued this highly", not a
    # single number for everyone (see app/main.py:explain_prediction).
    fee_slope, fee_intercept = np.polyfit(paid["log_value_before"], paid["log_transfer_fee"], 1)
    fee_regression = {"slope": float(fee_slope), "intercept": float(fee_intercept)}

    # Historical context for the "origin/destination club value" explanation
    # (see app/main.py:explain_prediction) - a bare "€900m vs. a typical
    # attacker's €172m" swing never says why a bigger squad valuation
    # matters. Checked directly on this same filtered df: of every
    # success_score component, rating_pct correlates with log(to_club_value)
    # far more than any other (r=0.29, vs. 0.23 for attacking_pct, 0.22 for
    # possession_pct, -0.07 for defensive_pct - moving to a bigger-budget
    # club doesn't uniformly help every component, it's concentrated in
    # post-move rating). log(from_club_value) shows the same pattern, weaker
    # (r=0.19) - players leaving a bigger club already tend to be better
    # players, so they keep rating well after leaving too. Fit rating_pct ~
    # log(value) for both directions so the explanation can quote the actual
    # regression-implied percentile gap for *this* prediction's specific
    # values, the same "fit a line, evaluate it at this transfer's own
    # numbers" idea as fee_regression above, rather than a single baked-in
    # number for everyone.
    rating_known = df.dropna(subset=["rating_pct"])
    to_slope, to_intercept = np.polyfit(rating_known["log_to_club_value"], rating_known["rating_pct"], 1)
    from_slope, from_intercept = np.polyfit(rating_known["log_from_club_value"], rating_known["rating_pct"], 1)
    club_value_rating_regression = {
        "to": {"slope": float(to_slope), "intercept": float(to_intercept)},
        "from": {"slope": float(from_slope), "intercept": float(from_intercept)},
    }

    # Historical context for the "destination/origin league" explanation
    # (see app/main.py:explain_prediction/league_context_note) - a bare
    # "vs. a typical transfer's Premier League" swing with no explanation
    # reads as "moving to Spain is inherently better". This used to also
    # compute a per-league average fee-to-value premium and cite it as the
    # driver of the gap - removed (see league_context_note's docstring)
    # after checking directly that a league's fee premium barely
    # correlates with its average success_score across the 14 leagues with
    # a trustworthy sample (r=+0.21, p=0.47). So this just states the real
    # baseline gap as a fact without inventing a cause, same pattern
    # position_success_baseline below already used. A league with too few
    # transfers to trust a stable average (MIN_LEAGUE_SAMPLE) is left out
    # entirely rather than shown a noisy number - falls back to the plain
    # swing-only explanation.
    MIN_LEAGUE_SAMPLE = 15
    to_counts = df["to_domestic_competition_id"].value_counts()
    league_success_baseline_to = {
        lg: round(float(v), 1)
        for lg, v in df.groupby("to_domestic_competition_id")[TARGET].mean().items()
        if to_counts.get(lg, 0) >= MIN_LEAGUE_SAMPLE
    }
    from_counts = df["from_domestic_competition_id"].value_counts()
    league_success_baseline_from = {
        lg: round(float(v), 1)
        for lg, v in df.groupby("from_domestic_competition_id")[TARGET].mean().items()
        if from_counts.get(lg, 0) >= MIN_LEAGUE_SAMPLE
    }

    # Historical context for the "position"/"sub-position" explanation (see
    # app/main.py:explain_prediction) - swapping a transfer's actual
    # position against whichever category happens to be the dataset's mode
    # (Defender/Centre-Back) read as "being an attacker instead of a
    # defender" causing the swing, with no explanation at all. Checked
    # directly on this same filtered df: real, if modest, baseline
    # differences do exist by position (50.0 for goalkeepers to 54.6 for
    # midfielders) and sub-position (50.4 for centre-backs to 65.0 for
    # second strikers) - but nothing here isolates *why* (could be the
    # scoring formula's own per-position weighting, could be market-
    # evaluation differences, could be both), so the note states the
    # baseline gap as a fact without inventing a mechanism, same pattern
    # league_context_note above now uses too. Same MIN_LEAGUE_SAMPLE gate
    # as the league baselines - several sub-positions (e.g. "Attack", 2
    # rows) are too thin to trust.
    position_counts = df["position"].value_counts()
    position_success_baseline = {
        p: round(float(v), 1)
        for p, v in df.groupby("position")[TARGET].mean().items()
        if position_counts.get(p, 0) >= MIN_LEAGUE_SAMPLE
    }
    sub_position_counts = df["sub_position"].value_counts()
    sub_position_success_baseline = {
        p: round(float(v), 1)
        for p, v in df.groupby("sub_position")[TARGET].mean().items()
        if sub_position_counts.get(p, 0) >= MIN_LEAGUE_SAMPLE
    }

    # Historical context for the "foot" explanation - checked directly on
    # this same filtered df: left-footed transfers average 54.3, "both"
    # 55.3, right-footed 52.4 - a real but small gap with no clear football
    # mechanism found behind it (unlike league/club-value above), so the
    # note is deliberately hedged rather than asserting a cause that hasn't
    # been verified.
    foot_counts = df["foot"].value_counts()
    foot_success_baseline = {
        f: round(float(v), 1)
        for f, v in df.groupby("foot")[TARGET].mean().items()
        if foot_counts.get(f, 0) >= MIN_LEAGUE_SAMPLE
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(
        {"scaler": comp_scaler, "index": comp_index, "meta": comp_meta, "features": NUMERIC_FEATURES},
        os.path.join(MODEL_DIR, "comparables.joblib"),
    )
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "test_mae": round(mae, 2),
            "test_r2": round(r2, 3),
            "baseline_mae": round(baseline_mae, 2),
            "n_train": len(train),
            "n_test": len(test),
            "reference_values": reference_values,
            "reference_values_paid": reference_values_paid,
            "reference_values_by_position": reference_values_by_position,
            "sub_position_reference_by_position": sub_position_reference_by_position,
            "recent_performance_samples": recent_performance_samples,
            "position_conditional_features": POSITION_CONDITIONAL_FEATURES,
            "fee_regression": fee_regression,
            "club_value_rating_regression": club_value_rating_regression,
            "pct_free_transfers": round(pct_free_transfers, 3),
            # Needed at serving time to turn a hypothetical prediction's raw
            # height_in_cm into the height_vs_position feature the model
            # actually expects - see app/main.py:build_feature_row.
            "position_height_means": position_height_means_full,
            "league_success_baseline_to": league_success_baseline_to,
            "league_success_baseline_from": league_success_baseline_from,
            "position_success_baseline": position_success_baseline,
            "sub_position_success_baseline": sub_position_success_baseline,
            "foot_success_baseline": foot_success_baseline,
            # Needed at serving time to turn a hypothetical prediction's raw
            # pre_fotmob_* inputs into the 4 position-relative composites the
            # model actually expects (pretransfer_percentile_tables for the
            # per-stat/position lookup, pretransfer_fotmob_composite_medians
            # for a bucket with no real data at all) - see
            # app/main.py:build_feature_row.
            "pretransfer_percentile_tables": pretransfer_tables_full,
            "pretransfer_fotmob_composite_medians": pretransfer_composite_medians_full,
            "pretransfer_fotmob_raw_stats": PRETRANSFER_FOTMOB_FEATURES,
            "pretransfer_fotmob_composites": PRETRANSFER_FOTMOB_COMPOSITES,
            "pretransfer_inverted_stats": list(PRETRANSFER_INVERTED_STATS),
        }, f, indent=2)
    print(f"Saved model + metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
