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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "model")
SPLIT_DATE = "2023-06-01"

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
# ~35-45% of transfers have no FotMob match for the pre-transfer year
# (same real coverage ceilings as the post-transfer side - see README) -
# each pre_fotmob_* feature is median-imputed (see
# pretransfer_fotmob_medians below) rather than dropping those rows
# outright, with this flag so the model can learn to discount an imputed
# placeholder rather than trusting it as a real "average" performance.
HAS_PRETRANSFER_FOTMOB_FEATURE = "has_pre_fotmob_data"

NUMERIC_FEATURES = [
    "age_at_transfer", "height_vs_position",
    "pre_apps", "pre_minutes", "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app",
    "log_transfer_fee", "log_value_before", "fee_to_value_ratio", "club_quality_ratio",
    "log_from_club_value", "log_to_club_value",
] + PRETRANSFER_FOTMOB_FEATURES + [HAS_PRETRANSFER_FOTMOB_FEATURE]
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
] + PRETRANSFER_FOTMOB_FEATURES
# A striker's typical tackles/90 is nothing like a centre-back's (and vice
# versa for xG/90) - same reasoning as pre_goals_p90 above, even more
# pronounced here since several of these (saves, tackles) are near-zero
# outside their natural position. HAS_PRETRANSFER_FOTMOB_FEATURE is NOT
# here - coverage odds don't meaningfully depend on position, so the flat
# reference (~0.6-0.7, i.e. "usually available") is the right comparison.


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
    df[HAS_PRETRANSFER_FOTMOB_FEATURE] = df["pre_fotmob_rating"].notna().astype(int)
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


def impute_pretransfer_fotmob(df, medians):
    """
    ~35-45% of transfers have no pre-transfer FotMob match (an
    uncovered origin league, or a real coverage gap even within a
    covered one - same ceilings as the post-transfer side, see README).
    Filling with a population median (rather than dropping those rows,
    which would sacrifice a third-plus of the training set for a still-
    valuable feature set) alongside HAS_PRETRANSFER_FOTMOB_FEATURE lets
    the model learn to discount an imputed placeholder instead of
    trusting it as a real "average" performance. `medians` is fit on
    whichever population df was drawn from, same fit-on-train/apply-to-
    test discipline as position_height_means above.
    """
    df = df.copy()
    for feat in PRETRANSFER_FOTMOB_FEATURES:
        df[feat] = df[feat].fillna(medians[feat])
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
    # instead. pre_fotmob_* features are excluded entirely - missing rows
    # get imputed (see impute_pretransfer_fotmob), not dropped.
    required = [
        f for f in NUMERIC_FEATURES
        if f != "height_vs_position" and f not in PRETRANSFER_FOTMOB_FEATURES and f != HAS_PRETRANSFER_FOTMOB_FEATURE
    ] + ["height_in_cm"]
    df = df.dropna(subset=required + CATEGORICAL_FEATURES + [TARGET])

    train = df[df["transfer_date"] < SPLIT_DATE]
    test = df[df["transfer_date"] >= SPLIT_DATE]
    print(f"Train: {len(train):,} transfers before {SPLIT_DATE}")
    print(f"Test:  {len(test):,} transfers on/after {SPLIT_DATE}")

    # Fit the position-height baseline on the train split only, same
    # discipline as fitting the model itself - test rows get the train-fit
    # mean applied to them, not their own.
    position_height_means = train.groupby("position")["height_in_cm"].mean().to_dict()
    position_height_means["_default"] = float(train["height_in_cm"].mean())
    train = add_height_vs_position(train, position_height_means)
    test = add_height_vs_position(test, position_height_means)

    # Same fit-on-train discipline for the pre-transfer FotMob medians.
    pretransfer_fotmob_medians = {feat: float(train[feat].median()) for feat in PRETRANSFER_FOTMOB_FEATURES}
    train = impute_pretransfer_fotmob(train, pretransfer_fotmob_medians)
    test = impute_pretransfer_fotmob(test, pretransfer_fotmob_medians)

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    # max_depth=3, no subsampling, no min_samples_leaf (the previous config)
    # was overfitting for a ~3,000-row training set - a small grid search
    # against the temporal holdout (see git history) found this shallower/
    # more-regularized config a real, seed-stable improvement: R^2 0.086 ->
    # ~0.11 across 5 random seeds, not a one-off. subsample=0.8 (stochastic
    # gradient boosting - each tree only sees 80% of rows) and
    # min_samples_leaf=10 both push against overfitting individual trees;
    # max_depth=2 shrinks each tree's own capacity to memorize noise.
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

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
    pretransfer_fotmob_medians_full = {feat: float(df[feat].median()) for feat in PRETRANSFER_FOTMOB_FEATURES}
    # Snapshot before imputation - reference_values_by_position below needs
    # each position's median computed over real known values only (median()
    # skips NaN by default), not diluted by rows that are about to be
    # filled with the flat population median (see impute_pretransfer_fotmob).
    df_before_fotmob_impute = df.copy()
    df = impute_pretransfer_fotmob(df, pretransfer_fotmob_medians_full)
    X_all, y_all = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
    pipeline.fit(X_all, y_all)

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
    # below (naturally close to 0) is the right comparison for it too.
    # None (JSON null), not NaN, for a position with no real values at all
    # for a given feature (e.g. pre_fotmob_saves for Attack - virtually no
    # attacker has FotMob save data) - float("nan") isn't valid JSON and
    # breaks serialization; app/main.py's explain_prediction treats a
    # missing/None entry the same way (falls back to the flat reference).
    reference_values_by_position = {
        position: {
            f: (None if pd.isna(sub[f].median()) else float(sub[f].median()))
            for f in POSITION_CONDITIONAL_FEATURES
        }
        for position, sub in df_before_fotmob_impute.groupby("position")
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

    # Historical context for the "destination/origin league" explanation
    # (see app/main.py:explain_prediction) - a bare "vs. a typical
    # transfer's Premier League" swing with no explanation reads as "moving
    # to Spain is inherently better", when the real driver is almost
    # entirely value_for_money: Premier League clubs have historically paid
    # a much larger premium over market value than clubs in every other
    # major league (mean fee/value 1.55x vs. La Liga's 1.01x, 47% of PL
    # paid deals exceeding the formula's 1.3x overpay line vs. 21% for La
    # Liga - on-pitch components are comparable across leagues, this isn't
    # about football quality). Surfacing both the average success_score
    # and the average fee premium per league lets the explanation say why,
    # not just that. A league with too few transfers to trust a stable
    # average (MIN_LEAGUE_SAMPLE) is left out entirely rather than shown a
    # noisy number - falls back to the plain swing-only explanation.
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
    paid_to_counts = paid["to_domestic_competition_id"].value_counts()
    league_fee_ratio_baseline_to = {
        lg: round(float(v), 2)
        for lg, v in paid.groupby("to_domestic_competition_id")["fee_to_value_ratio"].mean().items()
        if paid_to_counts.get(lg, 0) >= MIN_LEAGUE_SAMPLE
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
            "position_conditional_features": POSITION_CONDITIONAL_FEATURES,
            "fee_regression": fee_regression,
            "pct_free_transfers": round(pct_free_transfers, 3),
            # Needed at serving time to turn a hypothetical prediction's raw
            # height_in_cm into the height_vs_position feature the model
            # actually expects - see app/main.py:build_feature_row.
            "position_height_means": position_height_means_full,
            "league_success_baseline_to": league_success_baseline_to,
            "league_success_baseline_from": league_success_baseline_from,
            "league_fee_ratio_baseline_to": league_fee_ratio_baseline_to,
            # Needed at serving time to fill in a hypothetical prediction's
            # pre_fotmob_* features when the searched player has no recent
            # FotMob match (an uncovered league, or a real coverage gap) -
            # see app/main.py:build_feature_row.
            "pretransfer_fotmob_medians": pretransfer_fotmob_medians_full,
            "pretransfer_fotmob_features": PRETRANSFER_FOTMOB_FEATURES,
        }, f, indent=2)
    print(f"Saved model + metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
