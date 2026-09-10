"""
Compute an honest, held-out model prediction for every scored permanent
transfer, then save each one's gap vs. what actually happened
(data/prediction_surprises.csv) - the "biggest surprises" leaderboard's
data source (app/main.py's /api/surprises).

Reuses train_model.py's exact feature-engineering functions and feature
lists so this mirrors the deployed model's own training pipeline exactly -
just with 5-fold cross-validation swapped in for its one temporal split, so
every transfer gets a prediction from a model that never saw that
transfer's own outcome (or, for the FotMob percentile tables/composite
medians and the position-height baseline, its own row) during fitting.
The deployed model.joblib is never touched here - it's refit on the full
dataset for serving live predictions, which is the right call for accuracy,
but makes its own predictions on historical transfers partly circular (it
saw the answer). Cross-validated predictions avoid that, at the cost of
each one coming from a slightly different, held-out-observation model
rather than the one shipped to users - the right trade for "how surprising
was this outcome", the wrong one for "predict this live".
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.train_model import (
    CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES,
    PRETRANSFER_FOTMOB_COMPOSITES, PRETRANSFER_FOTMOB_HAS_DATA_FLAGS,
    TARGET, add_derived_features, add_height_vs_position,
    build_pretransfer_percentile_tables, compute_pretransfer_fotmob_composites,
    impute_pretransfer_fotmob_composites,
)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "prediction_surprises.csv")
N_FOLDS = 5
RANDOM_STATE = 42


def build_pipeline():
    """Same ColumnTransformer + Ridge(alpha=1.0) as train_model.py's deployed model - see that file for why."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])


def main():
    """Load transfers_processed.csv, generate a 5-fold out-of-fold prediction for every row that has one, and write data/prediction_surprises.csv."""
    df = pd.read_csv(DATA_PATH)
    df = add_derived_features(df)
    # Same required-columns filter as train_model.py's main() - a transfer
    # missing one of these was never part of the model's training data
    # either, so it can't get an honest prediction here.
    required = [
        f for f in NUMERIC_FEATURES
        if f != "height_vs_position" and f not in PRETRANSFER_FOTMOB_COMPOSITES and f not in PRETRANSFER_FOTMOB_HAS_DATA_FLAGS
    ] + ["height_in_cm"]
    df = df.dropna(subset=required + CATEGORICAL_FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Scoring {len(df):,} transfers across {N_FOLDS} folds")

    predicted = np.full(len(df), np.nan)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        # Fit every feature-engineering statistic on this fold's train rows
        # only, same discipline as train_model.py's temporal split - a test
        # row must never leak into the numbers used to build its own
        # features.
        position_height_means = train.groupby("position")["height_in_cm"].mean().to_dict()
        position_height_means["_default"] = float(train["height_in_cm"].mean())
        train = add_height_vs_position(train, position_height_means)
        test = add_height_vs_position(test, position_height_means)

        pretransfer_tables = build_pretransfer_percentile_tables(train)
        train = compute_pretransfer_fotmob_composites(train, pretransfer_tables)
        test = compute_pretransfer_fotmob_composites(test, pretransfer_tables)
        composite_medians = {feat: float(train[feat].median()) for feat in PRETRANSFER_FOTMOB_COMPOSITES}
        train = impute_pretransfer_fotmob_composites(train, composite_medians)
        test = impute_pretransfer_fotmob_composites(test, composite_medians)

        pipeline = build_pipeline()
        pipeline.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET])
        preds = pipeline.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        predicted[test_idx] = np.clip(preds, 0.0, 100.0)
        print(f"  fold {fold}/{N_FOLDS}: {len(test_idx):,} transfers predicted")

    df["predicted_score"] = predicted.round(1)
    df["surprise_delta"] = (df[TARGET] - df["predicted_score"]).round(1)

    mae = (df[TARGET] - df["predicted_score"]).abs().mean()
    ss_res = ((df[TARGET] - df["predicted_score"]) ** 2).sum()
    ss_tot = ((df[TARGET] - df[TARGET].mean()) ** 2).sum()
    print(f"\nOut-of-fold MAE: {mae:.2f}, R^2: {1 - ss_res / ss_tot:.3f}")

    out = df[["player_id", "transfer_date", "predicted_score", "surprise_delta"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
