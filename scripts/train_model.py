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

NUMERIC_FEATURES = [
    "age_at_transfer", "height_in_cm",
    "pre_apps", "pre_minutes", "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app",
    "log_transfer_fee", "log_value_before", "fee_to_value_ratio", "club_quality_ratio",
    "log_from_club_value", "log_to_club_value",
]
CATEGORICAL_FEATURES = [
    "position", "foot", "from_domestic_competition_id", "to_domestic_competition_id",
]
TARGET = "success_score"


def add_derived_features(df):
    df = df.copy()
    df["log_transfer_fee"] = np.log1p(df["transfer_fee"].fillna(0))
    df["log_value_before"] = np.log1p(df["value_before"])
    df["log_from_club_value"] = np.log1p(df["from_total_market_value"])
    df["log_to_club_value"] = np.log1p(df["to_total_market_value"])
    df["foot"] = df["foot"].fillna("unknown")
    df["from_domestic_competition_id"] = df["from_domestic_competition_id"].fillna("unknown")
    df["to_domestic_competition_id"] = df["to_domestic_competition_id"].fillna("unknown")
    return df


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["transfer_date"])
    df = add_derived_features(df)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])

    train = df[df["transfer_date"] < SPLIT_DATE]
    test = df[df["transfer_date"] >= SPLIT_DATE]
    print(f"Train: {len(train):,} transfers before {SPLIT_DATE}")
    print(f"Test:  {len(test):,} transfers on/after {SPLIT_DATE}")

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
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
    X_all, y_all = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
    pipeline.fit(X_all, y_all)

    print("Building comparable-transfers index...")
    comp_scaler = StandardScaler()
    comp_matrix = comp_scaler.fit_transform(df[NUMERIC_FEATURES])
    comp_index = NearestNeighbors(n_neighbors=6).fit(comp_matrix)
    comp_meta = df[[
        "name", "transfer_date", "from_club_name", "to_club_name", "success_score",
    ]].reset_index(drop=True)

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
        }, f, indent=2)
    print(f"Saved model + metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
