import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = FastAPI(title="Transfer Success Predictor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

pipeline = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
comparables = joblib.load(os.path.join(MODEL_DIR, "comparables.joblib"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

transfers_df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
players_df = pd.read_csv(os.path.join(DATA_DIR, "players_lookup.csv"))
clubs_df = pd.read_csv(os.path.join(DATA_DIR, "clubs_lookup.csv"))

NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

EXAMPLE_TRANSFER_KEYS = [
    ("Erling Haaland", "Man City"),
    ("Ousmane Dembélé", "Barcelona"),
    ("Cole Palmer", "Chelsea"),
    ("Jadon Sancho", "Man Utd"),
]

SCORE_COMPONENTS = [
    ("perf_level_pct", "Performance level", "Goal contributions per 90 vs. same-position peers"),
    ("perf_delta_pct", "Performance change", "Improved or declined vs. their level before the move"),
    ("value_growth_pct", "Market value growth", "How much the player's valuation rose"),
    ("playing_time_pct", "Playing time", "Appearances made - established starter vs. bench/injured"),
    ("value_for_money_pct", "Value for money", "Output vs. what was paid, relative to their market value"),
]

FEATURE_LABELS = {
    "age_at_transfer": "Age at transfer",
    "height_in_cm": "Height",
    "pre_apps": "Recent appearances",
    "pre_minutes": "Recent minutes played",
    "pre_goals_p90": "Recent goals per 90",
    "pre_ga_p90": "Recent goal contributions per 90",
    "pre_mins_per_app": "Minutes per appearance",
    "log_transfer_fee": "Transfer fee",
    "log_value_before": "Market value before the move",
    "fee_to_value_ratio": "Fee relative to market value",
    "club_quality_ratio": "Step up/down in club quality",
    "log_from_club_value": "Origin club's squad value",
    "log_to_club_value": "Destination club's squad value",
    "position": "Position",
    "foot": "Preferred foot",
    "from_domestic_competition_id": "Origin league",
    "to_domestic_competition_id": "Destination league",
}


class PredictRequest(BaseModel):
    age_at_transfer: float = Field(..., ge=15, le=42)
    height_in_cm: float = Field(..., ge=150, le=210)
    position: str
    foot: str
    pre_apps: float = Field(..., ge=0)
    pre_minutes: float = Field(..., ge=0)
    pre_goals_p90: float = Field(..., ge=0)
    pre_ga_p90: float = Field(..., ge=0)
    pre_mins_per_app: float = Field(..., ge=0)
    transfer_fee: float = Field(..., ge=0)
    value_before: float = Field(..., gt=0)
    from_domestic_competition_id: str
    to_domestic_competition_id: str
    from_total_market_value: float = Field(..., ge=0)
    to_total_market_value: float = Field(..., ge=0)


def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    row = req.model_dump()
    row["log_transfer_fee"] = np.log1p(row["transfer_fee"])
    row["log_value_before"] = np.log1p(row["value_before"])
    row["log_from_club_value"] = np.log1p(row["from_total_market_value"])
    row["log_to_club_value"] = np.log1p(row["to_total_market_value"])
    row["fee_to_value_ratio"] = row["transfer_fee"] / max(row["value_before"], 1)
    row["club_quality_ratio"] = row["to_total_market_value"] / max(row["from_total_market_value"], 1)
    return pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def explain_prediction(feature_row: pd.DataFrame, base_score: float, top_k: int = 5):
    """
    Approximate per-feature contributions by swapping one feature at a time
    to its "typical transfer" reference value (median/mode from training
    data) and seeing how much the prediction moves. A positive contribution
    means the actual value pushed the score up relative to a typical
    transfer; negative means it pulled the score down. This is a simple,
    transparent stand-in for a proper SHAP explanation.
    """
    reference = metadata["reference_values"]
    contributions = []
    for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        modified = feature_row.copy()
        modified[feat] = reference[feat]
        modified_score = float(pipeline.predict(modified)[0])
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "contribution": round(base_score - modified_score, 1),
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_k]


def find_comparables(feature_row: pd.DataFrame, k: int = 5):
    x = feature_row[comparables["features"]].values
    x_scaled = comparables["scaler"].transform(x)
    dist, idx = comparables["index"].kneighbors(x_scaled, n_neighbors=k)
    rows = comparables["meta"].iloc[idx[0]]
    out = []
    for _, r in rows.iterrows():
        out.append({
            "name": r["name"],
            "transfer_date": str(r["transfer_date"])[:10],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "success_score": float(r["success_score"]),
        })
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "model_metadata": metadata}


@app.get("/api/examples")
def examples():
    out = []
    for name, to_club in EXAMPLE_TRANSFER_KEYS:
        match = transfers_df[
            (transfers_df["name"] == name) & (transfers_df["to_club_name"] == to_club)
        ]
        if match.empty:
            continue
        r = match.iloc[-1]
        breakdown = [
            {"label": label, "description": desc, "value": round(float(r[key]), 1)}
            for key, label, desc in SCORE_COMPONENTS
        ]
        out.append({
            "name": r["name"],
            "transfer_date": str(r["transfer_date"])[:10],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "success_score": float(r["success_score"]),
            "pre_ga_p90": round(float(r["pre_ga_p90"]), 2),
            "post_ga_p90": round(float(r["post_ga_p90"]), 2),
            "tenure_days": int(r["tenure_days"]),
            "still_at_club": bool(r["still_at_club"]),
            "breakdown": breakdown,
        })
    return out


@app.get("/api/players/search")
def search_players(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    mask = players_df["name"].str.contains(q, case=False, na=False, regex=False)
    rows = players_df[mask].head(limit)
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/search")
def search_clubs(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    mask = clubs_df["name"].str.contains(q, case=False, na=False, regex=False)
    rows = clubs_df[mask].head(limit)
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/{club_id}")
def get_club(club_id: int):
    row = clubs_df[clubs_df["club_id"] == club_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="club not found")
    return row.iloc[0].fillna("").to_dict()


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        feature_row = build_feature_row(req)
        raw_score = float(pipeline.predict(feature_row)[0])
        score = max(0.0, min(100.0, raw_score))
        comps = find_comparables(feature_row)
        explanation = explain_prediction(feature_row, raw_score)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "success_score": round(score, 1),
        "comparable_transfers": comps,
        "explanation": explanation,
        "model_test_mae": metadata["test_mae"],
        "model_test_r2": metadata["test_r2"],
    }


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
