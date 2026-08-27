import json
import os
import unicodedata

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


def fold_accents(value):
    """Lowercase with diacritics stripped, e.g. 'Dembélé' -> 'dembele', so a plain-ASCII search matches accented names."""
    if not isinstance(value, str):
        return value
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


pipeline = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
comparables = joblib.load(os.path.join(MODEL_DIR, "comparables.joblib"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

transfers_df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
players_df = pd.read_csv(os.path.join(DATA_DIR, "players_lookup.csv"))
clubs_df = pd.read_csv(os.path.join(DATA_DIR, "clubs_lookup.csv"))

players_df["_name_fold"] = players_df["name"].map(fold_accents)
clubs_df["_name_fold"] = clubs_df["name"].map(fold_accents)
transfers_df["_name_fold"] = transfers_df["name"].map(fold_accents)
transfers_df["_from_club_fold"] = transfers_df["from_club_name"].map(fold_accents)
transfers_df["_to_club_fold"] = transfers_df["to_club_name"].map(fold_accents)
competitions_df = pd.read_csv(os.path.join(DATA_DIR, "competitions_lookup.csv"))
_dup_names = competitions_df["name"][competitions_df["name"].duplicated(keep=False)]
competitions_df["display_name"] = competitions_df.apply(
    lambda r: f"{r['name']} ({r['country_name']})" if r["name"] in _dup_names.values and pd.notna(r["country_name"]) else r["name"],
    axis=1,
)
LEAGUE_NAMES = dict(zip(competitions_df["competition_id"], competitions_df["display_name"]))

NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

EXAMPLE_TRANSFER_KEYS = [
    ("Erling Haaland", "Man City"),
    ("Ousmane Dembélé", "Barcelona"),
    ("Cole Palmer", "Chelsea"),
    ("Jadon Sancho", "Man Utd"),
]

POSITION_PLURAL = {
    "Attack": "attackers", "Midfield": "midfielders",
    "Defender": "defenders", "Goalkeeper": "goalkeepers",
}


def resale_outcome_phrase(fee_paid, fee_received):
    fee_paid = 0 if pd.isna(fee_paid) else fee_paid
    if fee_received > fee_paid:
        return "— a profitable flip for the club, regardless of on-pitch performance"
    if fee_received < fee_paid:
        return "— sold for less than the club paid, a loss independent of on-pitch performance"
    return "— resold for the same fee paid, breaking even"


def describe_components(r):
    eur_m = lambda v: "free" if pd.isna(v) or v == 0 else f"€{v / 1_000_000:.0f}m"
    position_plural = POSITION_PLURAL.get(r["position"], r["position"])
    to_league = LEAGUE_NAMES.get(r["to_domestic_competition_id"], r["to_domestic_competition_id"])
    return [
        {
            "label": "Performance level",
            "value": round(float(r["perf_level_pct"]), 1),
            "description": (
                f"{r['post_ga_p90']:.2f} goal contributions/90 at {r['to_club_name']} "
                f"({r['post_ga_p90_vs_league']:.1f}x the {to_league} average for {position_plural}), "
                f"ranked vs. other {position_plural}"
            ),
        },
        {
            "label": "Performance change",
            "value": round(float(r["perf_delta_pct"]), 1),
            "description": (
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x — beat the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high (some pullback from a peak is normal)"
                if r["post_ga_p90_vs_league"] >= r["expected_post_ga_p90_vs_league"] else
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x — below the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high"
            ),
        },
        {
            "label": "Market value growth",
            "value": round(float(r["value_growth_pct"]), 1),
            "description": f"{eur_m(r['value_before'])} → {eur_m(r['value_after'])} market value",
        },
        {
            "label": "Playing time",
            "value": round(float(r["playing_time_pct"]), 1),
            "description": (
                f"{int(r['post_apps'])} of {int(r['team_games_in_tenure'])} games "
                f"{r['to_club_name']} played during the tenure "
                f"({r['pct_team_games_played'] * 100:.0f}% - captures injuries/rotation, "
                f"blended with raw appearance count for sustained presence)"
            ),
        },
        {
            "label": "Value for money",
            "value": round(float(r["value_for_money_pct"]), 1),
            "description": f"{eur_m(r['transfer_fee'])} fee vs. {eur_m(r['value_before'])} market value at the time",
        },
    ] + ([
        {
            "label": "Resale profit",
            "value": round(float(r["resale_profit_pct"]), 1),
            "description": (
                f"Bought for {eur_m(r['transfer_fee'])}, later resold for {eur_m(r['next_transfer_fee'])} "
                + resale_outcome_phrase(r["transfer_fee"], r["next_transfer_fee"])
            ),
        },
    ] if bool(r["has_resale_data"]) else [])

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

LOG_FEATURES = {"log_transfer_fee", "log_value_before", "log_from_club_value", "log_to_club_value"}


def format_feature_value(feat, value):
    if feat in LOG_FEATURES:
        value = np.expm1(value)
        return f"€{value / 1_000_000:.1f}m"
    if feat == "age_at_transfer":
        return f"{value:.1f} yrs"
    if feat == "height_in_cm":
        return f"{value:.0f} cm"
    if feat in ("pre_goals_p90", "pre_ga_p90"):
        return f"{value:.2f} per 90"
    if feat == "pre_apps":
        return f"{value:.0f} apps"
    if feat == "pre_minutes":
        return f"{value:.0f} mins"
    if feat == "pre_mins_per_app":
        return f"{value:.0f} min/app"
    if feat in ("fee_to_value_ratio", "club_quality_ratio"):
        return f"{value:.2f}×"
    if feat in ("from_domestic_competition_id", "to_domestic_competition_id"):
        return LEAGUE_NAMES.get(value, value)
    return str(value)


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
        actual_value = feature_row[feat].iloc[0]
        modified = feature_row.copy()
        modified[feat] = reference[feat]
        modified_score = float(pipeline.predict(modified)[0])
        contribution = round(base_score - modified_score, 1)
        actual_display = format_feature_value(feat, actual_value)
        typical_display = format_feature_value(feat, reference[feat])
        direction = "raising" if contribution >= 0 else "lowering"
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "contribution": contribution,
            "actual_value": actual_display,
            "typical_value": typical_display,
            "detail": (
                f"{actual_display} vs. a typical transfer's {typical_display}, "
                f"{direction} the score by {abs(contribution)} pts"
            ),
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


def build_transfer_card(r):
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "transfer_date": str(r["transfer_date"])[:10],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "success_score": float(r["success_score"]),
        "pre_ga_p90": round(float(r["pre_ga_p90"]), 2),
        "post_ga_p90": round(float(r["post_ga_p90"]), 2),
        "tenure_days": int(r["tenure_days"]),
        "still_at_club": bool(r["still_at_club"]),
        "breakdown": describe_components(r),
    }


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
        out.append(build_transfer_card(match.iloc[-1]))
    return out


@app.get("/api/transfers/detail")
def transfer_detail(player_id: int, transfer_date: str):
    match = transfers_df[
        (transfers_df["player_id"] == player_id) & (transfers_df["transfer_date"] == transfer_date)
    ]
    if match.empty:
        raise HTTPException(status_code=404, detail="transfer not found")
    return build_transfer_card(match.iloc[0])


@app.get("/api/players/search")
def search_players(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    mask = players_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = players_df[mask].head(limit).drop(columns=["_name_fold"])
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/search")
def search_clubs(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    mask = clubs_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = clubs_df[mask].head(limit).drop(columns=["_name_fold"])
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/{club_id}")
def get_club(club_id: int):
    row = clubs_df[clubs_df["club_id"] == club_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="club not found")
    return row.iloc[0].drop("_name_fold").fillna("").to_dict()


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
    comp_scores = [c["success_score"] for c in comps]
    score_range = [round(min(comp_scores), 1), round(max(comp_scores), 1)] if comp_scores else [score, score]
    return {
        "success_score": round(score, 1),
        "score_range": score_range,
        "comparable_transfers": comps,
        "explanation": explanation,
        "model_test_mae": metadata["test_mae"],
        "model_test_r2": metadata["test_r2"],
    }


class CompareRequest(BaseModel):
    a: PredictRequest
    b: PredictRequest
    label_a: str = "Option A"
    label_b: str = "Option B"


@app.post("/api/compare")
def compare(req: CompareRequest):
    result_a = predict(req.a)
    result_b = predict(req.b)
    return {
        "a": {**result_a, "label": req.label_a},
        "b": {**result_b, "label": req.label_b},
        "delta": round(result_a["success_score"] - result_b["success_score"], 1),
    }


TRANSFER_SORT_FIELDS = {
    "success_score", "transfer_date", "age_at_transfer", "transfer_fee", "tenure_days",
}


@app.get("/api/filters")
def get_filters():
    positions = sorted(transfers_df["position"].dropna().unique().tolist())
    league_ids = transfers_df["to_domestic_competition_id"].dropna().unique().tolist()
    leagues = sorted(
        ({"id": lid, "name": LEAGUE_NAMES.get(lid, lid)} for lid in league_ids),
        key=lambda x: x["name"],
    )
    return {"positions": positions, "leagues": leagues}


@app.get("/api/transfers")
def list_transfers(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "success_score",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
):
    df = transfers_df
    if position:
        df = df[df["position"] == position]
    if league:
        df = df[df["to_domestic_competition_id"] == league]
    if q:
        q_fold = fold_accents(q)
        mask = (
            df["_name_fold"].str.contains(q_fold, na=False)
            | df["_to_club_fold"].str.contains(q_fold, na=False)
            | df["_from_club_fold"].str.contains(q_fold, na=False)
        )
        df = df[mask]

    sort_field = sort if sort in TRANSFER_SORT_FIELDS else "success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = []
    for _, r in page.iterrows():
        fee = r["transfer_fee"]
        results.append({
            "player_id": int(r["player_id"]),
            "name": r["name"],
            "position": r["position"],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "to_league": LEAGUE_NAMES.get(r["to_domestic_competition_id"], r["to_domestic_competition_id"]),
            "transfer_date": str(r["transfer_date"])[:10],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "transfer_fee": None if pd.isna(fee) else float(fee),
            "tenure_days": int(r["tenure_days"]),
            "still_at_club": bool(r["still_at_club"]),
            "success_score": float(r["success_score"]),
        })
    return {"total": total, "limit": limit, "offset": offset, "results": results}


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
