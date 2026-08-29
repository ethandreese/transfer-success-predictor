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
loans_df = pd.read_csv(os.path.join(DATA_DIR, "loans_processed.csv"))
players_df = pd.read_csv(os.path.join(DATA_DIR, "players_lookup.csv"))
clubs_df = pd.read_csv(os.path.join(DATA_DIR, "clubs_lookup.csv"))

players_df["_name_fold"] = players_df["name"].map(fold_accents)
clubs_df["_name_fold"] = clubs_df["name"].map(fold_accents)
transfers_df["_name_fold"] = transfers_df["name"].map(fold_accents)
transfers_df["_from_club_fold"] = transfers_df["from_club_name"].map(fold_accents)
transfers_df["_to_club_fold"] = transfers_df["to_club_name"].map(fold_accents)
loans_df["_name_fold"] = loans_df["name"].map(fold_accents)
loans_df["_from_club_fold"] = loans_df["from_club_name"].map(fold_accents)
loans_df["_to_club_fold"] = loans_df["to_club_name"].map(fold_accents)
competitions_df = pd.read_csv(os.path.join(DATA_DIR, "competitions_lookup.csv"))
_dup_names = competitions_df["name"][competitions_df["name"].duplicated(keep=False)]
competitions_df["display_name"] = competitions_df.apply(
    lambda r: f"{r['name']} ({r['country_name']})" if r["name"] in _dup_names.values and pd.notna(r["country_name"]) else r["name"],
    axis=1,
)
LEAGUE_NAMES = dict(zip(competitions_df["competition_id"], competitions_df["display_name"]))

league_baselines_df = pd.read_csv(os.path.join(DATA_DIR, "league_baselines.csv"))
LEAGUE_POSITION_BASELINE = {
    (r["competition_id"], r["position"]): r["ga_p90_baseline"] for _, r in league_baselines_df.iterrows()
}


def league_display_name(competition_id):
    """
    Look up a league's display name, or "Unknown league" for the rare club
    that isn't in clubs.csv at all (~7 in transfers_processed.csv, ~23 in
    the smaller loans_processed.csv - obscure clubs the dataset never
    populated a domestic_competition_id for), where competition_id itself
    is NaN. LEAGUE_NAMES.get(competition_id, competition_id) alone would
    return that same NaN back out (a float NaN never equals itself, so the
    dict lookup always misses), which isn't JSON-serializable and 500s any
    endpoint that returns it.
    """
    if pd.isna(competition_id):
        return "Unknown league"
    return LEAGUE_NAMES.get(competition_id, competition_id)


def league_ga_baseline(competition_id, position):
    """Goal contributions/90 baseline for this (league, position), falling back to the position's overall average."""
    return LEAGUE_POSITION_BASELINE.get(
        (competition_id, position),
        LEAGUE_POSITION_BASELINE.get(("_default", position), 0.3),
    )


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
    """Plain-English verdict on a resale: profit, loss, or break-even, given what the club paid vs. what it later sold the player for."""
    fee_paid = 0 if pd.isna(fee_paid) else fee_paid
    if fee_received > fee_paid:
        return "a profitable flip for the club, regardless of on-pitch performance"
    if fee_received < fee_paid:
        return "sold for less than the club paid, a loss independent of on-pitch performance"
    return "resold for the same fee paid, breaking even"


def describe_resale_profit(r, eur_m):
    """Build the 'Resale profit' breakdown row's description: the fee paid/received, the outcome (profit/loss), and how much it counts given the tenure-scaled weight (see compute_resale_weight in build_dataset.py)."""
    tenure_years = r["tenure_days"] / 365.25
    weight_pct = r["resale_weight"] * 100
    outcome = resale_outcome_phrase(r["transfer_fee"], r["next_transfer_fee"])
    weight_note = (
        f"counts for only {weight_pct:.0f}% of the score here — after a {tenure_years:.1f}-year tenure "
        f"the club already got most of its value from having them play, regardless of the sale price"
        if tenure_years > 3
        else f"counts for {weight_pct:.0f}% of the score here, weighted higher since it was a short "
        f"({tenure_years:.1f}-year) tenure"
    )
    return (
        f"Bought for {eur_m(r['transfer_fee'])}, later resold for {eur_m(r['next_transfer_fee'])} "
        f"— {outcome}. This {weight_note}."
    )


FOTMOB_COMPONENT_LABELS = {
    "rating": "FotMob rating",
    "attacking": "Attacking output",
    "defensive": "Defensive contribution",
    "possession": "Possession & passing",
}


def describe_fotmob_component(component, r, position_plural, is_loan=False):
    """
    Build one FotMob-derived breakdown row's description (rating/attacking/
    defensive/possession - see compute_fotmob_component_pcts in
    build_dataset.py for why they're kept separate rather than blended into
    one number). The underlying per-90 rates driving "attacking" aren't
    stored directly (only the raw season totals are), so chances-created/90
    is recomputed here the same way build_dataset.py derives it - from
    fotmob_total_att_assist and fotmob_total_minutes. Shared between
    describe_components() (permanent transfers) and
    describe_loan_components() (loans) - is_loan only changes the wording
    ("on loan at"/"other loan spells" vs. "at"/"other {position}"), purely
    cosmetic: unlike every other component, rating/attacking/defensive/
    possession are ranked against transfers and loans *combined* (see
    attach_fotmob_components in build_dataset.py), so the number itself
    means the same thing on both kinds of card - only the phrasing differs.

    Each bucket's _pct is an average of whichever of its underlying stats
    are actually available (see compute_fotmob_component_pcts), so a
    component can be "known" even when one specific stat behind it isn't -
    e.g. attacking_pct valid from goals/chances-created alone with xG/xA
    missing that season. parts() builds the description from only the
    sub-stats that are actually present, instead of formatting a NaN
    straight into the string ("nan xG/90").
    """
    seasons = int(r["fotmob_seasons_used"])
    season_note = "1 season" if seasons == 1 else f"{seasons} seasons"
    minutes_per_90 = max(r["fotmob_total_minutes"] / 90, 1)
    chances_created_p90 = r["fotmob_total_att_assist"] / minutes_per_90

    def parts(*pairs):
        """pairs is (value, format-string) tuples - drop any whose value is NaN, then join what's left."""
        return ", ".join(fmt.format(v) for v, fmt in pairs if pd.notna(v))

    if component == "rating":
        detail = parts((r["fotmob_rating"], "{:.2f} average match rating"))
    elif component == "attacking":
        detail = parts(
            (r["fotmob_goals_per_90"], "{:.2f} goals/90"),
            (r["fotmob_expected_goals_per_90"], "{:.2f} xG/90"),
            (r["fotmob_expected_assists_per_90"], "{:.2f} xA/90"),
            (chances_created_p90, "{:.1f} chances created/90"),
        )
    elif component == "defensive" and r["position"] == "Goalkeeper":
        detail = parts(
            (r["fotmob_saves"], "{:.1f} saves/90"),
            (r["fotmob__save_percentage"], "{:.0f}% save rate"),
            (r["fotmob_goals_conceded"], "{:.1f} goals conceded/90"),
        )
    elif component == "defensive":
        detail = parts(
            (r["fotmob_total_tackle"], "{:.1f} tackles/90"),
            (r["fotmob_interception"], "{:.1f} interceptions/90"),
            (r["fotmob_effective_clearance"], "{:.1f} clearances/90"),
            (r["fotmob_ball_recovery"], "{:.1f} recoveries/90"),
        )
    else:  # possession
        detail = parts(
            (r["fotmob_accurate_pass"], "{:.1f} accurate passes/90"),
            (r["fotmob_won_contest"], "{:.1f} dribbles/90"),
        )

    # "vs. other {position}" here (not "other loan spells"/"other transfers"
    # like the rest of each card's rows) because these four components are
    # ranked against transfers and loans combined - see
    # attach_fotmob_components in build_dataset.py.
    at_club = f"on loan at {r['to_club_name']}" if is_loan else f"at {r['to_club_name']}"
    return f"{detail} — averaged across {season_note} {at_club}, ranked vs. other {position_plural}"


def eur_m(v):
    """Format a euro amount for display, e.g. 50_000_000 -> "€50m", 300_000 -> "€0.3m", 0/NaN -> "free"."""
    if pd.isna(v) or v == 0:
        return "free"
    millions = v / 1_000_000
    # Sub-million fees are common (e.g. a €300k sale) - one decimal place
    # keeps them from rounding down to a misleading "€0m".
    return f"€{millions:.1f}m" if millions < 1 else f"€{millions:.0f}m"


def describe_components(r):
    """
    Build the full "why this score" breakdown for one row of
    transfers_processed.csv: a list of {label, value, description} dicts,
    one per success-score component actually used for this transfer (5
    always, plus up to 4 FotMob-derived rows - rating/attacking/defensive/
    possession, each independently shown only when its own has_*_data flag
    is true - and "Resale profit" when has_resale_data is true - so 5 to 10
    rows total). Used by both /api/examples and /api/transfers/detail via
    build_transfer_card().
    """
    position_plural = POSITION_PLURAL.get(r["position"], r["position"])
    to_league = league_display_name(r["to_domestic_competition_id"])
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
            "description": (
                f"{eur_m(r['value_before'])} → peaked at {eur_m(r['value_peak'])} (now {eur_m(r['value_after'])})"
                if r["value_peak"] > r["value_after"] * 1.05
                else f"{eur_m(r['value_before'])} → {eur_m(r['value_after'])} market value"
            ),
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
    ] + [
        {
            "label": FOTMOB_COMPONENT_LABELS[component],
            "value": round(float(r[f"{component}_pct"]), 1),
            "description": describe_fotmob_component(component, r, position_plural),
        }
        for component in ("rating", "attacking", "defensive", "possession")
        if bool(r[f"has_{component}_data"])
    ] + ([
        {
            "label": "Resale profit",
            "value": round(float(r["resale_profit_pct"]), 1),
            "description": describe_resale_profit(r, eur_m),
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


def format_feature_value(feat, value, context=None):
    """
    Render one model feature's raw value in human-readable form for the
    prediction explanation (e.g. a log-transformed fee back to "€50.0m", a
    league code to its display name). `context` (the request's position and
    origin league) is only used for pre_ga_p90, to append a league-relative
    "(X.Xx their current league's average)" note - see league_ga_baseline.
    """
    if feat in LOG_FEATURES:
        value = np.expm1(value)
        if feat == "log_transfer_fee" and value == 0:
            return "free"
        return f"€{value / 1_000_000:.1f}m"
    if feat == "age_at_transfer":
        return f"{value:.1f} yrs"
    if feat == "height_in_cm":
        return f"{value:.0f} cm"
    if feat == "pre_ga_p90":
        base = f"{value:.2f} per 90"
        if context:
            baseline = league_ga_baseline(context.get("from_domestic_competition_id"), context.get("position"))
            if baseline > 0:
                base += f" ({value / baseline:.1f}x their current league's average)"
        return base
    if feat == "pre_goals_p90":
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
    """
    A hypothetical transfer to score: a player's pre-transfer profile
    (age/position/recent performance/market value) plus the destination
    club and fee. Mirrors NUMERIC_FEATURES/CATEGORICAL_FEATURES in
    train_model.py after the derived columns are added by build_feature_row.
    """
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
    """Turn a PredictRequest into the single-row DataFrame the model pipeline expects, computing the log/ratio features it was trained on."""
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
    position = feature_row["position"].iloc[0]
    context = {
        "position": position,
        "from_domestic_competition_id": feature_row["from_domestic_competition_id"].iloc[0],
    }

    # log_transfer_fee and fee_to_value_ratio are both 0 for free transfers
    # (out-of-contract moves, academy graduates), which are >50% of the
    # dataset - so their overall "typical" reference is 0, and comparing an
    # actual fee against "a typical €0m" is misleading. When the transfer
    # being explained itself has a real fee, compare it against the
    # typical *paid* transfer instead; a free transfer still compares
    # against the overall reference, which correctly reflects that being
    # free is itself common.
    is_paid_transfer = feature_row["log_transfer_fee"].iloc[0] > 0
    paid_reference = metadata["reference_values_paid"]

    # A striker's typical goal contributions, or a goalkeeper's typical
    # height, look nothing like the whole population's - compare these
    # against the same-position median instead of a flat one (see
    # reference_values_by_position in train_model.py).
    position_conditional = set(metadata["position_conditional_features"])
    position_reference = metadata["reference_values_by_position"].get(position, {})

    # A €100m fee for a player already valued at €70m isn't remarkable -
    # a flat "typical paid fee" (~€6m) makes any big-money move for an
    # already-valuable player look like a wild outlier. Compare the actual
    # fee against what's typically paid for a player valued this highly
    # instead (fee_regression: log(fee) ~ log(value), fit on paid
    # transfers - see train_model.py), using *this* transfer's own
    # value_before.
    fee_reg = metadata["fee_regression"]
    log_value_before = feature_row["log_value_before"].iloc[0]
    expected_log_fee = fee_reg["intercept"] + fee_reg["slope"] * log_value_before

    contributions = []
    for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        typical_label = "a typical transfer's"
        if feat == "log_transfer_fee" and is_paid_transfer:
            reference_value = expected_log_fee
            typical_label = "what's typically paid for a similarly-valued player:"
        elif feat in position_conditional and feat in position_reference:
            reference_value = position_reference[feat]
            typical_label = f"a typical {POSITION_PLURAL.get(position, position).rstrip('s')}'s"
        elif is_paid_transfer and feat in paid_reference:
            reference_value = paid_reference[feat]
            typical_label = "a typical paid transfer's"
        else:
            reference_value = reference[feat]

        actual_value = feature_row[feat].iloc[0]
        modified = feature_row.copy()
        modified[feat] = reference_value
        modified_score = float(pipeline.predict(modified)[0])
        contribution = round(base_score - modified_score, 1)
        actual_display = format_feature_value(feat, actual_value, context)
        typical_display = format_feature_value(feat, reference_value, context)
        direction = "raising" if contribution >= 0 else "lowering"
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "contribution": contribution,
            "actual_value": actual_display,
            "typical_value": typical_display,
            "detail": (
                f"{actual_display} vs. {typical_label} {typical_display}, "
                f"{direction} the score by {abs(contribution)} pts"
            ),
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_k]


def find_comparables(feature_row: pd.DataFrame, k: int = 5):
    """
    Look up the k most similar historical transfers to `feature_row` using
    the nearest-neighbors index built in train_model.py (Euclidean distance
    over the scaled numeric features). Used both to show "most similar
    historical transfers" and, via their success_score spread, as the
    predicted score_range in /api/predict.
    """
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
    """
    Build the JSON shape shared by /api/examples and /api/transfers/detail
    for one row of transfers_processed.csv: identity/route, the score, and
    the full describe_components() breakdown.
    """
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


def describe_loan_components(r):
    """
    Build the "why this score" breakdown for one row of loans_processed.csv:
    4 components always, plus up to 4 FotMob-derived rows (rating/
    attacking/defensive/possession, each independently shown only when its
    own has_*_data flag is true - so 4 to 8 rows total). Unlike
    describe_components(), there's no "resale profit" row - a loan doesn't
    end in a sale of its own - and no "value for money" row - most loans
    carry no real fee, see data/loan_score_weights.json.
    """
    position_plural = POSITION_PLURAL.get(r["position"], r["position"])
    to_league = league_display_name(r["to_domestic_competition_id"])
    return [
        {
            "label": "Performance level",
            "value": round(float(r["perf_level_pct"]), 1),
            "description": (
                f"{r['post_ga_p90']:.2f} goal contributions/90 while on loan at {r['to_club_name']} "
                f"({r['post_ga_p90_vs_league']:.1f}x the {to_league} average for {position_plural}), "
                f"ranked vs. other loan spells"
            ),
        },
        {
            "label": "Performance change",
            "value": round(float(r["perf_delta_pct"]), 1),
            "description": (
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x on loan — beat the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high (some pullback from a peak is normal)"
                if r["post_ga_p90_vs_league"] >= r["expected_post_ga_p90_vs_league"] else
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x on loan — below the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high"
            ),
        },
        {
            "label": "Market value growth",
            "value": round(float(r["value_growth_pct"]), 1),
            "description": (
                f"{eur_m(r['value_before'])} → peaked at {eur_m(r['value_peak'])} (now {eur_m(r['value_after'])}) during the loan"
                if r["value_peak"] > r["value_after"] * 1.05
                else f"{eur_m(r['value_before'])} → {eur_m(r['value_after'])} market value during the loan"
            ),
        },
        {
            "label": "Playing time",
            "value": round(float(r["playing_time_pct"]), 1),
            "description": (
                f"{int(r['post_apps'])} of {int(r['team_games_in_tenure'])} games "
                f"{r['to_club_name']} played during the loan "
                f"({r['pct_team_games_played'] * 100:.0f}% - usually the central question a loan gets "
                f"judged on, blended with raw appearance count)"
            ),
        },
    ] + [
        {
            "label": FOTMOB_COMPONENT_LABELS[component],
            "value": round(float(r[f"{component}_pct"]), 1),
            "description": describe_fotmob_component(component, r, position_plural, is_loan=True),
        }
        for component in ("rating", "attacking", "defensive", "possession")
        if bool(r[f"has_{component}_data"])
    ]


def build_loan_card(r):
    """
    Build the JSON shape shared by /api/loans and /api/loans/detail for one
    row of loans_processed.csv: identity/route, the score, and the full
    describe_loan_components() breakdown.
    """
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "transfer_date": str(r["transfer_date"])[:10],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "loan_success_score": float(r["loan_success_score"]),
        "pre_ga_p90": round(float(r["pre_ga_p90"]), 2),
        "post_ga_p90": round(float(r["post_ga_p90"]), 2),
        "tenure_days": int(r["tenure_days"]),
        "still_on_loan": bool(r["still_on_loan"]),
        "breakdown": describe_loan_components(r),
    }


@app.get("/api/health")
def health():
    """Liveness check plus a dump of the deployed model's training metadata (feature lists, test metrics)."""
    return {"status": "ok", "model_metadata": metadata}


@app.get("/api/examples")
def examples():
    """Return the curated homepage cards (EXAMPLE_TRANSFER_KEYS) as full transfer cards."""
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
    """Look up one specific historical transfer by (player_id, transfer_date) and return its full card - used by the browse page's click-to-view modal."""
    match = transfers_df[
        (transfers_df["player_id"] == player_id) & (transfers_df["transfer_date"] == transfer_date)
    ]
    if match.empty:
        raise HTTPException(status_code=404, detail="transfer not found")
    return build_transfer_card(match.iloc[0])


@app.get("/api/players/search")
def search_players(q: str, limit: int = 10):
    """Accent-insensitive substring search over players_lookup.csv, for the prediction form's player autocomplete."""
    if len(q) < 2:
        return []
    mask = players_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = players_df[mask].head(limit).drop(columns=["_name_fold"])
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/search")
def search_clubs(q: str, limit: int = 10):
    """Accent-insensitive substring search over clubs_lookup.csv, for the destination-club autocomplete."""
    if len(q) < 2:
        return []
    mask = clubs_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = clubs_df[mask].head(limit).drop(columns=["_name_fold"])
    return rows.fillna("").to_dict(orient="records")


@app.get("/api/clubs/{club_id}")
def get_club(club_id: int):
    """Look up one club by id - used to fetch a selected player's *current* club details (for the "origin club" side of a prediction)."""
    row = clubs_df[clubs_df["club_id"] == club_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="club not found")
    return row.iloc[0].drop("_name_fold").fillna("").to_dict()


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Score a hypothetical transfer: run the model, clip to [0, 100], and
    attach a likely score_range (min/max among the nearest comparable
    historical transfers - a single point estimate would overstate how
    confident a R^2~0.10 model can be), the top-5 feature explanation, and
    the comparables themselves.
    """
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
    """Two hypothetical transfers to score side by side, with display labels for the compare page."""
    a: PredictRequest
    b: PredictRequest
    label_a: str = "Option A"
    label_b: str = "Option B"


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Score both scenarios via predict() and return them together with the point gap between them, for the compare page."""
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
    """List the distinct positions and destination leagues present in transfers_processed.csv, for the browse page's filter dropdowns."""
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
    """Paginated, filterable, sortable listing of every scored transfer, for the browse page's table."""
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
            "to_league": league_display_name(r["to_domestic_competition_id"]),
            "transfer_date": str(r["transfer_date"])[:10],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "transfer_fee": None if pd.isna(fee) else float(fee),
            "tenure_days": int(r["tenure_days"]),
            "still_at_club": bool(r["still_at_club"]),
            "success_score": float(r["success_score"]),
        })
    return {"total": total, "limit": limit, "offset": offset, "results": results}


LOAN_SORT_FIELDS = {"loan_success_score", "transfer_date", "age_at_transfer", "tenure_days"}


@app.get("/api/loans/filters")
def get_loan_filters():
    """List the distinct positions and loan-destination leagues present in loans_processed.csv, for the loans page's filter dropdowns."""
    positions = sorted(loans_df["position"].dropna().unique().tolist())
    league_ids = loans_df["to_domestic_competition_id"].dropna().unique().tolist()
    leagues = sorted(
        ({"id": lid, "name": LEAGUE_NAMES.get(lid, lid)} for lid in league_ids),
        key=lambda x: x["name"],
    )
    return {"positions": positions, "leagues": leagues}


@app.get("/api/loans")
def list_loans(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "loan_success_score",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
):
    """Paginated, filterable, sortable listing of every scored loan spell, for the loans page's table."""
    df = loans_df
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

    sort_field = sort if sort in LOAN_SORT_FIELDS else "loan_success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = []
    for _, r in page.iterrows():
        results.append({
            "player_id": int(r["player_id"]),
            "name": r["name"],
            "position": r["position"],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "to_league": league_display_name(r["to_domestic_competition_id"]),
            "transfer_date": str(r["transfer_date"])[:10],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "tenure_days": int(r["tenure_days"]),
            "still_on_loan": bool(r["still_on_loan"]),
            "loan_success_score": float(r["loan_success_score"]),
        })
    return {"total": total, "limit": limit, "offset": offset, "results": results}


@app.get("/api/loans/detail")
def loan_detail(player_id: int, transfer_date: str):
    """Look up one specific loan spell by (player_id, transfer_date) and return its full card - used by the loans page's click-to-view modal."""
    match = loans_df[
        (loans_df["player_id"] == player_id) & (loans_df["transfer_date"] == transfer_date)
    ]
    if match.empty:
        raise HTTPException(status_code=404, detail="loan not found")
    return build_loan_card(match.iloc[0])


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
