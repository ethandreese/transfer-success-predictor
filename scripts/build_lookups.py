"""
Build small (git-friendly) lookup tables the web app uses for player/club
search and autofill, derived from the large raw Transfermarkt CSVs that are
NOT committed to the repo.

Outputs:
  data/players_lookup.csv - one row per active-ish player with recent
    (trailing 365 day) performance, used to autofill the prediction form
  data/clubs_lookup.csv   - one row per club with a market-value proxy
"""
import os

import numpy as np
import pandas as pd

from build_dataset import compute_club_value_proxy, load_appearances
from fetch_fotmob_stats import LEAGUE_MAP

RAW_DIR = os.environ.get(
    "TRANSFERMARKT_RAW_DIR",
    os.path.expanduser(
        "~/.cache/kagglehub/datasets/davidcariboo/player-scores/versions/677"
    ),
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
WINDOW_DAYS = 365
REFERENCE_DATE = pd.Timestamp.today().normalize()
MIN_MARKET_VALUE = 1_000_000
CURRENT_FOTMOB_STATS_PATH = os.path.join(DATA_DIR, "raw", "current_fotmob_stats_cache.csv")
RECENT_FOTMOB_RAW_COLS = [
    "recent_fotmob_rating", "recent_fotmob_total_minutes",
    "recent_fotmob_expected_goals_per_90", "recent_fotmob_expected_assists_per_90",
    "recent_fotmob_won_contest", "recent_fotmob_total_att_assist",
    "recent_fotmob_total_tackle", "recent_fotmob_interception",
    "recent_fotmob_effective_clearance", "recent_fotmob_ball_recovery",
    "recent_fotmob_accurate_pass", "recent_fotmob_saves",
    "recent_fotmob__save_percentage", "recent_fotmob_goals_conceded",
]


def load_current_fotmob_stats():
    """
    Load each player's current-club FotMob snapshot built by
    scripts/fetch_current_fotmob_stats.py (optional, like the other two
    FotMob scripts) - the live-prediction-form equivalent of
    scripts/fetch_pretransfer_fotmob_stats.py's pre-transfer-year stats,
    autofilled from the searched player the same way recent_apps/
    recent_goals_p90/etc. already are (see app/main.py:build_feature_row
    for how a "recent_fotmob_*" column here becomes a "pre_fotmob_*"
    model feature). Degrades gracefully (empty frame) if the cache hasn't
    been built yet.
    """
    if not os.path.exists(CURRENT_FOTMOB_STATS_PATH):
        print(f"  (no current-FotMob stats cache at {CURRENT_FOTMOB_STATS_PATH} - run scripts/fetch_current_fotmob_stats.py to enable it; continuing without it)")
        return pd.DataFrame(columns=["player_id"] + RECENT_FOTMOB_RAW_COLS)
    return pd.read_csv(CURRENT_FOTMOB_STATS_PATH)


def build_clubs_lookup():
    """
    Write data/clubs_lookup.csv: every club with a positive squad-value
    proxy (sum of its current players' market values), used by the app's
    club search/autocomplete and to derive origin/destination club
    strength for predictions.
    """
    clubs = pd.read_csv(
        os.path.join(RAW_DIR, "clubs.csv"),
        usecols=["club_id", "name", "domestic_competition_id"],
    )
    players = pd.read_csv(
        os.path.join(RAW_DIR, "players.csv"),
        usecols=["current_club_id", "market_value_in_eur"],
    )
    # compute_club_value_proxy (build_dataset.py), not a second, independent
    # groupby-sum here - this app/main.py-facing proxy and the training-data
    # one must stay the same computation, not two hand-copies that happen to
    # agree today (same precedent as load_appearances() above).
    proxy = compute_club_value_proxy(players)
    clubs["club_value_proxy"] = clubs["club_id"].map(proxy).fillna(0)
    clubs = clubs[clubs["club_value_proxy"] > 0].sort_values(
        "club_value_proxy", ascending=False
    )
    out_path = os.path.join(DATA_DIR, "clubs_lookup.csv")
    clubs.to_csv(out_path, index=False)
    print(f"Wrote {len(clubs):,} clubs to {out_path}")


def build_players_lookup():
    """
    Write data/players_lookup.csv: every player worth >= MIN_MARKET_VALUE,
    with their appearances/goals/assists from the trailing WINDOW_DAYS (only
    counting games played for their *current* club) rolled up into
    recent_* per-90 stats. Used by the app's player search to auto-fill the
    "pre-transfer performance" fields of a hypothetical prediction.
    """
    players = pd.read_csv(
        os.path.join(RAW_DIR, "players.csv"),
        usecols=[
            "player_id", "name", "date_of_birth", "position", "sub_position",
            "foot", "height_in_cm", "current_club_id", "current_club_name",
            "market_value_in_eur", "current_club_domestic_competition_id",
        ],
        parse_dates=["date_of_birth"],
    )
    players = players[players["market_value_in_eur"].fillna(0) >= MIN_MARKET_VALUE]

    # load_appearances() (build_dataset.py), not a plain read_csv here - it
    # applies data/manual_appearance_corrections.csv (a handful of away
    # games misattributed to the home club in the raw dataset - see that
    # function's docstring), a fix this script used to skip entirely since
    # it re-reads appearances.csv independently rather than going through
    # build_dataset.py. Checked directly whether that gap is currently
    # live: re-ran the same detection scan (>=3 distinct clubs within any
    # 30-day window) against just the last WINDOW_DAYS instead of the full
    # historical range this file's own correction list was built from, and
    # found only Africa Cup of Nations call-ups (a real gap in club
    # appearances during international duty, not a misattributed one)
    # triggering it - no live instance today, but nothing was stopping a
    # future one from silently reaching recent_apps/recent_minutes here
    # uncorrected the way historical scoring already guards against.
    appearances = load_appearances()[["player_id", "player_club_id", "date", "goals", "assists", "minutes_played"]]
    window_start = REFERENCE_DATE - pd.Timedelta(days=WINDOW_DAYS)
    recent = appearances[
        (appearances["date"] >= window_start) & (appearances["date"] <= REFERENCE_DATE)
    ]
    recent = recent[recent["player_club_id"] == recent["player_id"].map(
        players.set_index("player_id")["current_club_id"]
    )]
    stats = recent.groupby("player_id").agg(
        recent_apps=("date", "count"),
        recent_minutes=("minutes_played", "sum"),
        recent_goals=("goals", "sum"),
        recent_assists=("assists", "sum"),
    )
    stats["recent_ga_p90"] = (
        (stats["recent_goals"] + stats["recent_assists"]) / stats["recent_minutes"].clip(lower=1) * 90
    )
    stats["recent_goals_p90"] = stats["recent_goals"] / stats["recent_minutes"].clip(lower=1) * 90
    stats["recent_mins_per_app"] = stats["recent_minutes"] / stats["recent_apps"].clip(lower=1)

    players = players.merge(stats, on="player_id", how="left")
    players["age_now"] = (REFERENCE_DATE - players["date_of_birth"]).dt.days / 365.25

    # A player currently at a club outside LEAGUE_MAP (e.g. Messi/Inter Miami,
    # Son/LAFC - both MLS, not one of the 23 tracked leagues) has no reliable
    # "recent" signal here at all, but naively summing whatever appearance
    # rows happen to exist can still produce a real-looking number: both
    # clubs show up in appearances.csv with a handful of rows from the 2025
    # Club World Cup, a one-off international tournament, not real ongoing
    # league coverage - and outside that fluke, the more common case is
    # simply zero matching rows, which .fillna(0) below would otherwise
    # silently equate with "played 0 minutes recently" (a real, meaningful
    # signal for a player at a *covered* club - e.g. returning from a long
    # injury) rather than "we don't have real data for this club at all".
    # Forcing every recent_* column to NaN for an uncovered league - even
    # when the naive aggregation above found a nonzero value - keeps that
    # distinction: build_feature_row/explain_prediction in app/main.py
    # median-impute a real NaN and say so in the explanation, instead of
    # quietly feeding the model a fabricated hard 0 (or a stray Club World
    # Cup cameo) as if it were this player's real current form.
    covered_league = players["current_club_domestic_competition_id"].isin(LEAGUE_MAP)
    for c in ["recent_apps", "recent_minutes", "recent_goals", "recent_assists",
              "recent_ga_p90", "recent_goals_p90", "recent_mins_per_app"]:
        players[c] = np.where(covered_league, players[c].fillna(0), np.nan)

    players = players.merge(load_current_fotmob_stats(), on="player_id", how="left")
    recent_minutes_per_90 = (players["recent_fotmob_total_minutes"] / 90).clip(lower=1)
    players["recent_fotmob_chances_created_p90"] = players["recent_fotmob_total_att_assist"] / recent_minutes_per_90

    players = players.sort_values("market_value_in_eur", ascending=False)
    out_path = os.path.join(DATA_DIR, "players_lookup.csv")
    players.to_csv(out_path, index=False)
    print(f"Wrote {len(players):,} players to {out_path}")


def main():
    """Build both lookup tables. Run after scripts/build_dataset.py whenever the raw dataset changes."""
    build_clubs_lookup()
    build_players_lookup()


if __name__ == "__main__":
    main()
