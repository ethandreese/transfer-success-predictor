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

import pandas as pd

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


def build_clubs_lookup():
    clubs = pd.read_csv(
        os.path.join(RAW_DIR, "clubs.csv"),
        usecols=["club_id", "name", "domestic_competition_id"],
    )
    players = pd.read_csv(
        os.path.join(RAW_DIR, "players.csv"),
        usecols=["current_club_id", "market_value_in_eur"],
    ).dropna(subset=["current_club_id"])
    proxy = (
        players.groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .rename("club_value_proxy")
    )
    proxy.index = proxy.index.astype(int)
    clubs["club_value_proxy"] = clubs["club_id"].map(proxy).fillna(0)
    clubs = clubs[clubs["club_value_proxy"] > 0].sort_values(
        "club_value_proxy", ascending=False
    )
    out_path = os.path.join(DATA_DIR, "clubs_lookup.csv")
    clubs.to_csv(out_path, index=False)
    print(f"Wrote {len(clubs):,} clubs to {out_path}")


def build_players_lookup():
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

    appearances = pd.read_csv(
        os.path.join(RAW_DIR, "appearances.csv"),
        usecols=["player_id", "player_club_id", "date", "goals", "assists", "minutes_played"],
        parse_dates=["date"],
    )
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

    for c in ["recent_apps", "recent_minutes", "recent_goals", "recent_assists",
              "recent_ga_p90", "recent_goals_p90", "recent_mins_per_app"]:
        players[c] = players[c].fillna(0)

    players = players.sort_values("market_value_in_eur", ascending=False)
    out_path = os.path.join(DATA_DIR, "players_lookup.csv")
    players.to_csv(out_path, index=False)
    print(f"Wrote {len(players):,} players to {out_path}")


def main():
    build_clubs_lookup()
    build_players_lookup()


if __name__ == "__main__":
    main()
