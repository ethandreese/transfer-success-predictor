"""
Fetch each player's *current* FotMob snapshot (trailing 365 days at their
current club) for data/players_lookup.csv - the live-prediction-form
equivalent of scripts/fetch_pretransfer_fotmob_stats.py, which does the
same thing for a specific past transfer's pre-transfer year instead of
"right now". A hypothetical prediction has no transfer_date to compute a
historical pre-transfer window from - it needs a snapshot of the searched
player's recent form at whatever club they're at today, autofilled into
the predict form the same way recent_apps/recent_goals_p90/etc. already
are (see scripts/build_lookups.py).

Reuses the same season-fetch/cache and matching machinery as the other
two FotMob scripts. Unlike those, this is keyed by player + current club,
not by a specific transfer event - and there's no mid-season-move
contamination to worry about, since FotMob already attributes a season to
whichever club a player is CURRENTLY registered at, which is exactly the
club this script wants anyway.

Optional, same pattern as the other two: build_lookups.py degrades
gracefully (every player treated as missing recent-FotMob data) if
data/raw/current_fotmob_stats_cache.csv doesn't exist.
"""
import os

import httpx
import pandas as pd

from fetch_fotmob_stats import (
    LEAGUE_MAP, aggregate_tenure_stats, find_fotmob_id, load_or_fetch_season, seasons_overlapping,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "current_fotmob_stats_cache.csv")
PLAYERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "players_lookup.csv")
WINDOW_DAYS = 365  # matches scripts/build_lookups.py's WINDOW_DAYS for recent_* stats


def run_league_current(client, comp_id, players, window_start, today):
    """Every player currently at a club in this league, matched and aggregated over the trailing-year seasons. Mirrors fetch_pretransfer_fotmob_stats.py's run_league_pretransfer, minus the contamination check (not applicable - see module docstring)."""
    league_id, label = LEAGUE_MAP[comp_id]
    seasons = [s for s in seasons_overlapping(window_start, today)]
    print(f"=== {comp_id} - {label} (fotmob id {league_id}) - {len(players):,} candidate players ===")

    season_tables = {season: load_or_fetch_season(client, comp_id, league_id, season) for season in seasons}
    covered = [s for s in seasons if season_tables.get(s) is not None]
    if not covered:
        print("  No season coverage for this window yet - skipping")
        return pd.DataFrame()

    results = []
    for _, p in players.iterrows():
        lookup_row = p.copy()
        lookup_row["to_club_name"] = p["current_club_name"]
        fotmob_id = find_fotmob_id(lookup_row, season_tables, list(reversed(covered)), comp_id)
        if fotmob_id is None:
            continue
        agg = aggregate_tenure_stats(fotmob_id, covered, season_tables)
        if agg is None:
            continue
        results.append({"player_id": p["player_id"], **{f"recent_{k}": v for k, v in agg.items()}})

    results_df = pd.DataFrame(results)
    print(f"  Matched {len(results_df)} / {len(players)} players")
    return results_df


def main():
    """Fetch every league in LEAGUE_MAP, match+aggregate every current player's trailing-year FotMob snapshot, write to data/raw/current_fotmob_stats_cache.csv for build_lookups.py to consume."""
    today = pd.Timestamp.today().normalize()
    window_start = today - pd.Timedelta(days=WINDOW_DAYS)
    players = pd.read_csv(PLAYERS_PATH)

    all_results = []
    with httpx.Client(timeout=30) as client:
        for comp_id in LEAGUE_MAP:
            league_players = players[players["current_club_domestic_competition_id"] == comp_id].copy()
            if league_players.empty:
                continue
            results_df = run_league_current(client, comp_id, league_players, window_start, today)
            if not results_df.empty:
                all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
