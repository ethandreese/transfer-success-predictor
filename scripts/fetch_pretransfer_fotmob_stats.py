"""
Fetch FotMob stats for each transfer's *pre*-transfer year (at the OLD
club), the mirror image of scripts/fetch_fotmob_stats.py (which stitches
the tenure *after* the move). The predict model (scripts/train_model.py)
only ever saw pre_goals_p90/pre_ga_p90 (Transfermarkt goal contributions)
as pre-transfer performance signal, because it was built before the
FotMob pipeline existed - this fills that gap with the same rating/
attacking/defensive/possession stats the historical score already gets
for the *post*-transfer side.

Reuses fetch_fotmob_stats.py's season-fetch/cache, club-matching, and
season-stitching machinery unmodified - the only two differences are the
window (transfer_date - 365 days .. transfer_date, not transfer_date ..
tenure_end) and which club column drives the match (from_club_name /
from_domestic_competition_id, not to_*). No new scraping needed: this
reads the SAME data/raw/fotmob_season_cache/*.json files
fetch_fotmob_stats.py already populated (see load_or_fetch_season) -
99.7% of origin leagues fall within the same 23-league LEAGUE_MAP already
covered for destinations, confirmed empirically before writing this.

Optional, same as fetch_fotmob_stats.py: build_dataset.py degrades
gracefully (every transfer treated as missing pre-transfer FotMob data)
if data/raw/pretransfer_fotmob_stats_cache.csv doesn't exist. Safe to
interrupt and re-run - resumes from the same season cache.
"""
import os

import httpx
import pandas as pd

from fetch_fotmob_stats import (
    LEAGUE_MAP, aggregate_tenure_stats, find_fotmob_id, load_or_fetch_season,
    season_label_for_date, seasons_overlapping,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "pretransfer_fotmob_stats_cache.csv")
TRANSFERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")
PRE_WINDOW_DAYS = 365  # matches PRE_WINDOW_DAYS in build_dataset.py - same pre-transfer window the existing pre_apps/pre_goals_p90 stats already use


def load_candidate_transfers():
    """Permanent transfers only - loans aren't in scope for the predict model this feeds (see train_model.py)."""
    transfers = pd.read_csv(TRANSFERS_PATH, parse_dates=["transfer_date"])
    return transfers[transfers["transfer_date"] >= "2013-08-01"]


def run_league_pretransfer(client, comp_id, transfers):
    """
    Full fetch+stitch for one ORIGIN league: every season, every transfer's
    pre-transfer year aggregated. comp_id here is the league the player was
    LEAVING (from_domestic_competition_id), unlike fetch_fotmob_stats.py's
    run_league where comp_id is the destination. Mirrors that function
    closely; see its docstring for the season-stitching/contamination
    reasoning this reuses unchanged.
    """
    league_id, label = LEAGUE_MAP[comp_id]
    print(f"=== {comp_id} - {label} (fotmob id {league_id}) - {len(transfers):,} candidate transfers ===")

    season_tables = {season: load_or_fetch_season(client, comp_id, league_id, season) for season in _all_season_labels(transfers)}
    covered = [s for s, t in season_tables.items() if t is not None]
    print(f"  Seasons with coverage: {covered[0] if covered else 'NONE'} .. {covered[-1] if covered else 'NONE'} ({len(covered)} seasons)")

    results = []
    for _, t in transfers.iterrows():
        pre_start = t["transfer_date"] - pd.Timedelta(days=PRE_WINDOW_DAYS)
        pre_seasons = [s for s in seasons_overlapping(pre_start, t["transfer_date"]) if s in covered]
        if not pre_seasons:
            continue

        # Mirrors fetch_fotmob_stats.py's contamination check exactly (same
        # underlying condition - origin league == destination league,
        # mid-season move - just checked from the origin side): FotMob
        # attributes the WHOLE move season to the club the player is
        # registered at when fetched, so an intra-league mid-season move's
        # departure season would show (wrongly) at the NEW club if fetched
        # under the OLD club's leaderboard too - excluded either way.
        is_intra_league = t["to_domestic_competition_id"] == comp_id
        is_midseason_move = t["transfer_date"].month not in (6, 7, 8)
        move_season = season_label_for_date(t["transfer_date"])
        contaminated = move_season if (is_intra_league and is_midseason_move and move_season in pre_seasons) else None
        agg_seasons = [s for s in pre_seasons if s != contaminated]
        if not agg_seasons:
            continue

        # find_fotmob_id reads transfer_row["to_club_name"] - substitute in
        # the ORIGIN club so the existing matcher can be reused unmodified.
        lookup_row = t.copy()
        lookup_row["to_club_name"] = t["from_club_name"]
        fotmob_id = find_fotmob_id(lookup_row, season_tables, list(reversed(agg_seasons)), comp_id)
        if fotmob_id is None:
            continue
        agg = aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables)
        if agg is None:
            continue

        results.append({
            "player_id": t["player_id"], "transfer_date": t["transfer_date"],
            "from_club_name": t["from_club_name"], "origin_competition_id": comp_id,
            **{f"pre_{k}": v for k, v in agg.items()},
        })

    results_df = pd.DataFrame(results)
    print(f"  Matched {len(results_df)} / {len(transfers)} transfers")
    return results_df


def _all_season_labels(transfers):
    """Every season label any transfer in this batch's pre-window could touch, so load_or_fetch_season isn't called for seasons nothing here needs."""
    labels = set()
    for _, t in transfers.iterrows():
        pre_start = t["transfer_date"] - pd.Timedelta(days=PRE_WINDOW_DAYS)
        labels.update(seasons_overlapping(pre_start, t["transfer_date"]))
    return sorted(labels)


def main():
    """Fetch every league in LEAGUE_MAP as an ORIGIN league, stitch every permanent transfer's pre-transfer year, write to data/raw/pretransfer_fotmob_stats_cache.csv for build_dataset.py to consume."""
    df = load_candidate_transfers()

    all_results = []
    with httpx.Client(timeout=30) as client:
        for comp_id in LEAGUE_MAP:
            league_rows = df[df["from_domestic_competition_id"] == comp_id].copy()
            if league_rows.empty:
                continue
            results_df = run_league_pretransfer(client, comp_id, league_rows)
            if not results_df.empty:
                all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
