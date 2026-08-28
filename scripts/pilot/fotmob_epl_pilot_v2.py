"""
Extends fotmob_epl_pilot.py from a single season to the EPL's full history
of FotMob coverage (2016/2017 onward - nothing exists before that, see
README pilot notes) and works out "season stitching": our transfer score
is computed over a player's whole tenure (join date -> departure), but
FotMob only exposes whole-*season* leaderboards, so a multi-season tenure
needs several seasons' numbers combined, and a mid-season tenure start
needs care about which season(s) actually reflect time at the new club.

Two stitching problems, found by probing real data before writing this:

1. Multi-season tenures. Mechanical - fetch every season the tenure
   overlaps and combine: sum count-type stats (goals, clean sheets),
   minutes-weight-average rate-type stats (rating, tackles per 90, etc.).

2. Same-season, same-league moves. FotMob's season leaderboard attributes
   a player's ENTIRE season total to whichever club they're registered at
   when the page is fetched - not split by stint. Confirmed empirically:
   Marc Guehi (Crystal Palace -> Man City, 19 Jan 2026) shows 35
   matches/3150 minutes under "Manchester City" for 2025/2026, which is
   almost a full season - his Palace appearances before the move are
   folded in. This only contaminates a transfer where BOTH clubs are in
   the Premier League and the move happened mid-season (not the Jun-Aug
   window) - a cross-league arrival's "before" minutes live in a
   different league's leaderboard entirely, so they can't leak in. Of
   633 EPL-destination transfers since 2016/17, 209 are Premier-League-
   to-Premier-League moves and 51 of those are mid-season - that
   contaminated first season is excluded from the aggregate below (using
   only the season(s) after it), and if that's the ONLY season the tenure
   touches, the transfer is reported as identified-but-no-usable-data
   rather than silently scored on mixed-club numbers.

Player identity (which FotMob id a Transfermarkt row corresponds to) is
still resolved via fotmob_epl_pilot's name+club matching, but only once
per transfer (checking seasons newest-first) rather than once per season -
once we know the id, later/earlier seasons are looked up directly by id,
which is both cheaper and avoids re-matching on a contaminated season's
misleading club/name pairing.

Not wired into the scoring formula - reports coverage only.
"""
import difflib
import json
import os
import sys

import httpx
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fotmob_epl_pilot import (  # noqa: E402
    HEADERS, EPL_LEAGUE_ID, STAT_CATEGORIES, normalize_name, club_names_match,
    fetch_league_stat_categories, fetch_stat_list,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MATCHED_OUT_PATH = os.path.join(os.path.dirname(__file__), "fotmob_epl_full_history_matched.csv")
REPORT_OUT_PATH = os.path.join(os.path.dirname(__file__), "fotmob_epl_full_history_report.txt")
TRANSFERS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transfers_processed.csv")

# Header text (captured from the live API) tells us whether a stat is a
# per-90/percentage rate (minutes-weighted average across seasons) or a
# season-total count (summed across seasons) - see module docstring point 1.
STAT_HEADERS = {
    "rating": "FotMob rating", "mins_played": "Minutes played",
    "goals_per_90": "Goals per 90", "expected_goals_per_90": "Expected goals (xG) per 90",
    "expected_assists_per_90": "Expected assist (xA) per 90", "accurate_pass": "Accurate passes per 90",
    "total_att_assist": "Chances created", "big_chance_created": "Big chances created",
    "won_contest": "Successful dribbles per 90", "total_tackle": "Tackles per 90",
    "interception": "Interceptions per 90", "effective_clearance": "Clearances per 90",
    "defensive_contributions": "Defensive actions per 90", "ball_recovery": "Recoveries per 90",
    "clean_sheet": "Clean sheets", "saves": "Saves per 90", "_save_percentage": "Save percentage",
    "goals_conceded": "Goals conceded per 90", "fouls": "Fouls committed per 90",
}
RATE_STATS = {name for name, header in STAT_HEADERS.items() if "per 90" in header or "percentage" in header or name == "rating"}
SUM_STATS = set(STAT_HEADERS) - RATE_STATS

FIRST_COVERED_SEASON_YEAR = 2016  # "2016/2017" - nothing before this for EPL, see docstring
CURRENT_SEASON_YEAR = 2026        # "2026/2027" - today's date is 2026-08-28
ALL_SEASON_LABELS = [f"{y}/{y + 1}" for y in range(FIRST_COVERED_SEASON_YEAR, CURRENT_SEASON_YEAR + 1)]


def season_label_for_date(date):
    """Map a calendar date to its EPL season label, using Aug 1 as the season-start cutoff."""
    return f"{date.year}/{date.year + 1}" if date.month >= 8 else f"{date.year - 1}/{date.year}"


def seasons_overlapping(transfer_date, tenure_end):
    """All season labels a [transfer_date, tenure_end] tenure window touches, clipped to today's season."""
    today = pd.Timestamp.today().normalize()
    end = min(tenure_end, today)
    start_label = season_label_for_date(transfer_date)
    end_label = season_label_for_date(end)
    start_year = int(start_label.split("/")[0])
    end_year = int(end_label.split("/")[0])
    return [f"{y}/{y + 1}" for y in range(start_year, end_year + 1)]


def load_or_fetch_season(client, season_label):
    """
    Return a DataFrame of every FotMob player's stats for one EPL season
    (one row per player, columns = STAT_CATEGORIES), or None if FotMob has
    no coverage for that season at all (confirmed true for pre-2016/17).
    Cached to disk since the full history needs ~11 seasons * ~19 stat
    fetches = ~200 requests, and this script gets re-run while iterating.
    """
    cache_path = os.path.join(CACHE_DIR, f"epl_{season_label.replace('/', '-')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return pd.DataFrame(cached) if cached is not None else None

    print(f"  Fetching {season_label}...")
    categories = fetch_league_stat_categories(client, season_label)
    if categories is None:
        with open(cache_path, "w") as f:
            json.dump(None, f)
        return None

    players = {}
    for stat_name in STAT_CATEGORIES:
        url = categories.get(stat_name)
        if not url:
            continue
        for row in fetch_stat_list(client, url):
            pid = row["ParticiantId"]
            p = players.setdefault(pid, {
                "fotmob_id": pid, "fotmob_name": row["ParticipantName"],
                "fotmob_team": row["TeamName"], "minutes_played": row.get("MinutesPlayed"),
                "matches_played": row.get("MatchesPlayed"),
            })
            p[f"fotmob_{stat_name}"] = row["StatValue"]

    records = list(players.values())
    with open(cache_path, "w") as f:
        json.dump(records, f)
    return pd.DataFrame(records)


def find_fotmob_id(transfer_row, season_tables, seasons_newest_first):
    """
    Resolve the FotMob player id for one transfer by name+destination-club
    match, checked newest-season-first (a later season is more likely to
    show a clean, unambiguous club attribution than the join season, which
    for a same-league mid-season move is exactly the contaminated one -
    see module docstring). Returns None if no season yields a match.
    """
    norm_name = normalize_name(transfer_row["name"])
    club_norm = normalize_name(transfer_row["to_club_name"])
    for season in seasons_newest_first:
        table = season_tables.get(season)
        if table is None or table.empty:
            continue
        table = table.copy()
        table["norm_name"] = table["fotmob_name"].map(normalize_name)
        candidates = table[table["norm_name"] == norm_name]
        if candidates.empty:
            close = difflib.get_close_matches(norm_name, table["norm_name"].tolist(), n=3, cutoff=0.6)
            candidates = table[table["norm_name"].isin(close)]
        if candidates.empty:
            continue
        club_hits = candidates[candidates["fotmob_team"].map(lambda t: club_names_match(club_norm, normalize_name(t)))]
        if len(club_hits) >= 1:
            return club_hits.iloc[0]["fotmob_id"]
    return None


def aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables):
    """
    Combine one player's per-season rows (from `agg_seasons`, already with
    the contaminated join-season excluded) into one tenure-level record:
    rate stats (RATE_STATS) are minutes-weighted averages, count stats
    (SUM_STATS) are summed. A season where the id doesn't appear at all
    (e.g. injured that whole season) is just skipped, not treated as zero.
    """
    rows = []
    for season in agg_seasons:
        table = season_tables.get(season)
        if table is None or table.empty:
            continue
        hit = table[table["fotmob_id"] == fotmob_id]
        if not hit.empty:
            rows.append(hit.iloc[0])
    if not rows:
        return None

    out = {"fotmob_seasons_used": len(rows), "fotmob_total_minutes": sum(r["minutes_played"] for r in rows),
           "fotmob_total_matches": sum(r["matches_played"] for r in rows)}
    total_minutes = out["fotmob_total_minutes"] or 1
    for stat_name in STAT_CATEGORIES:
        col = f"fotmob_{stat_name}"
        values = [(r[col], r["minutes_played"]) for r in rows if col in r and pd.notna(r[col])]
        if not values:
            out[col] = None
        elif stat_name in RATE_STATS:
            out[col] = sum(v * w for v, w in values) / sum(w for _, w in values if w) if any(w for _, w in values) else sum(v for v, _ in values) / len(values)
        else:
            out[col] = sum(v for v, _ in values)
    return out


def main():
    """Fetch all covered EPL seasons, stitch per-tenure stats for every EPL-destination transfer since 2016/17, report coverage."""
    df = pd.read_csv(TRANSFERS_PATH, parse_dates=["transfer_date"])
    df["tenure_end"] = df["transfer_date"] + pd.to_timedelta(df["tenure_days"], unit="D")
    epl = df[(df["to_domestic_competition_id"] == "GB1") & (df["transfer_date"] >= f"{FIRST_COVERED_SEASON_YEAR}-08-01")].copy()
    print(f"{len(epl):,} EPL-destination transfers since {FIRST_COVERED_SEASON_YEAR}/{FIRST_COVERED_SEASON_YEAR + 1}")

    print(f"\nLoading/fetching {len(ALL_SEASON_LABELS)} seasons of EPL stats (cached after first run)...")
    season_tables = {}
    with httpx.Client() as client:
        for season in ALL_SEASON_LABELS:
            season_tables[season] = load_or_fetch_season(client, season)
    covered_seasons = [s for s, t in season_tables.items() if t is not None]
    print(f"Seasons with real FotMob coverage: {covered_seasons}")

    results = []
    identified_no_data = 0
    unidentified = 0
    for _, t in epl.iterrows():
        tenure_seasons = [s for s in seasons_overlapping(t["transfer_date"], t["tenure_end"]) if s in covered_seasons]
        if not tenure_seasons:
            continue  # tenure entirely pre-2016/17, or entirely in an uncovered season

        is_intra_epl = t["from_domestic_competition_id"] == "GB1"
        is_midseason_join = t["transfer_date"].month not in (6, 7, 8)
        join_season = season_label_for_date(t["transfer_date"])
        contaminated = join_season if (is_intra_epl and is_midseason_join and join_season in tenure_seasons) else None
        agg_seasons = [s for s in tenure_seasons if s != contaminated]

        fotmob_id = find_fotmob_id(t, season_tables, list(reversed(tenure_seasons)))
        if fotmob_id is None:
            unidentified += 1
            continue

        agg = aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables)
        if agg is None:
            identified_no_data += 1
            continue

        results.append({**t.to_dict(), "fotmob_id": fotmob_id, "contaminated_season_excluded": contaminated, **agg})

    results_df = pd.DataFrame(results)
    results_df.to_csv(MATCHED_OUT_PATH, index=False)

    lines = []
    lines.append(f"EPL transfers since 2016/17 with any season coverage: {len(epl[epl['transfer_date'] >= f'{FIRST_COVERED_SEASON_YEAR}-08-01'])}")
    lines.append(f"  -> matched with usable FotMob tenure stats: {len(results_df)} ({100 * len(results_df) / len(epl):.0f}%)")
    lines.append(f"  -> identified but zero usable seasons (fully contaminated tenure): {identified_no_data}")
    lines.append(f"  -> not identified at all (name/club match failed): {unidentified}")
    lines.append("")
    lines.append("Stat coverage among matched (non-null %):")
    for stat_name in STAT_CATEGORIES:
        col = f"fotmob_{stat_name}"
        pct = 100 * results_df[col].notna().mean() if col in results_df.columns and len(results_df) else 0
        lines.append(f"  {stat_name:28s} {pct:5.1f}%")
    lines.append("")
    n_contaminated = (results_df["contaminated_season_excluded"].notna()).sum() if len(results_df) else 0
    lines.append(f"Transfers where a same-season Premier League-to-Premier League move's join season was excluded from the aggregate: {n_contaminated}")
    lines.append(f"Multi-season tenures (>1 season stitched together): {(results_df['fotmob_seasons_used'] > 1).sum() if len(results_df) else 0}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT_OUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {MATCHED_OUT_PATH}")
    print(f"Wrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
