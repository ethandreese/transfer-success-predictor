"""
Pilot: can FotMob's Premier League 2025/2026 season stats (rating, tackles,
passes, key passes, saves, etc.) be pulled and matched onto our existing
2025 EPL-destination transfers in data/transfers_processed.csv?

Scope is deliberately narrow (one league, one season) to test two open
questions before committing to a full integration:
  1. Is the data reachable at all without hitting FotMob's anti-scraping
     gateway (apps.fotmob.com/searchapi returns 403 - locked behind a
     signed-request scheme). The www.fotmob.com/api/data/* +
     data.fotmob.com/stats/* endpoints used here are what the site's own
     Next.js frontend calls and returned plain 200s with no signature in
     manual testing - not guaranteed to stay that way.
  2. Can FotMob players be matched onto Transfermarkt players by name alone
     (no shared ID between the two sites)?

Not wired into the scoring formula - this only reports match rate and
stat coverage so we can decide whether a real integration is worth doing.
"""
import difflib
import json
import os
import re
import time
import unicodedata

import httpx
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
}
EPL_LEAGUE_ID = 47
SEASON = "2025/2026"

STAT_CATEGORIES = [
    "rating", "mins_played", "goals_per_90", "expected_goals_per_90",
    "expected_assists_per_90", "accurate_pass", "total_att_assist",
    "big_chance_created", "won_contest", "total_tackle", "interception",
    "effective_clearance", "defensive_contributions", "ball_recovery",
    "clean_sheet", "saves", "_save_percentage", "goals_conceded", "fouls",
]

OUT_DIR = os.path.dirname(__file__)
RAW_CACHE_PATH = os.path.join(OUT_DIR, "fotmob_epl_2025_raw.json")
MATCHED_OUT_PATH = os.path.join(OUT_DIR, "fotmob_epl_2025_matched.csv")
UNMATCHED_OUT_PATH = os.path.join(OUT_DIR, "fotmob_epl_2025_unmatched.csv")

TRANSFERS_PATH = os.path.join(OUT_DIR, "..", "..", "data", "transfers_processed.csv")


def normalize_name(name):
    """Strip accents/punctuation and lowercase, so 'Aït-Nouri' == 'ait nouri'."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def fetch_league_stat_categories(client):
    """Get the season's stat category list + each category's fetchAllUrl."""
    resp = client.get(
        "https://www.fotmob.com/api/data/leagues",
        params={"id": EPL_LEAGUE_ID, "season": SEASON},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    season_id = data["details"]["selectedSeason"]
    print(f"  Selected season: {season_id} (requested {SEASON})")
    cats = {p["name"]: p["fetchAllUrl"] for p in data["stats"]["players"]}
    return cats


def fetch_stat_list(client, url):
    """Fetch one data.fotmob.com/stats/.../<stat>.json leaderboard and return its StatList rows."""
    resp = client.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["TopLists"][0]["StatList"]


def build_fotmob_player_table(client, categories):
    """One row per FotMob player id, columns = each stat's value."""
    players = {}  # fotmob_id -> dict
    for stat_name in STAT_CATEGORIES:
        url = categories.get(stat_name)
        if not url:
            print(f"  ! no fetchAllUrl for stat '{stat_name}', skipping")
            continue
        rows = fetch_stat_list(client, url)
        print(f"  {stat_name}: {len(rows)} players")
        for row in rows:
            pid = row["ParticiantId"]
            p = players.setdefault(pid, {
                "fotmob_id": pid,
                "fotmob_name": row["ParticipantName"],
                "fotmob_team": row["TeamName"],
                "minutes_played": row.get("MinutesPlayed"),
                "matches_played": row.get("MatchesPlayed"),
            })
            p[f"fotmob_{stat_name}"] = row["StatValue"]
        time.sleep(0.3)  # polite pacing, this is a small one-off pilot fetch
    return pd.DataFrame(players.values())


def load_epl_2025_transfers():
    """Load our processed transfers and filter to Premier League destinations dated in 2025."""
    df = pd.read_csv(TRANSFERS_PATH, parse_dates=["transfer_date"])
    epl = df[
        (df["to_domestic_competition_id"] == "GB1")
        & (df["transfer_date"].dt.year == 2025)
    ].copy()
    return epl


def match_players(transfers, fotmob_df):
    """
    Match each Transfermarkt transfer onto a FotMob player by normalized name,
    since neither site exposes the other's player id. When a name maps to
    several FotMob players (e.g. common names), disambiguate using whether
    the destination club name overlaps the FotMob team name. Returns
    (matched_df, unmatched_df).
    """
    fotmob_df = fotmob_df.copy()
    fotmob_df["norm_name"] = fotmob_df["fotmob_name"].map(normalize_name)
    by_name = {}
    for _, row in fotmob_df.iterrows():
        by_name.setdefault(row["norm_name"], []).append(row)

    matched_rows = []
    unmatched_rows = []
    for _, t in transfers.iterrows():
        norm = normalize_name(t["name"])
        candidates = by_name.get(norm, [])
        if len(candidates) == 1:
            fm = candidates[0]
            combined = {**t.to_dict(), **fm.to_dict(), "match_method": "exact_name"}
            matched_rows.append(combined)
            continue
        if len(candidates) > 1:
            # disambiguate by destination club name substring match
            club_norm = normalize_name(t["to_club_name"])
            club_hits = [c for c in candidates if club_norm in normalize_name(c["fotmob_team"]) or normalize_name(c["fotmob_team"]) in club_norm]
            if len(club_hits) == 1:
                fm = club_hits[0]
                combined = {**t.to_dict(), **fm.to_dict(), "match_method": "exact_name+club"}
                matched_rows.append(combined)
                continue

        # Fallback for name-format mismatches between the two sites (e.g.
        # Transfermarkt's "El Hadji Malick Diouf" vs FotMob's "Malick Diouf",
        # or accented-spelling drift like "Yeremy" vs "Yeremi") - only
        # accepted when the destination club also matches, so a wrong-but-
        # similar name elsewhere in the league can't silently steal the row.
        club_norm = normalize_name(t["to_club_name"])
        club_pool = [c for c in fotmob_df.itertuples() if club_norm in normalize_name(c.fotmob_team) or normalize_name(c.fotmob_team) in club_norm]
        pool_names = [normalize_name(c.fotmob_name) for c in club_pool]
        close = difflib.get_close_matches(norm, pool_names, n=1, cutoff=0.6)
        if close:
            fm = club_pool[pool_names.index(close[0])]
            combined = {**t.to_dict(), **fm._asdict(), "match_method": "fuzzy_name+club"}
            matched_rows.append(combined)
            continue

        unmatched_rows.append(t.to_dict())

    return pd.DataFrame(matched_rows), pd.DataFrame(unmatched_rows)


def main():
    """Run the full pilot: fetch FotMob EPL 2025/26 stats, match onto our 2025 EPL transfers, report coverage."""
    print("Fetching FotMob Premier League 2025/2026 stat categories...")
    with httpx.Client() as client:
        categories = fetch_league_stat_categories(client)
        print(f"Building per-player stat table from {len(STAT_CATEGORIES)} categories...")
        fotmob_df = build_fotmob_player_table(client, categories)

    print(f"\nFotMob player table: {len(fotmob_df):,} distinct players appear in at least one leaderboard")
    fotmob_df.to_json(RAW_CACHE_PATH, orient="records", indent=2)

    print("\nLoading 2025 EPL-destination transfers from our dataset...")
    transfers = load_epl_2025_transfers()
    print(f"{len(transfers):,} candidate transfers")

    matched, unmatched = match_players(transfers, fotmob_df)
    print(f"\nMatched {len(matched):,} / {len(transfers):,} ({100*len(matched)/len(transfers):.0f}%)")

    print("\nStat coverage among matched players (non-null %):")
    for stat_name in STAT_CATEGORIES:
        col = f"fotmob_{stat_name}"
        if col in matched.columns:
            pct = 100 * matched[col].notna().mean()
            print(f"  {stat_name:28s} {pct:5.1f}%")
        else:
            print(f"  {stat_name:28s}  n/a (category missing)")

    matched.to_csv(MATCHED_OUT_PATH, index=False)
    unmatched.to_csv(UNMATCHED_OUT_PATH, index=False)
    print(f"\nWrote {MATCHED_OUT_PATH}")
    print(f"Wrote {UNMATCHED_OUT_PATH}")

    if len(unmatched):
        print("\nUnmatched players (need manual review):")
        for _, r in unmatched.iterrows():
            print(f"  {r['name']:28s} -> {r['to_club_name']}")


if __name__ == "__main__":
    main()
