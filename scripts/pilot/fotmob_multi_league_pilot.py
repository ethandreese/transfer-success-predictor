"""
Generalizes fotmob_epl_pilot_v2.py (which validated the season-stitching
approach on the Premier League alone) to all 23 destination leagues that
appear in data/transfers_processed.csv, using the same fetch/cache/stitch
machinery - see that file's docstring for the two stitching problems
(multi-season tenures, same-league mid-season moves) this reuses.

FotMob league ids were resolved from https://www.fotmob.com/api/data/
allLeagues (its "countries" list gives every country's league ids) and
spot-verified by fetching each id and checking details.name/country
matches (see LEAGUE_MAP below).

The one thing that does NOT carry over cleanly from the EPL pilot: club
name matching. EPL_CLUB_ALIASES in fotmob_epl_pilot.py is a hand-built
table of ~19 English abbreviation pairs (Man City/Manchester City etc.)
found by inspecting real unmatched rows. Pulling the equivalent short-name
lists for the other 22 leagues shows the same problem is worse and more
inconsistent elsewhere - Bundesliga transfers alone use both "Dortmund"
and "Bor. Dortmund", both "Frankfurt" and "E. Frankfurt", for the same
club - so a hand-built table isn't practical to build honestly for 22
more leagues in one pass. Uses a generic matcher instead (normalize,
strip a small deny-list of pure corporate-form tokens like "fc"/"ac"/
"1.", then substring or shared-significant-word or fuzzy-ratio match) -
weaker than the hand-tuned EPL table, so expect a lower match rate on
leagues with messier short names, reported per league below rather than
papered over with one overall number.
"""
import difflib
import json
import os
import re
import sys
import time
import unicodedata

import httpx
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fotmob_epl_pilot import HEADERS, STAT_CATEGORIES, fetch_league_stat_categories, fetch_stat_list  # noqa: E402
from fotmob_epl_pilot_v2 import STAT_HEADERS, RATE_STATS, SUM_STATS  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_DIR = os.path.dirname(__file__)
TRANSFERS_PATH = os.path.join(OUT_DIR, "..", "..", "data", "transfers_processed.csv")

# competition_id (data/competitions_lookup.csv) -> (fotmob league id, label).
# Verified by fetching https://www.fotmob.com/api/data/leagues?id=<id> and
# checking details.name/details.country against competitions_lookup.csv.
LEAGUE_MAP = {
    "GB1": (47, "Premier League (England)"), "IT1": (55, "Serie A (Italy)"),
    "TR1": (71, "Super Lig (Turkey)"), "L1": (54, "Bundesliga (Germany)"),
    "FR1": (53, "Ligue 1 (France)"), "ES1": (87, "LaLiga (Spain)"),
    "RU1": (63, "Premier League (Russia)"), "NL1": (57, "Eredivisie (Netherlands)"),
    "GR1": (135, "Super League 1 (Greece)"), "UKR1": (441, "Premier League (Ukraine)"),
    "BE1": (40, "First Division A (Belgium)"), "PO1": (61, "Liga Portugal (Portugal)"),
    "DK1": (46, "Superligaen (Denmark)"), "SC1": (64, "Premiership (Scotland)"),
    "A1": (38, "Bundesliga (Austria)"), "SER1": (182, "Super Liga (Serbia)"),
    "SE1": (67, "Allsvenskan (Sweden)"), "C1": (69, "Super League (Switzerland)"),
    "NO1": (59, "Eliteserien (Norway)"), "TS1": (122, "1. Liga (Czechia)"),
    "PL1": (196, "Ekstraklasa (Poland)"), "RO1": (189, "Liga I (Romania)"),
    "KR1": (252, "HNL (Croatia)"),
}

FIRST_SEASON_YEAR = 2013  # matches the full transfers_processed.csv range - let coverage probing find each league's real start
CURRENT_SEASON_YEAR = 2026  # "2026/2027" - today's date is 2026-08-28
ALL_SEASON_LABELS = [f"{y}/{y + 1}" for y in range(FIRST_SEASON_YEAR, CURRENT_SEASON_YEAR + 1)]

# Pure corporate-form tokens that carry no identifying information, seen
# across the leagues in our data (German "1.FC"/"VfB"/"TSG", Italian "AC"/
# "SSC", Spanish "CD"/"UD"/"RCD", French "AJ"/"OGC"/"SM", English "AFC" -
# NOT "City"/"United"/"Town", which combine with a place name to actually
# distinguish clubs, e.g. Leicester City vs Leicester).
CLUB_FILLER_TOKENS = {
    "fc", "cf", "ac", "afc", "sc", "sv", "ca", "cd", "sd", "ud", "rcd", "vfb", "vfl",
    "tsg", "ssc", "aj", "og", "ogc", "sm", "fk", "sk", "pfc", "bc", "calcio", "club",
    "1", "04", "05", "1925", "de", "la", "le", "der",
}

# unicodedata's NFKD decomposition only handles letters that ARE a base
# letter + combining diacritic (e.g. e-acute). These Nordic/Slavic letters
# are their own codepoints, not decomposable that way, so the ascii-encode/
# ignore step in normalize_club() below would silently DROP them instead
# of transliterating them - "København" was coming out "Kbenhavn" and no
# longer matching "Copenhagen" at all. Mapped explicitly using each
# letter's standard transliteration instead.
NORDIC_SLAVIC_TRANSLATION = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "ß": "ss",
})

# A small number of clubs whose FotMob name and Transfermarkt short name
# share no substring, common significant word, or high fuzzy ratio at all -
# an acronym (PSG), an old/nickname form (Stade Rennais -> Rennes), or a
# name that's in a different language entirely (Copenhagen -> Kobenhavn).
# The rest of each league's clubs are left to the generic matcher; these
# are only the ones found by inspecting real unmatched Ligue 1/Denmark
# rows (see fotmob_multi_league_pilot's follow-up commit).
LEAGUE_CLUB_ALIASES = {
    "FR1": {"psg": "paris saint germain", "stade rennais": "rennes"},
    "DK1": {"copenhagen": "fc kobenhavn", "fc copenhagen": "fc kobenhavn"},
}


def normalize_club(name):
    """
    Same accent/punctuation stripping as fotmob_epl_pilot.normalize_name,
    factored out so it's reusable on club names here, plus an explicit
    pre-translation for Latin-extended letters NFKD doesn't decompose
    (o/a/ae-with-ring-or-slash etc.) - found because "FC Kobenhavn" was
    silently losing its o to "FC Kbenhavn" and no longer matching
    "Copenhagen" at all once that letter vanished, which would otherwise
    affect any Danish/Norwegian/Polish/Croatian name using them.
    """
    n = str(name).translate(NORDIC_SLAVIC_TRANSLATION)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def club_significant_words(norm_name):
    """Words left after dropping CLUB_FILLER_TOKENS and anything under 4 characters (too short to be distinguishing on its own)."""
    return {w for w in norm_name.split() if w not in CLUB_FILLER_TOKENS and len(w) >= 4}


def club_names_match_generic(norm_a, norm_b, comp_id=None):
    """
    Checked in order: an explicit LEAGUE_CLUB_ALIASES entry for this
    league (for the handful of pairs no generic rule can catch - an
    acronym, a nickname, a different-language name), then three
    progressively looser generic checks: exact/substring match, a shared
    significant word (post-filler-stripping), or a high fuzzy-ratio
    (catches transliteration drift, e.g. Cyrillic names romanized
    slightly differently by the two sites).
    """
    aliases = LEAGUE_CLUB_ALIASES.get(comp_id, {})
    if aliases.get(norm_a) == norm_b or aliases.get(norm_b) == norm_a:
        return True
    if norm_a == norm_b or norm_a in norm_b or norm_b in norm_a:
        return True
    if club_significant_words(norm_a) & club_significant_words(norm_b):
        return True
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.72


def season_label_for_date(date):
    """Map a calendar date to its (Northern European, Aug-May) season label - every league here uses this convention."""
    return f"{date.year}/{date.year + 1}" if date.month >= 8 else f"{date.year - 1}/{date.year}"


def seasons_overlapping(transfer_date, tenure_end):
    """All season labels a [transfer_date, tenure_end] tenure window touches, clipped to today's season."""
    today = pd.Timestamp.today().normalize()
    end = min(tenure_end, today)
    start_year = int(season_label_for_date(transfer_date).split("/")[0])
    end_year = int(season_label_for_date(end).split("/")[0])
    return [f"{y}/{y + 1}" for y in range(start_year, end_year + 1)]


def load_or_fetch_season(client, comp_id, league_id, season_label):
    """Same as fotmob_epl_pilot_v2's version, parameterized by league - see that file for the caching rationale."""
    cache_path = os.path.join(CACHE_DIR, f"{comp_id}_{season_label.replace('/', '-')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return pd.DataFrame(cached) if cached is not None else None

    categories = fetch_league_stat_categories(client, season=season_label, league_id=league_id)
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
        time.sleep(0.15)  # polite pacing - this pilot ends up making several thousand requests across 23 leagues

    records = list(players.values())
    with open(cache_path, "w") as f:
        json.dump(records, f)
    return pd.DataFrame(records)


def find_fotmob_id(transfer_row, season_tables, seasons_newest_first, comp_id=None):
    """Same identity-resolution strategy as fotmob_epl_pilot_v2 (newest-season-first, name+club), using the generic club matcher above."""
    norm_name = normalize_club(transfer_row["name"])  # club normalizer works fine for player names too (same char stripping)
    club_norm = normalize_club(transfer_row["to_club_name"])
    for season in seasons_newest_first:
        table = season_tables.get(season)
        if table is None or table.empty:
            continue
        table = table.copy()
        table["norm_name"] = table["fotmob_name"].map(normalize_club)
        candidates = table[table["norm_name"] == norm_name]
        if candidates.empty:
            close = difflib.get_close_matches(norm_name, table["norm_name"].tolist(), n=3, cutoff=0.6)
            candidates = table[table["norm_name"].isin(close)]
        if candidates.empty:
            continue
        club_hits = candidates[candidates["fotmob_team"].map(lambda t: club_names_match_generic(club_norm, normalize_club(t), comp_id))]
        if len(club_hits) >= 1:
            return club_hits.iloc[0]["fotmob_id"]
    return None


def aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables):
    """Same aggregation as fotmob_epl_pilot_v2: minutes-weighted average for RATE_STATS, sum for SUM_STATS, across the usable (non-contaminated) seasons."""
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
    for stat_name in STAT_CATEGORIES:
        col = f"fotmob_{stat_name}"
        values = [(r[col], r["minutes_played"]) for r in rows if col in r and pd.notna(r[col])]
        if not values:
            out[col] = None
        elif stat_name in RATE_STATS:
            weight_sum = sum(w for _, w in values if w)
            out[col] = sum(v * w for v, w in values) / weight_sum if weight_sum else sum(v for v, _ in values) / len(values)
        else:
            out[col] = sum(v for v, _ in values)
    return out


def run_league(client, comp_id, transfers):
    """
    Full pilot for one league: fetch/cache every season, stitch tenure
    stats for every transfer into that league, return (matched_df,
    coverage_dict) - see fotmob_epl_pilot_v2.main for the single-league
    version this generalizes.
    """
    league_id, label = LEAGUE_MAP[comp_id]
    print(f"\n=== {comp_id} - {label} (fotmob id {league_id}) ===")
    print(f"{len(transfers):,} candidate transfers")

    season_tables = {}
    for season in ALL_SEASON_LABELS:
        season_tables[season] = load_or_fetch_season(client, comp_id, league_id, season)
    covered = [s for s, t in season_tables.items() if t is not None]
    print(f"  Seasons with coverage: {covered[0] if covered else 'NONE'} .. {covered[-1] if covered else 'NONE'} ({len(covered)} seasons)")

    results = []
    identified_no_data = 0
    unidentified = 0
    n_contaminated = 0
    for _, t in transfers.iterrows():
        tenure_seasons = [s for s in seasons_overlapping(t["transfer_date"], t["tenure_end"]) if s in covered]
        if not tenure_seasons:
            continue

        is_intra_league = t["from_domestic_competition_id"] == comp_id
        is_midseason_join = t["transfer_date"].month not in (6, 7, 8)
        join_season = season_label_for_date(t["transfer_date"])
        contaminated = join_season if (is_intra_league and is_midseason_join and join_season in tenure_seasons) else None
        agg_seasons = [s for s in tenure_seasons if s != contaminated]
        if contaminated:
            n_contaminated += 1

        fotmob_id = find_fotmob_id(t, season_tables, list(reversed(tenure_seasons)), comp_id)
        if fotmob_id is None:
            unidentified += 1
            continue

        agg = aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables)
        if agg is None:
            identified_no_data += 1
            continue

        results.append({**t.to_dict(), "competition_id": comp_id, "fotmob_id": fotmob_id, **agg})

    results_df = pd.DataFrame(results)
    n_with_coverage = sum(1 for _, t in transfers.iterrows() if seasons_overlapping(t["transfer_date"], t["tenure_end"]) and any(s in covered for s in seasons_overlapping(t["transfer_date"], t["tenure_end"])))
    coverage = {
        "competition_id": comp_id, "label": label, "n_transfers": len(transfers),
        "n_seasons_covered": len(covered), "first_covered_season": covered[0] if covered else None,
        "n_with_any_season_coverage": n_with_coverage, "n_matched": len(results_df),
        "n_contaminated_excluded": n_contaminated, "n_identified_no_data": identified_no_data,
        "n_unidentified": unidentified,
        "match_rate_of_covered": round(100 * len(results_df) / n_with_coverage, 1) if n_with_coverage else None,
    }
    print(f"  Matched {len(results_df)}/{n_with_coverage} of transfers with any season coverage ({coverage['match_rate_of_covered']}%)")
    return results_df, coverage


def main():
    """Run the full-history pilot across every league in LEAGUE_MAP, write one combined matched CSV and a per-league coverage report."""
    df = pd.read_csv(TRANSFERS_PATH, parse_dates=["transfer_date"])
    df["tenure_end"] = df["transfer_date"] + pd.to_timedelta(df["tenure_days"], unit="D")
    df = df[df["transfer_date"] >= f"{FIRST_SEASON_YEAR}-08-01"]

    all_results = []
    all_coverage = []
    with httpx.Client(timeout=30) as client:
        for comp_id in LEAGUE_MAP:
            league_transfers = df[df["to_domestic_competition_id"] == comp_id].copy()
            if league_transfers.empty:
                continue
            results_df, coverage = run_league(client, comp_id, league_transfers)
            if not results_df.empty:
                all_results.append(results_df)
            all_coverage.append(coverage)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(os.path.join(OUT_DIR, "fotmob_all_leagues_matched.csv"), index=False)

    coverage_df = pd.DataFrame(all_coverage)
    coverage_df.to_csv(os.path.join(OUT_DIR, "fotmob_all_leagues_coverage.csv"), index=False)
    print("\n\n=== SUMMARY ===")
    print(coverage_df.to_string(index=False))
    total_transfers = coverage_df["n_transfers"].sum()
    total_matched = coverage_df["n_matched"].sum()
    print(f"\nTotal: {total_matched:,} / {total_transfers:,} transfers matched with usable FotMob stats ({100 * total_matched / total_transfers:.0f}%)")


if __name__ == "__main__":
    main()
