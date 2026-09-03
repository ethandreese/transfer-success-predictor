"""
Fetch FotMob's per-season stat leaderboards (rating, tackles, passes,
defensive actions, saves, etc.) for every league that appears as a
destination in data/transfers_processed.csv or data/loans_processed.csv,
and stitch each tenure (permanent transfer or loan spell) into one row of
aggregated stats - the piece of data that lets build_dataset.py add a real
defensive/technical performance signal for positions (defenders,
goalkeepers) where goal contributions alone are a weak or meaningless
signal (see data/score_weights.json and data/loan_score_weights.json).

This is the production version of the investigation carried out in
scripts/pilot/ (fotmob_epl_pilot.py -> fotmob_epl_pilot_v2.py ->
fotmob_multi_league_pilot.py, in that order - read those for the full
reasoning trail). Two things validated there and just reused here:

1. Season stitching. Our score is computed over a player's whole tenure
   (join date -> departure), but FotMob only exposes whole-*season*
   leaderboards. A multi-season tenure needs several seasons combined
   (minutes-weighted average for rate stats, summed for counts). A
   same-league mid-season move is worse: FotMob attributes a player's
   ENTIRE season total to whichever club they're registered at when
   fetched, not split by stint (confirmed on real cases, e.g. Marc Guehi's
   Jan 2026 Crystal Palace -> Man City move showing a near-full season
   under Man City) - so that join season is excluded from the aggregate
   whenever the move was intra-league and mid-season, using only the
   season(s) after it.
2. Club-name matching. Transfermarkt's short names and FotMob's full
   names often share no substring at all (Man City/Manchester City, PSG/
   Paris Saint-Germain, Copenhagen/Kobenhavn) - handled with a generic
   matcher (strip corporate-form tokens, then substring/shared-word/
   fuzzy-ratio) plus a small LEAGUE_CLUB_ALIASES table for the handful of
   pairs no generic rule catches, found by inspecting real unmatched rows
   per league.

FotMob's coverage has two real ceilings, not effort problems: no stats
at all before a league-specific season (varies by league, none confirmed
before 2013/2014 anywhere in our data), and for Ukraine specifically,
FotMob's own league page exposes only 5 basic categories (goals/assists/
cards) - none of which overlap STAT_CATEGORIES below, so it's a real,
permanent 0% for that one league (see README's "Known limitations").

Optional, like scripts/fetch_transfer_types.py: build_dataset.py degrades
gracefully (treats every transfer as having no FotMob data) if
data/raw/fotmob_stats_cache.csv doesn't exist yet. Safe to interrupt and
re-run - resumes from data/raw/fotmob_season_cache/*.json (gitignored,
regenerable) rather than refetching every league-season from scratch.
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
}

# competition_id (data/competitions_lookup.csv) -> (fotmob league id, label).
# Resolved from https://www.fotmob.com/api/data/allLeagues (its "countries"
# list gives every country's league ids) and spot-verified by fetching each
# id and checking details.name/details.country matches.
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

# The stat leaderboards pulled for every league-season - a subset of what
# FotMob offers (skips physical/speed-tracking data, which most leagues
# don't have anyway - see fotmob_epl_pilot.py's investigation). Header text
# (captured live) says whether each is a per-90/percentage rate (minutes-
# weighted average across a multi-season tenure) or a season-total count
# (summed) - see build_dataset.py's compute_fotmob_component_pcts for how
# these actually feed the score.
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
STAT_CATEGORIES = list(STAT_HEADERS)
RATE_STATS = {name for name, header in STAT_HEADERS.items() if "per 90" in header or "percentage" in header or name == "rating"}
SUM_STATS = set(STAT_HEADERS) - RATE_STATS

FIRST_SEASON_YEAR = 2013  # matches transfers_processed.csv's full range - let per-league coverage probing find each league's real start
_today = pd.Timestamp.today()
CURRENT_SEASON_YEAR = _today.year if _today.month >= 8 else _today.year - 1  # e.g. Aug 2026 onward -> season "2026/2027", matching season_label_for_date's own Aug cutoff below
ALL_SEASON_LABELS = [f"{y}/{y + 1}" for y in range(FIRST_SEASON_YEAR, CURRENT_SEASON_YEAR + 1)]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE_DIR = os.path.join(RAW_DIR, "fotmob_season_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_PATH = os.path.join(RAW_DIR, "fotmob_stats_cache.csv")
TRANSFERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")
LOANS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loans_processed.csv")

# Same packaged Transfermarkt dataset build_dataset.py reads (kagglehub cache,
# overridable the same way) - used here only to work out which significant
# words are safe to match clubs on within a league (see AMBIGUOUS_CLUB_WORDS).
TRANSFERMARKT_RAW_DIR = os.environ.get(
    "TRANSFERMARKT_RAW_DIR",
    os.path.expanduser("~/.cache/kagglehub/datasets/davidcariboo/player-scores/versions/677"),
)

# Pure corporate-form tokens that carry no identifying information (German
# "1.FC"/"VfB"/"TSG", Italian "AC"/"SSC", Spanish "CD"/"UD"/"RCD", French
# "AJ"/"OGC"/"SM", English "AFC") - NOT "City"/"United"/"Town", which
# combine with a place name to actually distinguish clubs (Leicester City
# vs Leicester).
CLUB_FILLER_TOKENS = {
    "fc", "cf", "ac", "afc", "sc", "sv", "ca", "cd", "sd", "ud", "rcd", "vfb", "vfl",
    "tsg", "ssc", "aj", "og", "ogc", "sm", "fk", "sk", "pfc", "bc", "calcio", "club",
    "1", "04", "05", "1925", "de", "la", "le", "der",
}

# unicodedata's NFKD decomposition only handles letters that ARE a base
# letter + combining diacritic (e.g. e-acute). These Nordic/Slavic letters
# are their own codepoints, not decomposable that way, so a plain ascii-
# encode/ignore would silently DROP them instead of transliterating them -
# "Kobenhavn" was coming out "Kbenhavn" and no longer matching "Copenhagen"
# at all. Mapped explicitly using each letter's standard transliteration.
NORDIC_SLAVIC_TRANSLATION = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "ß": "ss",
})

# Leagues whose FotMob season identifier is a single calendar year (their
# whole season plays out within one calendar year, roughly Mar-Nov - unlike
# every other league here's Aug-May season, which straddles two calendar
# years the same way our "YYYY/YYYY+1" labels do), not the split-year
# format every other league in LEAGUE_MAP uses. Verified live against
# https://www.fotmob.com/api/data/leagues?id=<id> - allAvailableSeasons -
# for every league in LEAGUE_MAP; only these two came back single-year
# ("2026","2025",...) rather than split-year ("2026/2027","2025/2026",...).
# Passing our usual "YYYY/YYYY+1" label as the season param for these two
# doesn't error - FotMob silently falls back to its *current* season -
# which was returning the identical roster for every requested season
# (checked directly: every cached NO1 season file held the same 21-player
# Rosenborg roster). A real, silent contamination bug, not just a missed
# match: a matched transfer was being scored against whatever season
# happened to be current at fetch time, not its actual tenure.
SINGLE_YEAR_SEASON_LEAGUES = {"NO1", "SE1"}


def fotmob_season_param(comp_id, season_label):
    """
    Convert our internal split-year season label ("2019/2020") into the
    season identifier FotMob's API actually expects for this league - a
    no-op for every league except SINGLE_YEAR_SEASON_LEAGUES. For Norway/
    Sweden, whose one real season is a calendar year, use the label's
    second year: a Mar-Nov Nordic season "Y" overlaps our "Y-1/Y" label's
    Mar-Jul portion (5 months) more than our "Y/Y+1" label's Aug-Nov
    portion (4 months), so "Y-1/Y" -> FotMob season "Y" is the closer
    single-season approximation of a window that inherently straddles two
    calendar years either way. This only affects which season's stats a
    transfer's tenure gets stitched from at the edges (Aug-Nov joins get
    attributed one calendar year later than their exact join date) - a
    real but minor imprecision, and a large improvement over the previous
    silent-contamination bug.
    """
    if comp_id in SINGLE_YEAR_SEASON_LEAGUES:
        return season_label.split("/")[1]
    return season_label


# A small number of clubs whose FotMob name and Transfermarkt short name
# share no substring, common significant word, or high fuzzy ratio at all -
# an acronym (PSG, "Man City"), an old/nickname form ("Stade Rennais" ->
# "Rennes"), or a name in a different language ("Copenhagen" -> a Danish
# spelling). Found by inspecting real unmatched rows per league; the rest
# of each league's clubs are left to the generic matcher.
LEAGUE_CLUB_ALIASES = {
    "GB1": {
        "brighton": "brighton and hove albion", "huddersfield": "huddersfield town",
        "ipswich": "ipswich town", "leeds": "leeds united", "leicester": "leicester city",
        "luton": "luton town", "man city": "manchester city", "man utd": "manchester united",
        "newcastle": "newcastle united", "norwich": "norwich city",
        "nott m forest": "nottingham forest", "nottm forest": "nottingham forest",
        "sheff utd": "sheffield united", "swansea": "swansea city", "tottenham": "tottenham hotspur",
        "west brom": "west bromwich albion", "west ham": "west ham united",
        "wolves": "wolverhampton wanderers",
    },
    "FR1": {"psg": "paris saint germain", "stade rennais": "rennes"},
    "DK1": {"copenhagen": "fc kobenhavn", "fc copenhagen": "fc kobenhavn"},
}


def normalize_club(name):
    """Lowercase, strip accents/punctuation, and pre-translate Nordic/Slavic letters NFKD can't decompose - used for both club and player names."""
    n = str(name).translate(NORDIC_SLAVIC_TRANSLATION)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


# Known transliteration variants of the same word - clubs.csv spells three
# of RU1's Moscow clubs "Moskva"/"Moskau" while to_club_name (matching
# FotMob's English spelling) says "Moscow". Without normalizing these to
# one canonical spelling, "moscow" would look falsely unique to whichever
# club happens to already say "Moscow" in clubs.csv (Torpedo), instead of
# correctly ambiguous across every real Moscow club - see
# AMBIGUOUS_CLUB_WORDS, which is exactly what let Loko Moscow resolve to
# Dinamo Moscow's stats before this was added.
CLUB_WORD_SYNONYMS = {"moskva": "moscow", "moskau": "moscow"}


def club_significant_words(norm_name):
    """Words left after dropping CLUB_FILLER_TOKENS and anything under 4 characters (too short to be distinguishing on its own), with known transliteration variants (CLUB_WORD_SYNONYMS) folded to one spelling."""
    words = {w for w in norm_name.split() if w not in CLUB_FILLER_TOKENS and len(w) >= 4}
    return {CLUB_WORD_SYNONYMS.get(w, w) for w in words}


def _load_ambiguous_club_words():
    """
    Per competition_id, every significant word (see club_significant_words)
    shared by 2+ *different* real clubs in clubs.csv - e.g. ES1's "real"
    (Real Madrid/Sociedad/Betis/Oviedo/Valladolid/Zaragoza all have it),
    GB1's "city" (Man City/Leicester/Norwich/Swansea/Hull/Stoke/Cardiff) or
    "west" (West Ham United/West Bromwich Albion). club_names_match's
    shared-significant-word tier is meant to catch the SAME club under a
    different rendering (Bayern Munich/Bayern München on "bayern", Sporting
    CP/Sporting Lisbon on... - see below), but a word this common within one
    league isn't distinguishing at all - checked directly against every
    currently-fuzzy/exact-matched transfer: this tier alone was resolving
    Manchester City to Swansea City's stats, Real Madrid to Real Sociedad's
    or Real Betis's, West Ham United to West Bromwich Albion's, and three
    different Moscow-club and Danish-"Boldklub"-club pairs to each other -
    all different real clubs with no other basis for the match. A word
    unique to one club in its league (bayern, freiburg, dortmund, aarhus,
    braga - even though FotMob renders the last as "Sporting Braga") stays
    a valid signal; only words two or more real clubs in the same league
    actually share get excluded.

    Counted by club_id, not name text - clubs.csv itself spells three of
    RU1's Moscow clubs "Moskva"/"Moskau" and one in Cyrillic outright, so
    counting by raw name string would under- or over-count real clubs
    depending on which spelling a given row happens to use; club_id is the
    one thing guaranteed to mean "one real club" regardless of spelling
    (CLUB_WORD_SYNONYMS handles the transliteration side separately, so
    "moscow" itself still comes out ambiguous).
    """
    clubs_path = os.path.join(TRANSFERMARKT_RAW_DIR, "clubs.csv")
    if not os.path.exists(clubs_path):
        return {}
    clubs = pd.read_csv(clubs_path, usecols=["club_id", "name", "domestic_competition_id"])

    ambiguous = {}
    for comp_id, group in clubs.groupby("domestic_competition_id"):
        word_clubs = {}
        for club_id, name in zip(group["club_id"], group["name"]):
            for word in club_significant_words(normalize_club(name)):
                word_clubs.setdefault(word, set()).add(club_id)
        ambiguous[comp_id] = {w for w, ids in word_clubs.items() if len(ids) > 1}
    return ambiguous


AMBIGUOUS_CLUB_WORDS = _load_ambiguous_club_words()


def _load_next_competition():
    """
    For every (player_id, transfer_date) that shows up as some tenure's
    tenure_end in transfers_processed.csv/loans_processed.csv, the
    competition_id the player's real *next* transfer (transfers.csv +
    manual_transfers.csv, matched by player_id and that exact date) landed
    in. Powers the mirror image of the mid-season-join guard in run_league():
    that guard drops the join season when a player arrives mid-season from
    another club in the *same* league, because FotMob attributes the whole
    season to wherever they're registered when fetched, not split by stint
    (confirmed on Marc Guehi's Jan 2026 Crystal Palace -> Man City move
    showing a near-full season under Man City). The same mechanism cuts the
    other way at the *end* of a tenure: if a player leaves mid-season for
    another club in the same league, their outgoing tenure's own final
    season gets contaminated by whatever they did afterward - checked
    directly against every currently FotMob-matched transfer/loan whose
    tenure_end falls mid-season with a same-league next move: 206 of 259
    had their last aggregated season showing a fotmob_team that didn't even
    club_names_match the tenure's own destination (Memphis Depay's Barcelona
    tenure picking up his post-departure Atletico Madrid season, Danny
    Ings's Aston Villa tenure picking up his West Ham one, and so on) - not
    a rare edge case, the dominant outcome once a same-league next move
    exists. None if there's no matching next transfer (still there,
    retired, or its destination club isn't in clubs.csv).
    """
    transfers = pd.read_csv(
        os.path.join(TRANSFERMARKT_RAW_DIR, "transfers.csv"),
        usecols=["player_id", "transfer_date", "to_club_id"], parse_dates=["transfer_date"],
    )
    manual_path = os.path.join(os.path.dirname(__file__), "..", "data", "manual_transfers.csv")
    manual = (
        pd.read_csv(manual_path, usecols=["player_id", "transfer_date", "to_club_id"], parse_dates=["transfer_date"])
        if os.path.exists(manual_path) else pd.DataFrame(columns=["player_id", "transfer_date", "to_club_id"])
    )
    all_transfers = pd.concat([transfers, manual]).drop_duplicates(subset=["player_id", "transfer_date"])

    clubs_path = os.path.join(TRANSFERMARKT_RAW_DIR, "clubs.csv")
    if not os.path.exists(clubs_path):
        return {}
    clubs = pd.read_csv(clubs_path, usecols=["club_id", "domestic_competition_id"])
    club_comp = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))

    all_transfers["next_comp"] = all_transfers["to_club_id"].map(club_comp)
    return {
        (pid, date): comp
        for pid, date, comp in zip(all_transfers["player_id"], all_transfers["transfer_date"], all_transfers["next_comp"])
        if pd.notna(comp)
    }


NEXT_COMPETITION = _load_next_competition()


def club_names_match(norm_a, norm_b, comp_id=None):
    """
    Checked in order: an explicit LEAGUE_CLUB_ALIASES entry for this
    league, then three progressively looser generic checks: exact/
    substring match, a shared significant word (post-filler-stripping,
    excluding words AMBIGUOUS_CLUB_WORDS marks as shared by multiple real
    clubs in this league), or a high fuzzy-ratio (catches transliteration
    drift, e.g. Cyrillic names romanized slightly differently by the two
    sites).
    """
    aliases = LEAGUE_CLUB_ALIASES.get(comp_id, {})
    if aliases.get(norm_a) == norm_b or aliases.get(norm_b) == norm_a:
        return True
    if norm_a == norm_b or norm_a in norm_b or norm_b in norm_a:
        return True
    shared_words = club_significant_words(norm_a) & club_significant_words(norm_b)
    if shared_words - AMBIGUOUS_CLUB_WORDS.get(comp_id, set()):
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


def fetch_league_stat_categories(client, league_id, season):
    """
    Get one league-season's stat category list + each category's
    fetchAllUrl. Returns None if FotMob has no stats at all for that
    league-season (a real, permanent ceiling for some league/era
    combinations - see module docstring), rather than raising, so callers
    can probe a season's coverage cheaply.
    """
    resp = client.get(
        "https://www.fotmob.com/api/data/leagues",
        params={"id": league_id, "season": season},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    players = resp.json().get("stats", {}).get("players")
    if not players:
        return None
    return {p["name"]: p["fetchAllUrl"] for p in players}


def fetch_stat_list(client, url):
    """Fetch one data.fotmob.com/stats/.../<stat>.json leaderboard and return its StatList rows."""
    resp = client.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["TopLists"][0]["StatList"]


def load_or_fetch_season(client, comp_id, league_id, season_label):
    """One league-season's full per-player stat table (one row per FotMob player id), cached to disk so a re-run resumes instead of refetching."""
    cache_path = os.path.join(CACHE_DIR, f"{comp_id}_{season_label.replace('/', '-')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return pd.DataFrame(cached) if cached is not None else None

    categories = fetch_league_stat_categories(client, league_id, fotmob_season_param(comp_id, season_label))
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
        time.sleep(0.15)  # polite pacing - a full run makes several thousand requests across 23 leagues

    records = list(players.values())
    with open(cache_path, "w") as f:
        json.dump(records, f)
    return pd.DataFrame(records)


def find_fotmob_id(transfer_row, season_tables, seasons_newest_first, comp_id):
    """
    Resolve the FotMob player id for one transfer by name+destination-club
    match.

    Two passes, not one interleaved pass - this used to try exact-then-
    fuzzy *per season*, newest-first, returning on the first hit. That let
    a fuzzy false positive in an early (newest) season pre-empt a correct
    exact match sitting in an older season, whenever the player had zero
    presence in that newest season at all - confirmed on a real case:
    Marc-Andre ter Stegen had no FotMob data for the newest season checked
    (genuinely absent that season), so the old logic fell to a fuzzy
    name-only guess there and matched him to Andreas Christensen - a
    different Barcelona player entirely (ratio 0.600, exactly the old
    cutoff) - while his real, exact-match season (with real saves/rating/
    goals-conceded data) sat unchecked further down the list. Trying exact
    match across *every* season first fixes this: a real identity match
    from any season always outranks a fuzzy guess from a more recent one.

    Pass 2 (fuzzy) only runs when the exact name never appears in *any*
    season at all - not merely when a season's exact match exists but
    fails the club check. That distinction matters: if the exact name is
    found but attributed to a different club that season (e.g. already
    transferred on by the time FotMob's coverage picks up), we know
    exactly who this is, and a season where they don't play for this club
    is real information, not grounds to guess a different, merely
    similar-named player who does. Skipping that guard was producing its
    own false positives - e.g. Mamadou Sakho's real, correctly-rejected
    entries (wrong club by then) still got treated as a "close match"
    candidate pool alongside Mohamed Salah, and Salah's incidental
    Liverpool link won the club check despite the names bearing no real
    resemblance to each other.

    Checked directly against the full dataset after this fix: 72 transfers
    changed identity (all confirmed corrections, several by cross-checking
    the newly-resolved fotmob_name against the real player), zero cases
    where a previously-correct match broke.
    """
    norm_name = normalize_club(transfer_row["name"])
    club_norm = normalize_club(transfer_row["to_club_name"])
    name_appears_exactly = False

    for season in seasons_newest_first:
        table = season_tables.get(season)
        if table is None or table.empty:
            continue
        table = table.copy()
        table["norm_name"] = table["fotmob_name"].map(normalize_club)
        candidates = table[table["norm_name"] == norm_name]
        if candidates.empty:
            continue
        name_appears_exactly = True
        club_hits = candidates[candidates["fotmob_team"].map(lambda t: club_names_match(club_norm, normalize_club(t), comp_id))]
        if len(club_hits) >= 1:
            return club_hits.iloc[0]["fotmob_id"]

    if name_appears_exactly:
        return None

    for season in seasons_newest_first:
        table = season_tables.get(season)
        if table is None or table.empty:
            continue
        table = table.copy()
        table["norm_name"] = table["fotmob_name"].map(normalize_club)
        close = difflib.get_close_matches(norm_name, table["norm_name"].tolist(), n=3, cutoff=0.6)
        # get_close_matches' raw character-ratio cutoff turned out to accept
        # far more than the one asymmetric-ratio case first found here
        # (requiring the ratio to also clear 0.6 in reverse - see git
        # history): auditing every currently-fuzzy-matched transfer/loan for
        # a FotMob id claimed by two different real Transfermarkt players
        # turned up 34 such collisions, e.g. Mario Suarez wrongly resolving
        # to Mauro Zarate's stats, Cristian Ansaldi to Cristian Zapata's,
        # Habib Diarra to Habib Diallo's - none of these pairs share a
        # single real name token, but a shared first name or generically-
        # similar surname plus a same-club coincidence was enough to clear
        # 0.6 *in both directions*, since raw character overlap doesn't
        # know a name is made of discrete tokens. Every legitimate fuzzy
        # match found instead has the shorter name's tokens wholly contained
        # in the longer name's - a nickname/shortened form (Bremer subset of
        # "Gleison Bremer", Alisson subset of "Alisson Becker") or a fuller
        # name with an extra middle/second surname (Kerim Frei subset of
        # "Kerim Frei Koyunlu", Fode Ballo subset of "Fode Ballo-Toure") -
        # never a same-length pair that merely happens to share one token.
        # Requiring that containment instead of a raw ratio rejects every
        # confirmed-wrong collision above while keeping every confirmed-real
        # one.
        query_tokens = set(norm_name.split())
        close = [c for c in close if query_tokens <= set(c.split()) or set(c.split()) <= query_tokens]
        candidates = table[table["norm_name"].isin(close)]
        if candidates.empty:
            continue
        club_hits = candidates[candidates["fotmob_team"].map(lambda t: club_names_match(club_norm, normalize_club(t), comp_id))]
        if len(club_hits) >= 1:
            return club_hits.iloc[0]["fotmob_id"]
    return None


def aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables):
    """Combine one player's per-season rows into one tenure-level record: minutes-weighted average for RATE_STATS, summed for SUM_STATS. A season the id doesn't appear in at all (e.g. injured that whole season) is skipped, not treated as zero."""
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
    """Full fetch+stitch for one league: every season, every transfer's tenure aggregated. Returns a DataFrame with one row per successfully-matched transfer."""
    league_id, label = LEAGUE_MAP[comp_id]
    print(f"=== {comp_id} - {label} (fotmob id {league_id}) - {len(transfers):,} candidate transfers ===")

    season_tables = {season: load_or_fetch_season(client, comp_id, league_id, season) for season in ALL_SEASON_LABELS}
    covered = [s for s, t in season_tables.items() if t is not None]
    print(f"  Seasons with coverage: {covered[0] if covered else 'NONE'} .. {covered[-1] if covered else 'NONE'} ({len(covered)} seasons)")

    results = []
    for _, t in transfers.iterrows():
        tenure_seasons = [s for s in seasons_overlapping(t["transfer_date"], t["tenure_end"]) if s in covered]
        if not tenure_seasons:
            continue

        is_intra_league = t["from_domestic_competition_id"] == comp_id
        is_midseason_join = t["transfer_date"].month not in (6, 7, 8)
        join_season = season_label_for_date(t["transfer_date"])
        join_contaminated = join_season if (is_intra_league and is_midseason_join and join_season in tenure_seasons) else None

        # Mirror image at the other end of the tenure - see NEXT_COMPETITION.
        next_comp = NEXT_COMPETITION.get((t["player_id"], t["tenure_end"]))
        is_midseason_leave = t["tenure_end"].month not in (6, 7, 8)
        leave_season = season_label_for_date(t["tenure_end"])
        leave_contaminated = leave_season if (next_comp == comp_id and is_midseason_leave and leave_season in tenure_seasons) else None

        agg_seasons = [s for s in tenure_seasons if s not in (join_contaminated, leave_contaminated)]

        fotmob_id = find_fotmob_id(t, season_tables, list(reversed(tenure_seasons)), comp_id)
        if fotmob_id is None:
            continue
        agg = aggregate_tenure_stats(fotmob_id, agg_seasons, season_tables)
        if agg is None:
            continue

        results.append({
            "player_id": t["player_id"], "transfer_date": t["transfer_date"],
            "to_club_name": t["to_club_name"], "competition_id": comp_id, **agg,
        })

    results_df = pd.DataFrame(results)
    print(f"  Matched {len(results_df)} / {len(transfers)} transfers")
    return results_df


def load_tenure_windows():
    """
    Every tenure window that needs FotMob stats: both permanent transfers
    (data/transfers_processed.csv) and loan spells (data/loans_processed.csv),
    unioned into one frame. run_league() only ever reads transfer_date/
    tenure_end/from_domestic_competition_id/to_club_name/name/player_id -
    present in both source files - so the two are safe to concatenate and
    run through the exact same fetch/match/stitch pipeline; a loan and a
    permanent transfer can never collide on (player_id, transfer_date) since
    each is a distinct real-world event. Reading the *committed* transfers/
    loans CSVs here isn't circular: their tenure/position/club columns come
    from the base Transfermarkt pipeline and don't depend on FotMob data, so
    this can run before build_dataset.py regenerates them with fresh FotMob
    columns.
    """
    transfers = pd.read_csv(TRANSFERS_PATH, parse_dates=["transfer_date"])
    loans = pd.read_csv(LOANS_PATH, parse_dates=["transfer_date"])
    combined = pd.concat([transfers, loans], ignore_index=True)
    combined["tenure_end"] = combined["transfer_date"] + pd.to_timedelta(combined["tenure_days"], unit="D")
    return combined[combined["transfer_date"] >= f"{FIRST_SEASON_YEAR}-08-01"]


def main():
    """Fetch every league in LEAGUE_MAP, stitch every permanent-transfer and loan tenure's stats, write the combined result to data/raw/fotmob_stats_cache.csv for build_dataset.py to consume."""
    df = load_tenure_windows()

    all_results = []
    with httpx.Client(timeout=30) as client:
        for comp_id in LEAGUE_MAP:
            league_rows = df[df["to_domestic_competition_id"] == comp_id].copy()
            if league_rows.empty:
                continue
            results_df = run_league(client, comp_id, league_rows)
            if not results_df.empty:
                all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
