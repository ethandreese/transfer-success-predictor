"""
Build a per-transfer training dataset from the raw Transfermarkt CSVs
(dcaribou/transfermarkt-datasets, mirrored on Kaggle as davidcariboo/player-scores).

For every transfer we compute:
  - pre-transfer features: age, position, physical attributes, fee, market
    value, and performance in the player's final year at the old club
  - a post-hoc "success score" (0-100) blending performance level/delta,
    market value growth, playing time, value-for-money, and (where FotMob
    data is available - see load_fotmob_stats) a defensive/technical
    contribution signal, all measured over the player's *entire tenure* at
    the new club (from the transfer until their next departure, or "now" if
    they're still there) rather than a fixed first-year window. A fixed
    window either penalizes slow starters who took time to adapt, or
    misses a player who started hot and faded once the honeymoon period
    ended.

Only the pre-transfer features are used as model inputs; the success score
is the training label.

Loan spells are excluded entirely (see load_transfers/load_transfer_types):
they aren't a permanent-transfer decision, so scoring them the same way
would judge a temporary loan spell as if a club had chosen to buy the
player outright. Detecting them requires data/raw/transfer_types_cache.csv
(built by scripts/fetch_transfer_types.py) since the packaged dataset itself
can't tell a loan from a free transfer - both parse to a fee of 0.

The base Transfermarkt dataset has no column at all for defense-specific
output (tackles, clean sheets, saves), so scripts/fetch_fotmob_stats.py
(optional, like fetch_transfer_types.py) separately pulls FotMob's season
stat leaderboards for the 23 leagues that appear as a transfer destination
and stitches each tenure's stats into data/raw/fotmob_stats_cache.csv -
see compute_fotmob_component_pcts for how that becomes four real score
components (rating, attacking, defensive, possession) instead of leaving
defenders and goalkeepers to be judged almost entirely on market value and
playing time.
"""
import json
import os
import numpy as np
import pandas as pd

SCORE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "score_weights.json")
with open(SCORE_WEIGHTS_PATH) as f:
    _score_weights_raw = json.load(f)
RESALE_WEIGHT_CURVE = _score_weights_raw["resale_weight_curve"]
POSITION_WEIGHTS = {
    k: v for k, v in _score_weights_raw.items()
    if not k.startswith("_") and k != "resale_weight_curve"
}

LOAN_SCORE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_score_weights.json")
with open(LOAN_SCORE_WEIGHTS_PATH) as f:
    _loan_score_weights_raw = json.load(f)
LOAN_POSITION_WEIGHTS = {k: v for k, v in _loan_score_weights_raw.items() if not k.startswith("_")}

RAW_DIR = os.environ.get(
    "TRANSFERMARKT_RAW_DIR",
    os.path.expanduser(
        "~/.cache/kagglehub/datasets/davidcariboo/player-scores/versions/677"
    ),
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")
LOANS_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loans_processed.csv")
TRANSFER_TYPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transfer_types_cache.csv")
FOTMOB_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fotmob_stats_cache.csv")

PRE_WINDOW_DAYS = 365
MIN_APPS_PER_WINDOW = 10
MIN_DATE = pd.Timestamp("2013-01-01")
MAX_DATE = pd.Timestamp.today().normalize()
REFERENCE_NOW = pd.Timestamp.today().normalize()


def load_transfer_types():
    """
    Load the transfer_type cache built by scripts/fetch_transfer_types.py,
    which re-fetches each player's real transfer history from transfermarkt's
    live API and classifies each move as paid/free/loan/unknown from its raw
    fee text - a distinction the packaged transfers.csv doesn't have, since
    its upstream ETL flattens any non-numeric fee (including "loan
    transfer"/"End of loan"/"Loan fee: EUR X") down to a plain 0, identical
    to a genuine free transfer. fee_raw (the original text) is kept too, so
    load_loan_spells() can tell a loan-out ("loan transfer") from the
    "End of loan" bookend that closes it. Returns an empty, all-"unknown"
    frame with the right columns if the cache hasn't been built yet, so
    callers degrade to their old (loan-unaware) behavior rather than failing.
    """
    if not os.path.exists(TRANSFER_TYPES_PATH):
        print(f"  (no transfer-type cache at {TRANSFER_TYPES_PATH} - run scripts/fetch_transfer_types.py to enable loan detection; continuing without it)")
        return pd.DataFrame(columns=["player_id", "transfer_date", "from_club_id", "to_club_id", "transfer_type", "fee_raw"])
    types = pd.read_csv(
        TRANSFER_TYPES_PATH,
        usecols=["player_id", "transfer_date", "from_club_id", "to_club_id", "transfer_type", "fee_raw"],
    )
    # Parsed separately (not via read_csv's parse_dates) because the live API
    # occasionally returns the sentinel "0000-00-00" for a transfer's date -
    # a known artifact the upstream dcaribou pipeline filters out too - and
    # even one such value makes parse_dates silently leave the whole column
    # as strings instead of raising. errors="coerce" turns just those rows
    # into NaT, which can never join onto anything (transfers.csv itself has
    # no such placeholder), so they're dropped here rather than causing a
    # dtype mismatch downstream.
    types["transfer_date"] = pd.to_datetime(types["transfer_date"], errors="coerce")
    types = types.dropna(subset=["transfer_date"])
    # A resumed fetch run could in principle append a player's rows twice;
    # de-dupe defensively on the natural key so the join below can't fan out.
    return types.drop_duplicates(subset=["player_id", "transfer_date", "from_club_id", "to_club_id"])


# The raw FotMob columns compute_fotmob_component_pcts() actually uses -
# see that function for which bucket (rating/attacking/defensive/possession)
# each one feeds. fetch_fotmob_stats.py's cache already has all of these
# (and a few more, e.g. fotmob_defensive_contributions - FotMob's own
# pre-bundled tackles+interceptions+clearances+recoveries composite, not
# read here since the four sub-stats it's built from are used directly
# instead, for more resolution than one blended number gives).
FOTMOB_RAW_COLS = [
    "fotmob_rating", "fotmob_total_minutes", "fotmob_seasons_used", "fotmob_total_matches",
    "fotmob_goals_per_90", "fotmob_expected_goals_per_90", "fotmob_expected_assists_per_90",
    "fotmob_won_contest", "fotmob_big_chance_created", "fotmob_total_att_assist",
    "fotmob_total_tackle", "fotmob_interception", "fotmob_effective_clearance", "fotmob_ball_recovery",
    "fotmob_accurate_pass", "fotmob_saves", "fotmob__save_percentage", "fotmob_goals_conceded",
]


def load_fotmob_stats():
    """
    Load the per-transfer FotMob tenure stats built by
    scripts/fetch_fotmob_stats.py (season-leaderboard data - rating,
    defensive actions/90, passes/90, saves/90, etc. - stitched across
    whatever seasons each tenure spans; see that script for the full
    fetch/match/stitch reasoning). Only covers permanent transfers into the
    23 leagues in LEAGUE_MAP there, and only from whatever season FotMob's
    own coverage happens to start for that specific league (nothing before
    2013/2014 anywhere, later still for most leagues) - so plenty of
    real transfers legitimately have no row here. Returns an empty frame
    with the right columns if the cache hasn't been built yet, so callers
    degrade to treating every transfer as missing FotMob data rather than
    failing.
    """
    if not os.path.exists(FOTMOB_STATS_PATH):
        print(f"  (no FotMob stats cache at {FOTMOB_STATS_PATH} - run scripts/fetch_fotmob_stats.py to enable the defensive/technical component; continuing without it)")
        return pd.DataFrame(columns=["player_id", "transfer_date"] + FOTMOB_RAW_COLS)
    stats = pd.read_csv(
        FOTMOB_STATS_PATH,
        usecols=["player_id", "transfer_date"] + FOTMOB_RAW_COLS,
        parse_dates=["transfer_date"],
    )
    # One row per (player_id, transfer_date) by construction (each transfer
    # is fetched/matched once) - de-duped defensively anyway so the merge
    # in main() can't fan out if fetch_fotmob_stats.py is ever re-run in a
    # way that appends rather than overwrites.
    return stats.drop_duplicates(subset=["player_id", "transfer_date"])


def _load_raw_candidate_transfers():
    """
    Shared first step for load_transfers() and load_loan_spells(): read
    transfers.csv, filter to the date range, and drop no-op moves
    (from_club == to_club). Does not yet know about transfer_type or
    tenure_end - both loaders need those computed differently (see each
    function's docstring).
    """
    df = pd.read_csv(
        os.path.join(RAW_DIR, "transfers.csv"),
        usecols=[
            "player_id", "transfer_date", "from_club_id", "to_club_id",
            "from_club_name", "to_club_name", "transfer_fee",
            "market_value_in_eur", "player_name",
        ],
        parse_dates=["transfer_date"],
    )
    df = df.dropna(subset=["transfer_date", "from_club_id", "to_club_id"])
    df = df[(df["transfer_date"] >= MIN_DATE) & (df["transfer_date"] <= MAX_DATE)]
    df = df[df["from_club_id"] != df["to_club_id"]]
    df["from_club_id"] = df["from_club_id"].astype(int)
    df["to_club_id"] = df["to_club_id"].astype(int)
    return df


def load_transfers():
    """
    Load candidate transfers and drop loan spells (see load_transfer_types) -
    they aren't a permanent-transfer decision, so scoring them the same way
    would judge a temporary loan as if a club had chosen to buy the player
    outright (see load_loan_spells for how loans are scored instead). Then
    compute two per-transfer fields that need the player's *next* (non-loan)
    transfer to work out:
      - tenure_end: the date the player left the new club (their next
        transfer's date), or REFERENCE_NOW if they haven't moved since
      - next_transfer_fee: the fee received for that next transfer (NaN if
        there isn't one yet), used later for the resale-profit component

    Dropping loan rows before these shift()-based lookups matters, not just
    for the loan transfer itself: it makes tenure_end and next_transfer_fee
    for the transfers *around* a loan skip straight past it too - e.g. for
    ClubA -> ClubB -> (loan to ClubC) -> ClubB -> ClubD, the ClubA -> ClubB
    transfer's tenure now correctly runs through to the ClubD sale (the loan
    spell folds back into "still registered at ClubB", which is what
    actually happened) instead of getting cut short at the loan-out date,
    and next_transfer_fee for it correctly reflects the ClubD sale instead
    of the loan's unfeed fee.

    Returns a DataFrame with one row per transfer and a "transfer_idx"
    column used as the join key by every other compute_* function in this
    module.
    """
    df = _load_raw_candidate_transfers()

    transfer_types = load_transfer_types().drop(columns=["fee_raw"])
    df = df.merge(transfer_types, on=["player_id", "transfer_date", "from_club_id", "to_club_id"], how="left")
    df["transfer_type"] = df["transfer_type"].fillna("unknown")
    print(f"  transfer_type breakdown: {df['transfer_type'].value_counts().to_dict()}")
    n_loans = int((df["transfer_type"] == "loan").sum())
    df = df[df["transfer_type"] != "loan"].drop(columns=["transfer_type"])
    print(f"  Dropped {n_loans:,} loan transfers (not a permanent-transfer decision)")

    df = df.reset_index(drop=True)
    df["transfer_idx"] = df.index

    # Tenure at the new club runs until the player's next (non-loan)
    # transfer, or until "now" if they haven't moved again since.
    df = df.sort_values(["player_id", "transfer_date"])
    df["tenure_end"] = df.groupby("player_id")["transfer_date"].shift(-1)
    df["next_transfer_fee"] = df.groupby("player_id")["transfer_fee"].shift(-1)
    df["tenure_end"] = df["tenure_end"].fillna(REFERENCE_NOW)
    df = df.sort_values("transfer_idx").reset_index(drop=True)
    return df


MIN_LOAN_TENURE_DAYS = 14


def load_loan_spells():
    """
    Load candidate transfers and keep only the loan-out leg of each loan
    spell - rows whose real fee text (from transfer_types_cache.csv) is
    "loan transfer" or "Loan fee: ...", not the "End of loan" bookend that
    closes it back out. tenure_end is the date of the player's very next
    transfer of any type (almost always that matching "End of loan" return,
    or REFERENCE_NOW if the loan is still ongoing) - computed the same way
    load_transfers() computes it for permanent moves, so a loan spell's
    windows can reuse the exact same compute_* helpers as a permanent
    transfer's. Spells shorter than MIN_LOAN_TENURE_DAYS are dropped as
    likely data artifacts (same-day duplicate entries etc).

    Unlike load_transfers(), rows are NOT dropped here for having too few
    post-loan appearances (see prepare_loans) - a loan spell where the
    player barely featured isn't missing data, it's the actual outcome
    (benched, frozen out, injured throughout) that the Loans tab exists to
    surface, not something to filter away.
    """
    df = _load_raw_candidate_transfers()

    types = load_transfer_types()
    df = df.merge(types, on=["player_id", "transfer_date", "from_club_id", "to_club_id"], how="left")
    df["transfer_type"] = df["transfer_type"].fillna("unknown")
    df["fee_raw"] = df["fee_raw"].fillna("")

    df = df.sort_values(["player_id", "transfer_date"])
    df["tenure_end"] = df.groupby("player_id")["transfer_date"].shift(-1)
    df["tenure_end"] = df["tenure_end"].fillna(REFERENCE_NOW)

    is_loan_start = df["transfer_type"].eq("loan") & ~df["fee_raw"].str.lower().str.contains("end of loan")
    loans = df[is_loan_start].drop(columns=["transfer_type", "fee_raw"]).copy()
    loans = loans[(loans["tenure_end"] - loans["transfer_date"]).dt.days >= MIN_LOAN_TENURE_DAYS]
    loans = loans.reset_index(drop=True)
    loans["transfer_idx"] = loans.index
    return loans


def load_players():
    """Load players.csv: one row per player with position, physical attributes, and current club/value."""
    df = pd.read_csv(
        os.path.join(RAW_DIR, "players.csv"),
        usecols=[
            "player_id", "name", "date_of_birth", "position",
            "sub_position", "foot", "height_in_cm",
            "current_club_id", "market_value_in_eur",
        ],
        parse_dates=["date_of_birth"],
    )
    return df


def load_clubs():
    """
    Load clubs.csv: just club_id and domestic_competition_id (the league).
    total_market_value in clubs.csv is unpopulated in this dataset version,
    so club quality is proxied instead by compute_club_value_proxy() below,
    summed from each club's current squad value in players.csv.
    """
    df = pd.read_csv(
        os.path.join(RAW_DIR, "clubs.csv"),
        usecols=["club_id", "domestic_competition_id"],
    )
    return df


def compute_club_value_proxy(players):
    """
    Approximate each club's overall squad strength as the sum of its
    current players' market values. Used as a stand-in for "how big/rich is
    this club" since clubs.csv's own total_market_value column is empty.
    """
    proxy = (
        players.dropna(subset=["current_club_id"])
        .groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .rename("club_value_proxy")
    )
    proxy.index = proxy.index.astype(int)
    return proxy


def load_appearances():
    """Load appearances.csv: one row per (player, game) with goals/assists/minutes and the competition it was in."""
    df = pd.read_csv(
        os.path.join(RAW_DIR, "appearances.csv"),
        usecols=[
            "player_id", "player_club_id", "competition_id", "date",
            "goals", "assists", "minutes_played",
        ],
        parse_dates=["date"],
    )
    return df


def load_team_games():
    """
    One row per (club, game) across all competitions the club played in,
    used to work out what share of the team's actual games a player
    appeared in during their tenure - not just a raw appearance count.
    """
    games = pd.read_csv(
        os.path.join(RAW_DIR, "games.csv"), usecols=["game_id", "date"], parse_dates=["date"],
    )
    club_games = pd.read_csv(os.path.join(RAW_DIR, "club_games.csv"), usecols=["game_id", "club_id"])
    return club_games.merge(games, on="game_id", how="left")


def compute_team_games_in_window(transfers, team_games):
    """
    For each transfer, how many games the *new club* played from the
    transfer date through the end of the player's tenure there (all
    competitions - league, domestic cup, continental). This is the
    denominator for "percent of available games played" - see
    compute_windowed_appearance_stats for the matching numerator.
    """
    merged = team_games.merge(
        transfers[["transfer_idx", "to_club_id", "transfer_date", "tenure_end"]],
        left_on="club_id", right_on="to_club_id", how="inner",
    )
    in_window = (
        (merged["date"] > merged["transfer_date"]) & (merged["date"] <= merged["tenure_end"])
    )
    counts = merged[in_window].groupby("transfer_idx").size().rename("team_games_in_tenure")
    return counts


MIN_LEAGUE_BASELINE_MINUTES = 5000


def compute_league_position_baselines(appearances, players):
    """
    For each (competition_id, position), the average goal contributions per
    90 minutes across *all* appearances in that league - not just the
    filtered transfer set - used to judge a player's output against how
    hard it actually is to contribute goals in that specific league/role,
    rather than against the whole dataset regardless of league. A decline
    in raw output after moving into a tougher-to-score-in league (e.g.
    Bundesliga -> Premier League, which averages ~12% fewer goal
    contributions per 90 for attackers) shouldn't be read the same as a
    decline moving into an easier one.
    """
    merged = appearances.merge(players[["player_id", "position"]], on="player_id", how="left")
    agg = merged.groupby(["competition_id", "position"]).agg(
        goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum"),
    )
    agg = agg[agg["minutes"] >= MIN_LEAGUE_BASELINE_MINUTES]
    agg["ga_p90"] = (agg["goals"] + agg["assists"]) / agg["minutes"].clip(lower=1) * 90
    league_position_baseline = agg["ga_p90"]

    position_agg = merged.groupby("position").agg(
        goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum"),
    )
    position_fallback = (position_agg["goals"] + position_agg["assists"]) / position_agg["minutes"].clip(lower=1) * 90

    return league_position_baseline, position_fallback


def lookup_league_baseline(league_ids, positions, baseline, fallback):
    """
    Vectorized lookup of the (league, position) -> ga_p90 baseline for a
    whole column of transfers at once. Falls back to the position's overall
    average (from compute_league_position_baselines) for any (league,
    position) pair with too little data to have its own baseline, and to
    the average of all positions' fallbacks as a last resort.
    """
    keys = list(zip(league_ids, positions))
    return pd.Series(
        [baseline.get(k, fallback.get(k[1], fallback.mean())) for k in keys],
        index=league_ids.index,
    )


LEAGUE_BASELINES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "league_baselines.csv")
FALLBACK_COMPETITION_ID = "_default"


def save_league_baselines(league_position_baseline, position_fallback):
    """
    Persist the (league, position) -> goal-contributions/90 baselines as a
    small committed CSV, so app/main.py can compute a league-adjusted
    performance feature for *live predictions* using the exact same
    baselines the label was built from, instead of only being available
    inside this script. Fallback rows use competition_id="_default".
    """
    real = league_position_baseline.reset_index()
    real.columns = ["competition_id", "position", "ga_p90_baseline"]
    fallback = position_fallback.reset_index()
    fallback.columns = ["position", "ga_p90_baseline"]
    fallback.insert(0, "competition_id", FALLBACK_COMPETITION_ID)
    out = pd.concat([real, fallback], ignore_index=True)
    out.to_csv(LEAGUE_BASELINES_PATH, index=False)
    print(f"Wrote {len(out):,} league/position baselines to {LEAGUE_BASELINES_PATH}")


def load_valuations():
    """Load player_valuations.csv: the full market-value history for every player, sorted by date (needed for merge_asof)."""
    df = pd.read_csv(
        os.path.join(RAW_DIR, "player_valuations.csv"),
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    )
    return df.sort_values("date")


def compute_windowed_appearance_stats(transfers, appearances):
    """
    For every transfer, sum up appearances/minutes/goals/assists in two
    windows:
      - "pre": the PRE_WINDOW_DAYS (1 year) before the transfer, at the OLD
        club - the player's form walking into the move
      - "post": from the transfer date through tenure_end, at the NEW club
        - their entire stint there, not just a fixed first year

    Returns (pre, post): two DataFrames indexed by transfer_idx with
    columns like pre_apps/pre_minutes/pre_goals/pre_assists (and the post_
    equivalents), ready to .join() onto the main transfers DataFrame.
    """
    merged = appearances.merge(
        transfers[["transfer_idx", "player_id", "transfer_date", "tenure_end", "from_club_id", "to_club_id"]],
        on="player_id",
        how="inner",
    )

    pre_window = pd.Timedelta(days=PRE_WINDOW_DAYS)

    pre_mask = (
        (merged["player_club_id"] == merged["from_club_id"])
        & (merged["date"] < merged["transfer_date"])
        & (merged["date"] >= merged["transfer_date"] - pre_window)
    )
    # Post window spans the player's *entire tenure* at the new club, not a
    # fixed first year - see module docstring.
    post_mask = (
        (merged["player_club_id"] == merged["to_club_id"])
        & (merged["date"] > merged["transfer_date"])
        & (merged["date"] <= merged["tenure_end"])
    )

    def agg(mask, prefix):
        """Sum apps/minutes/goals/assists per transfer for the rows selected by `mask`, prefixing column names (e.g. "pre_" or "post_")."""
        sub = merged[mask].groupby("transfer_idx").agg(
            apps=("date", "count"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
        )
        sub.columns = [f"{prefix}_{c}" for c in sub.columns]
        return sub

    pre = agg(pre_mask, "pre")
    post = agg(post_mask, "post")
    return pre, post


def nearest_valuation(transfers, valuations, on_col, direction, tolerance_days, suffix):
    """
    Find each player's market value nearest to a given date column
    (`on_col`, e.g. "transfer_date" or "tenure_end") via a time-based
    as-of join. `direction` controls whether "nearest" looks backward,
    forward, or either way; `tolerance_days` caps how far away a match can
    be before it's treated as missing. Returns a Series named
    "value_{suffix}", indexed by transfer_idx.
    """
    left = transfers[["transfer_idx", "player_id", on_col]].sort_values(on_col)
    right = valuations.rename(columns={"market_value_in_eur": f"value_{suffix}"})
    out = pd.merge_asof(
        left,
        right,
        left_on=on_col,
        right_on="date",
        by="player_id",
        direction=direction,
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return out.set_index("transfer_idx")[f"value_{suffix}"]


def compute_peak_valuation(transfers, valuations):
    """
    Highest market value reached at any point during the tenure, not just
    the value near the end of it. A long, valuable career naturally ends
    with a lower market value than its peak simply because of age - the
    end-of-tenure snapshot alone would read a hugely successful long
    tenure as a decline. (E.g. Heung-min Son joined Tottenham valued at
    ~25m, peaked at 90m mid-tenure, and was worth ~20m when he eventually
    left a decade later - the peak, not the exit value, is what actually
    reflects the value he added.)
    """
    merged = valuations.merge(
        transfers[["transfer_idx", "player_id", "transfer_date", "tenure_end"]],
        on="player_id", how="inner",
    )
    in_window = (merged["date"] > merged["transfer_date"]) & (merged["date"] <= merged["tenure_end"])
    return merged[in_window].groupby("transfer_idx")["market_value_in_eur"].max().rename("value_peak")


def percentile_rank(series):
    """Rank each value's position in the series as a 0-100 percentile (100 = highest)."""
    return series.rank(pct=True) * 100


# Which raw FotMob stats feed each of the four buckets
# compute_fotmob_component_pcts() computes - see that function's docstring
# for why they're kept separate instead of blended into one number.
FOTMOB_ATTACKING_STATS = [
    "fotmob_goals_per_90", "fotmob_expected_goals_per_90", "fotmob_expected_assists_per_90",
    "fotmob_chances_created_p90", "fotmob_big_chance_created_p90",
]
FOTMOB_DEFENSIVE_OUTFIELD_STATS = ["fotmob_total_tackle", "fotmob_interception", "fotmob_effective_clearance", "fotmob_ball_recovery"]
FOTMOB_DEFENSIVE_GK_STATS = ["fotmob_saves", "fotmob__save_percentage", "fotmob_goals_conceded_inv"]
# Dribbles (won_contest) are ball-carrying/retention under pressure - a
# possession skill, not attacking threat - grouped here rather than with
# attacking, matching how e.g. FBref categorizes take-ons under
# "Possession" rather than "Shooting"/"Passing".
FOTMOB_POSSESSION_STATS = ["fotmob_accurate_pass", "fotmob_won_contest"]
FOTMOB_COMPONENTS = ["rating", "attacking", "defensive", "possession"]


def compute_fotmob_component_pcts(df):
    """
    Four separate performance signals from FotMob's stitched tenure stats
    (see load_fotmob_stats), filling the gap the base Transfermarkt dataset
    has no column for at all: tackles, clean sheets, saves, or any other
    defense-specific output. Kept as four components - rating, attacking,
    defensive, possession - rather than blended into one, so a player's
    actual profile survives into the score: an attack-minded fullback and a
    purely defensive one could land on the same *blended* number despite
    having very different games, which defeats the point of pulling this
    data in the first place.

    Each is percentile-ranked *within position group*, like perf_level_pct,
    then (for the three multi-stat buckets) averaged across whichever of
    that bucket's raw stats are actually available for the row, so one
    missing sub-stat doesn't zero out the whole bucket. "Defensive" uses a
    genuinely different stat set by position - saves/save%/goals-conceded
    for goalkeepers, tackles/interceptions/clearances/recoveries for
    everyone else, since neither set means anything for the other position
    (confirmed empirically: most goalkeepers have no total_tackle data at
    all, and vice versa for saves). "Attacking" and "possession" are ranked
    the same way for every position without a manual split, matching how
    perf_level_pct already handles goalkeepers - a keeper's real (if
    usually low) attacking numbers just rank low within the goalkeeper
    group, which is harmless since attacking's weight is 0 for goalkeepers
    anyway (data/score_weights.json).

    Returns (rating_pct, attacking_pct, defensive_pct, possession_pct),
    each a Series aligned to df's index; NaN wherever a row has no FotMob
    data for that specific bucket at all - main() treats that the same way
    as unknown resale_profit: the component's weight is dropped for that
    row and the rest renormalized, not filled in with a fabricated neutral
    value.
    """
    is_gk = df["position"] == "Goalkeeper"

    df = df.copy()
    minutes_per_90 = (df["fotmob_total_minutes"] / 90).clip(lower=1)
    df["fotmob_chances_created_p90"] = df["fotmob_total_att_assist"] / minutes_per_90
    df["fotmob_big_chance_created_p90"] = df["fotmob_big_chance_created"] / minutes_per_90
    df["fotmob_goals_conceded_inv"] = -df["fotmob_goals_conceded"]  # fewer conceded is better - negate before ranking so higher percentile = better

    all_stats = (
        {"fotmob_rating"} | set(FOTMOB_ATTACKING_STATS)
        | set(FOTMOB_DEFENSIVE_OUTFIELD_STATS) | set(FOTMOB_DEFENSIVE_GK_STATS) | set(FOTMOB_POSSESSION_STATS)
    )
    pct = {stat: df.groupby("position")[stat].rank(pct=True) * 100 for stat in all_stats}

    def avg_pct(stats):
        return pd.concat([pct[s] for s in stats], axis=1).mean(axis=1, skipna=True)

    rating_pct = pct["fotmob_rating"]
    attacking_pct = avg_pct(FOTMOB_ATTACKING_STATS)
    possession_pct = avg_pct(FOTMOB_POSSESSION_STATS)

    defensive_pct = pd.Series(np.nan, index=df.index)
    defensive_pct.loc[~is_gk] = avg_pct(FOTMOB_DEFENSIVE_OUTFIELD_STATS).loc[~is_gk]
    defensive_pct.loc[is_gk] = avg_pct(FOTMOB_DEFENSIVE_GK_STATS).loc[is_gk]

    return rating_pct, attacking_pct, defensive_pct, possession_pct


def attach_fotmob_components(transfers_df, loans_df):
    """
    Merge FotMob raw stats onto both permanent transfers and loans, then
    rank rating/attacking/defensive/possession against the COMBINED
    population instead of each type separately (previously each called
    compute_fotmob_component_pcts on its own df in isolation).

    This is deliberately different from value_growth/playing_time, which
    stay ranked within their own population (see finish_loan_dataset's
    docstring: a loan's much shorter window makes raw appearance counts
    and value growth genuinely incomparable in scale to a permanent
    tenure's). The FotMob components don't have that problem - they're
    already per-90 rates (or, for rating, a plain average), so a loan
    spell and a permanent tenure with identical on-pitch output should
    land on the same percentile, not two different ones just because of
    which population happened to rank them. Combining also gives smaller
    slices (goalkeepers especially) a bigger, more stable reference
    population than either pool alone.

    Returns (transfers_df, loans_df), each with rating_pct/attacking_pct/
    defensive_pct/possession_pct/has_*_data columns attached, row order
    and index otherwise unchanged.
    """
    fotmob_stats = load_fotmob_stats()
    transfers_df = transfers_df.merge(fotmob_stats, on=["player_id", "transfer_date"], how="left")
    loans_df = loans_df.merge(fotmob_stats, on=["player_id", "transfer_date"], how="left")

    n_transfers = len(transfers_df)
    combined = pd.concat([transfers_df, loans_df], ignore_index=True)
    combined["rating_pct"], combined["attacking_pct"], combined["defensive_pct"], combined["possession_pct"] = (
        compute_fotmob_component_pcts(combined)
    )
    for component in FOTMOB_COMPONENTS:
        combined[f"has_{component}_data"] = combined[f"{component}_pct"].notna()
    combined["has_fotmob_data"] = combined[[f"has_{c}_data" for c in FOTMOB_COMPONENTS]].any(axis=1)

    transfers_df = combined.iloc[:n_transfers].reset_index(drop=True)
    loans_df = combined.iloc[n_transfers:].reset_index(drop=True)

    print(f"  Permanent transfers: {transfers_df['has_fotmob_data'].sum():,} / {len(transfers_df):,} have a usable score in at least one FotMob component")
    print(f"  Loans: {loans_df['has_fotmob_data'].sum():,} / {len(loans_df):,} have a usable score in at least one FotMob component")
    for component in FOTMOB_COMPONENTS:
        print(f"    {component}: {100 * combined[f'has_{component}_data'].mean():.0f}% of the combined population")

    return transfers_df, loans_df


def compute_resale_weight(tenure_years):
    """
    How much resale profit should count, as a function of tenure length -
    see data/score_weights.json's resale_weight_curve. A quick flip weights
    the resale outcome heavily (that's often the point of the deal); a long
    career barely moves regardless of the eventual sale price, since the
    club already extracted years of on-pitch value from the player.
    """
    c = RESALE_WEIGHT_CURVE
    return c["min"] + (c["max"] - c["min"]) * np.exp(-tenure_years / c["decay_years"])


def compute_expected_post_performance(df):
    """
    Fit post_ga_p90_vs_league ~ pre_ga_p90_vs_league per position (simple
    linear regression) and return the model's expected post-transfer level
    for each row. Used so "performance change" measures over/underperforming
    *what's statistically typical given how strong they already were*,
    rather than the raw before-after difference.

    A raw difference unfairly penalizes players who were already near the
    top: someone who was 2.6x their league's average has much more room to
    fall than to rise, so almost any real-world outcome short of getting
    even better reads as "decline" even when they're still elite. Everyone
    else in the dataset who started that high shows a similar pullback too
    (regression to the mean) - the fit line captures that normal pullback,
    so a player who pulls back by exactly the expected amount now scores
    neutrally instead of being marked down, and one who falls much further
    than that (a real bust, not just ceiling effects) still scores badly.
    """
    expected = pd.Series(index=df.index, dtype=float)
    for position, sub in df.groupby("position"):
        slope, intercept = np.polyfit(sub["pre_ga_p90_vs_league"], sub["post_ga_p90_vs_league"], 1)
        expected.loc[sub.index] = intercept + slope * sub["pre_ga_p90_vs_league"]
    return expected


def prepare_loans(players, clubs, appearances, valuations, team_games):
    """
    First half of the loan-spell pipeline - same windowed appearance/
    valuation machinery as main()'s permanent-transfer pipeline (every
    compute_* helper here only cares about a transfer-shaped frame -
    transfer_idx/player_id/transfer_date/tenure_end/from_club_id/
    to_club_id - not what kind of move it represents, so they're reused
    as-is on load_loan_spells()'s output instead of load_transfers()'s).
    Stops right where main() does for permanent transfers - age/value
    validity filtered, ready for a FotMob merge - rather than attaching
    FotMob stats and computing loan_success_score itself, so
    attach_fotmob_components() can rank loans against permanent transfers
    jointly before finish_loan_dataset() computes the loan-specific
    league-adjusted performance features and the final score. Split out
    from what used to be one build_loan_dataset() function for exactly
    that reason.
    """
    print("Loading loan spells...")
    loans = load_loan_spells()
    print(f"{len(loans):,} candidate loan spells")

    print("Computing pre/post appearance windows for loans...")
    pre, post = compute_windowed_appearance_stats(loans, appearances)
    team_games_in_tenure = compute_team_games_in_window(loans, team_games)

    df = loans.set_index("transfer_idx").join(pre).join(post).join(team_games_in_tenure)
    fill_cols = ["pre_apps", "post_apps", "pre_minutes", "post_minutes", "pre_goals", "post_goals",
                 "pre_assists", "post_assists", "team_games_in_tenure"]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Only the PRE window needs a minimum-appearances bar (establishing who
    # this player was walking into the loan) - the post/loan window
    # deliberately has none, see load_loan_spells's docstring.
    before = len(df)
    df = df[df["pre_apps"] >= MIN_APPS_PER_WINDOW]
    print(f"Kept {len(df):,} / {before:,} loan spells with >= {MIN_APPS_PER_WINDOW} pre-loan apps")

    df["pre_ga_p90"] = (df["pre_goals"] + df["pre_assists"]) / df["pre_minutes"].clip(lower=1) * 90
    df["post_ga_p90"] = (df["post_goals"] + df["post_assists"]) / df["post_minutes"].clip(lower=1) * 90
    df["pre_goals_p90"] = df["pre_goals"] / df["pre_minutes"].clip(lower=1) * 90
    df["pre_mins_per_app"] = df["pre_minutes"] / df["pre_apps"].clip(lower=1)

    print("Attaching market valuations for loans...")
    df = df.reset_index()
    pre_val = nearest_valuation(df, valuations, "transfer_date", "backward", 730, "before")
    post_val = nearest_valuation(df, valuations, "tenure_end", "nearest", 400, "after")
    df["value_before"] = df["transfer_idx"].map(pre_val)
    df["value_after"] = df["transfer_idx"].map(post_val)
    df["value_before"] = df["value_before"].fillna(df["market_value_in_eur"])
    peak_val = compute_peak_valuation(df, valuations)
    df["value_peak"] = df["transfer_idx"].map(peak_val)
    df["value_peak"] = df["value_peak"].fillna(df["value_after"])
    df["value_peak"] = df[["value_peak", "value_after"]].max(axis=1)
    df["tenure_days"] = (df["tenure_end"] - df["transfer_date"]).dt.days
    df["still_on_loan"] = df["tenure_end"] >= REFERENCE_NOW

    print("Attaching player and club attributes for loans...")
    player_attrs = players.drop(columns=["current_club_id", "market_value_in_eur"])
    df = df.merge(player_attrs, on="player_id", how="left")
    df["age_at_transfer"] = (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.25

    from_clubs = clubs.add_prefix("from_")
    to_clubs = clubs.add_prefix("to_")
    df = df.merge(from_clubs, left_on="from_club_id", right_on="from_club_id", how="left")
    df = df.merge(to_clubs, left_on="to_club_id", right_on="to_club_id", how="left")

    valid = (
        df["age_at_transfer"].between(15, 42)
        & df["value_before"].gt(0)
        & df["value_after"].notna()
    )
    return df[valid].copy()


def finish_loan_dataset(df, league_position_baseline, position_fallback):
    """
    Second half of the loan-spell pipeline: df already has FotMob
    components attached (rating_pct/attacking_pct/defensive_pct/
    possession_pct/has_*_data, ranked jointly with permanent transfers -
    see attach_fotmob_components) - this computes the loan-specific
    league-adjusted performance features and loan_success_score, and
    writes data/loans_processed.csv. Two differences from main()'s
    permanent-transfer formula:

      - No value_for_money or resale_profit component. Both assume a
        permanent sale (a fee paid once and, maybe, a later resale); most
        loans carry no real fee at all, and a loan doesn't end in a sale of
        its own. The remaining four components (plus the four FotMob ones)
        are reweighted per data/loan_score_weights.json, with playing_time
        typically the largest share - whether the loan actually delivered
        game time is usually the central question a loan gets judged on.
      - perf_level/perf_delta/value_growth/playing_time are still ranked
        within the loans population only, not mixed with permanent
        transfers - a loan's value growth or appearance count over a much
        shorter window isn't on the same scale as a permanent tenure's, so
        comparing a loan against permanent-transfer norms would be
        misleading in both directions. This does NOT apply to the FotMob
        components, which were already ranked against the combined
        population before this function runs (see attach_fotmob_components
        for why per-90 rates don't have the same scale problem).

    Writes data/loans_processed.csv.
    """
    print("Computing loan-adjusted performance features...")
    df["from_league_ga_baseline"] = lookup_league_baseline(
        df["from_domestic_competition_id"], df["position"], league_position_baseline, position_fallback,
    )
    df["to_league_ga_baseline"] = lookup_league_baseline(
        df["to_domestic_competition_id"], df["position"], league_position_baseline, position_fallback,
    )
    df["pre_ga_p90_vs_league"] = df["pre_ga_p90"] / df["from_league_ga_baseline"].clip(lower=0.05)
    df["post_ga_p90_vs_league"] = df["post_ga_p90"] / df["to_league_ga_baseline"].clip(lower=0.05)

    df["perf_level_pct"] = df.groupby("position")["post_ga_p90_vs_league"].rank(pct=True) * 100
    df["expected_post_ga_p90_vs_league"] = compute_expected_post_performance(df)
    perf_delta_residual = df["post_ga_p90_vs_league"] - df["expected_post_ga_p90_vs_league"]
    df["perf_delta_pct"] = perf_delta_residual.groupby(df["position"]).rank(pct=True) * 100

    growth_to_peak = df["value_peak"] / df["value_before"].clip(lower=1)
    growth_to_end = df["value_after"] / df["value_before"].clip(lower=1)
    df["value_growth_pct"] = 0.6 * percentile_rank(growth_to_peak) + 0.4 * percentile_rank(growth_to_end)

    df["pct_team_games_played"] = (df["post_apps"] / df["team_games_in_tenure"].clip(lower=1)).clip(upper=1.0)
    df["playing_time_pct"] = 0.6 * percentile_rank(df["pct_team_games_played"]) + 0.4 * percentile_rank(df["post_apps"])

    if df.empty:
        # .map(...).apply(pd.Series) can't infer the perf_level/perf_delta/
        # value_growth/playing_time columns from zero rows (there's nothing
        # to infer them from), so it would KeyError below rather than just
        # producing an empty result. This path is real, not hypothetical:
        # it's what happens on a fresh clone that skips the optional
        # fetch_transfer_types.py step, where load_loan_spells() has no
        # loan rows to find at all.
        df["loan_success_score"] = pd.Series(dtype=float)
    else:
        # Same two-part idea as the permanent-transfer formula in main(),
        # minus the resale-profit layering (loans never have that
        # component at all - see data/loan_score_weights.json): the 4 base
        # components plus the 4 FotMob ones, each FotMob component's weight
        # dropped and the rest renormalized independently for a loan
        # missing that specific bucket, rather than guessed at.
        w = df["position"].map(LOAN_POSITION_WEIGHTS).apply(pd.Series)
        base_score = (
            w["perf_level"] * df["perf_level_pct"]
            + w["perf_delta"] * df["perf_delta_pct"]
            + w["value_growth"] * df["value_growth_pct"]
            + w["playing_time"] * df["playing_time_pct"]
        )
        fotmob_numerator = sum(
            np.where(df[f"has_{c}_data"], w[c] * df[f"{c}_pct"].fillna(0), 0)
            for c in FOTMOB_COMPONENTS
        )
        fotmob_weight_known = sum(
            np.where(df[f"has_{c}_data"], w[c], 0)
            for c in FOTMOB_COMPONENTS
        )
        missing_fotmob_weight = sum(w[c] for c in FOTMOB_COMPONENTS) - fotmob_weight_known
        effective_weight_sum = 1 - missing_fotmob_weight  # all 8 loan weights sum to 1.0 by construction (data/loan_score_weights.json)
        df["loan_success_score"] = ((base_score + fotmob_numerator) / effective_weight_sum).round(1)

    cols = [
        "player_id", "name", "transfer_date", "from_club_name", "to_club_name",
        "position", "sub_position", "foot", "height_in_cm", "age_at_transfer",
        "transfer_fee", "market_value_in_eur", "value_before", "value_after", "value_peak",
        "from_domestic_competition_id", "to_domestic_competition_id",
        "pre_apps", "pre_minutes", "pre_goals", "pre_assists",
        "pre_ga_p90", "pre_goals_p90", "pre_mins_per_app",
        "post_apps", "post_minutes", "post_goals", "post_assists", "post_ga_p90",
        "team_games_in_tenure", "pct_team_games_played",
        "from_league_ga_baseline", "to_league_ga_baseline",
        "pre_ga_p90_vs_league", "post_ga_p90_vs_league", "expected_post_ga_p90_vs_league",
        "tenure_days", "still_on_loan",
        "has_fotmob_data", "has_rating_data", "has_attacking_data", "has_defensive_data", "has_possession_data",
    ] + FOTMOB_RAW_COLS + [
        "perf_level_pct", "perf_delta_pct", "value_growth_pct", "playing_time_pct",
        "rating_pct", "attacking_pct", "defensive_pct", "possession_pct",
        "loan_success_score",
    ]
    out = df[cols].sort_values("transfer_date")
    out.to_csv(LOANS_OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows to {LOANS_OUT_PATH}")


def main():
    """
    End-to-end pipeline: load the raw Transfermarkt CSVs, compute every
    pre-transfer feature and post-transfer outcome described in the module
    docstring, blend the outcomes into a 0-100 success_score per position
    (see data/score_weights.json), and write the result to
    data/transfers_processed.csv - the label + features scripts/train_model.py
    trains on. Also writes data/league_baselines.csv as a side effect (see
    save_league_baselines) so app/main.py can share the same league
    baselines for live-prediction explanations, and data/loans_processed.csv
    (see prepare_loans/finish_loan_dataset) - the same idea applied to loan
    spells instead of permanent transfers, with its own loan_success_score.
    Permanent transfers and loans are prepared mostly independently, but
    meet in the middle at attach_fotmob_components(), which ranks both
    populations' rating/attacking/defensive/possession components jointly
    (see that function for why, unlike every other component here).
    """
    print("Loading raw CSVs...")
    transfers = load_transfers()
    players = load_players()
    clubs = load_clubs()
    appearances = load_appearances()
    valuations = load_valuations()
    team_games = load_team_games()
    print(f"{len(transfers):,} candidate transfers in {MIN_DATE.date()}..{MAX_DATE.date()}")

    print("Computing league/position goal-contribution baselines...")
    league_position_baseline, position_fallback = compute_league_position_baselines(appearances, players)
    save_league_baselines(league_position_baseline, position_fallback)

    print("Computing team games played during each tenure...")
    team_games_in_tenure = compute_team_games_in_window(transfers, team_games)

    print("Computing pre/post appearance windows...")
    pre, post = compute_windowed_appearance_stats(transfers, appearances)

    df = transfers.set_index("transfer_idx").join(pre).join(post).join(team_games_in_tenure)
    df[["pre_apps", "post_apps", "pre_minutes", "post_minutes", "pre_goals", "post_goals",
        "pre_assists", "post_assists", "team_games_in_tenure"]] = df[[
        "pre_apps", "post_apps", "pre_minutes", "post_minutes", "pre_goals", "post_goals",
        "pre_assists", "post_assists", "team_games_in_tenure",
    ]].fillna(0)

    before = len(df)
    df = df[(df["pre_apps"] >= MIN_APPS_PER_WINDOW) & (df["post_apps"] >= MIN_APPS_PER_WINDOW)]
    print(f"Kept {len(df):,} / {before:,} transfers with >= {MIN_APPS_PER_WINDOW} apps in both windows")

    df["pre_ga_p90"] = (df["pre_goals"] + df["pre_assists"]) / df["pre_minutes"].clip(lower=1) * 90
    df["post_ga_p90"] = (df["post_goals"] + df["post_assists"]) / df["post_minutes"].clip(lower=1) * 90
    df["pre_goals_p90"] = df["pre_goals"] / df["pre_minutes"].clip(lower=1) * 90
    df["pre_mins_per_app"] = df["pre_minutes"] / df["pre_apps"].clip(lower=1)

    print("Attaching market valuations...")
    df = df.reset_index()
    pre_val = nearest_valuation(df, valuations, "transfer_date", "backward", 730, "before")
    # Value "after" is taken near the end of the tenure (or now, if still
    # there) rather than a fixed point, to match the full-tenure window.
    post_val = nearest_valuation(df, valuations, "tenure_end", "nearest", 400, "after")
    df["value_before"] = df["transfer_idx"].map(pre_val)
    df["value_after"] = df["transfer_idx"].map(post_val)
    df["value_before"] = df["value_before"].fillna(df["market_value_in_eur"])
    peak_val = compute_peak_valuation(df, valuations)
    df["value_peak"] = df["transfer_idx"].map(peak_val)
    # value_after's "nearest to tenure_end" lookup (with a 400-day
    # tolerance) can occasionally land just outside the exact tenure
    # window compute_peak_valuation searches, so take the max of the two
    # rather than let value_peak be lower than value_after - value_after
    # is itself a real valuation point that should count.
    df["value_peak"] = df["value_peak"].fillna(df["value_after"])
    df["value_peak"] = df[["value_peak", "value_after"]].max(axis=1)
    df["tenure_days"] = (df["tenure_end"] - df["transfer_date"]).dt.days
    df["still_at_club"] = df["tenure_end"] >= REFERENCE_NOW

    print("Attaching player and club attributes...")
    club_value_proxy = compute_club_value_proxy(players)
    player_attrs = players.drop(columns=["current_club_id", "market_value_in_eur"])
    df = df.merge(player_attrs, on="player_id", how="left")
    df["age_at_transfer"] = (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.25

    from_clubs = clubs.add_prefix("from_")
    to_clubs = clubs.add_prefix("to_")
    df = df.merge(from_clubs, left_on="from_club_id", right_on="from_club_id", how="left")
    df = df.merge(to_clubs, left_on="to_club_id", right_on="to_club_id", how="left")
    df["from_total_market_value"] = df["from_club_id"].map(club_value_proxy)
    df["to_total_market_value"] = df["to_club_id"].map(club_value_proxy)

    df["fee_to_value_ratio"] = df["transfer_fee"] / df["market_value_in_eur"].clip(lower=1)
    df["club_quality_ratio"] = df["to_total_market_value"] / df["from_total_market_value"].clip(lower=1)

    valid = (
        df["age_at_transfer"].between(15, 42)
        & df["value_before"].gt(0)
        & df["value_after"].notna()
    )
    df = df[valid].copy()

    print("Computing composite success scores...")
    # Goal contributions are judged against the league they were actually
    # produced in, not just the whole dataset - 0.9 G+A/90 means something
    # different in a league that averages 0.55 for that position than one
    # that averages 0.46. Dividing by each league's empirical baseline
    # (from ALL appearances in that league, not just our filtered transfer
    # set) turns raw output into "how many times the going rate for this
    # league and position", so a raw decline after moving into a tougher
    # league isn't penalized the same as one moving into an easier league.
    df["from_league_ga_baseline"] = lookup_league_baseline(
        df["from_domestic_competition_id"], df["position"], league_position_baseline, position_fallback,
    )
    df["to_league_ga_baseline"] = lookup_league_baseline(
        df["to_domestic_competition_id"], df["position"], league_position_baseline, position_fallback,
    )
    df["pre_ga_p90_vs_league"] = df["pre_ga_p90"] / df["from_league_ga_baseline"].clip(lower=0.05)
    df["post_ga_p90_vs_league"] = df["post_ga_p90"] / df["to_league_ga_baseline"].clip(lower=0.05)

    # Performance level/delta are ranked *within position group* - goal
    # contributions per 90 minutes isn't comparable between a striker and a
    # centre-back, and ranking against the whole dataset drowns out real
    # differences among attackers (everyone not a defender/keeper clusters
    # near the top).
    df["perf_level_pct"] = df.groupby("position")["post_ga_p90_vs_league"].rank(pct=True) * 100

    df["expected_post_ga_p90_vs_league"] = compute_expected_post_performance(df)
    perf_delta_residual = df["post_ga_p90_vs_league"] - df["expected_post_ga_p90_vs_league"]
    df["perf_delta_pct"] = perf_delta_residual.groupby(df["position"]).rank(pct=True) * 100

    # Market value growth blends two signals: growth to the *peak* value
    # reached during the tenure (60%) and growth to the value near the end
    # of it (40%). Peak alone would ignore a real late-tenure collapse
    # (injury, loss of form); end-value alone unfairly reads a long,
    # valuable career as a decline, since even the best players' market
    # value falls with age by the time they eventually leave - Heung-min
    # Son joined Tottenham valued at ~25m, peaked at 90m mid-tenure, and
    # was worth ~20m a decade later when he left. The peak is what
    # actually reflects the asset the club held, even though the end
    # value is what they'd have realized in a sale at that moment.
    growth_to_peak = df["value_peak"] / df["value_before"].clip(lower=1)
    growth_to_end = df["value_after"] / df["value_before"].clip(lower=1)
    df["value_growth_pct"] = (
        0.6 * percentile_rank(growth_to_peak)
        + 0.4 * percentile_rank(growth_to_end)
    )

    # Playing time blends two different signals: raw appearance count
    # rewards a long, sustained presence at the club, but says nothing
    # about *availability* - a player who stayed 4 years and made 120
    # appearances out of 500 team games (heavily injury-hit) looks similar
    # to one who made 120 out of 140 (a nailed-on starter for a shorter
    # spell) on raw count alone. Percent of the team's actual games played
    # surfaces that difference (injuries, rotation, loss of form) directly,
    # so the two are blended rather than using either alone.
    df["pct_team_games_played"] = (df["post_apps"] / df["team_games_in_tenure"].clip(lower=1)).clip(upper=1.0)
    df["playing_time_pct"] = (
        0.6 * percentile_rank(df["pct_team_games_played"])
        + 0.4 * percentile_rank(df["post_apps"])
    )

    # Value-for-money: did performance level justify what was paid relative
    # to the player's own market value at the time, rather than relative to
    # every other transfer's fee (which makes any nine-figure fee look
    # "expensive" even when it's a bargain for that specific player).
    fee_to_value_pct = percentile_rank(df["transfer_fee"].fillna(0) / df["value_before"].clip(lower=1))
    df["value_for_money_pct"] = percentile_rank(df["perf_level_pct"] - fee_to_value_pct)

    # Four FotMob-derived components (rating, attacking, defensive,
    # possession) instead of one blended composite - see
    # compute_fotmob_component_pcts for the full reasoning. Loans are
    # prepared here (rather than at the very end, where they used to be)
    # so attach_fotmob_components can rank permanent transfers and loans
    # jointly - see that function for why that's correct for these four
    # components specifically, unlike every other component in this
    # formula. Real coverage gaps mean not every matched transfer ends up
    # with a usable value in every bucket (e.g. matched, but missing that
    # specific bucket's sub-stats), so each row's has_*_data flags are
    # defined off that bucket's own _pct column directly, not off the
    # merge/match itself - so a flag is never true for a row whose weight
    # would have nothing real to multiply. Like resale_profit below, a
    # bucket's weight is dropped and the rest renormalized when its flag
    # is false, rather than guessing at a neutral value for data we don't
    # have.
    print("Preparing loan spells...")
    loans_df = prepare_loans(players, clubs, appearances, valuations, team_games)
    print("Attaching FotMob components (ranked across transfers and loans together)...")
    df, loans_df = attach_fotmob_components(df, loans_df)

    # Resale profit: did the buying club later resell the player for more
    # than they paid? A real, distinct signal from sporting performance - a
    # decent-but-unspectacular player who's later flipped for a profit is a
    # good outcome for the club even if he was never a star there. Only
    # counted when there's a genuine subsequent sale for a recorded fee
    # (~31% of transfers); next_transfer_fee can no longer land on a loan-out
    # (loan rows are dropped in load_transfers before tenure_end/
    # next_transfer_fee are computed), but it can still be a real free
    # transfer or unknown/unrecorded fee, both of which stay "no resale data"
    # rather than being guessed at either way.
    has_resale_data = df["next_transfer_fee"].notna() & (df["next_transfer_fee"] > 0)
    transfer_fee_filled = df["transfer_fee"].fillna(0)
    resale_profit_ratio = (
        (df["next_transfer_fee"] - transfer_fee_filled) / transfer_fee_filled.clip(lower=1_000_000)
    )
    df["resale_profit_pct"] = pd.Series(np.nan, index=df.index)
    df.loc[has_resale_data, "resale_profit_pct"] = percentile_rank(resale_profit_ratio[has_resale_data])
    df["has_resale_data"] = has_resale_data

    # Weights vary by position - see data/score_weights.json for why. When
    # resale_profit is unknown for a transfer, its weight is dropped and the
    # rest are renormalized to still sum to 1, rather than filling in a
    # fabricated "neutral" score for data we don't actually have. Same
    # pattern now applies to each of the 4 FotMob components independently
    # (has_rating_data, has_attacking_data, ...): folded in first, below,
    # since they need to be settled BEFORE the resale-profit renormalization
    # runs on top of the result.
    w = df["position"].map(POSITION_WEIGHTS).apply(pd.Series)
    other_weight_sum = 1 - w["resale_profit"]  # e.g. 0.92 - the reference weight left for everything except resale_profit

    five_component_score = (
        w["perf_level"] * df["perf_level_pct"]
        + w["perf_delta"] * df["perf_delta_pct"]
        + w["value_growth"] * df["value_growth_pct"]
        + w["playing_time"] * df["playing_time_pct"]
        + w["value_for_money"] * df["value_for_money_pct"]
    )
    # Each FotMob component only contributes its weighted percentile to the
    # numerator - and its weight to the denominator - on rows where it's
    # actually known; fillna(0) on the percentile side is safe precisely
    # because the has_*_data mask already excludes it from the weight sum
    # too, so it can never silently count as "0, i.e. worst possible" for a
    # row that's simply missing data.
    fotmob_numerator = sum(
        np.where(df[f"has_{c}_data"], w[c] * df[f"{c}_pct"].fillna(0), 0)
        for c in FOTMOB_COMPONENTS
    )
    fotmob_weight_sum = sum(
        np.where(df[f"has_{c}_data"], w[c], 0)
        for c in FOTMOB_COMPONENTS
    )
    six_component_score = five_component_score + fotmob_numerator
    # Rescale up to other_weight_sum: when every FotMob component is known
    # this is a no-op (six_component_score already sums to
    # other_weight_sum); when some/all are missing, this redistributes
    # their share proportionally across whatever else the row does have -
    # same renormalization idea as "without_resale" below, just one layer
    # earlier.
    missing_fotmob_weight = sum(w[c] for c in FOTMOB_COMPONENTS) - fotmob_weight_sum
    six_component_weight_sum = other_weight_sum - missing_fotmob_weight
    base_score = six_component_score / six_component_weight_sum * other_weight_sum

    # When resale data IS known, how much it counts scales with tenure
    # length (compute_resale_weight) rather than the flat reference weight
    # above - a quick flip weights the resale outcome heavily, a long
    # career barely at all, since the club already extracted years of
    # value regardless of the eventual sale price.
    df["resale_weight"] = compute_resale_weight(df["tenure_days"] / 365.25)
    rescale = (1 - df["resale_weight"]) / other_weight_sum
    with_resale = base_score * rescale + df["resale_weight"] * df["resale_profit_pct"]
    without_resale = base_score / other_weight_sum
    df["success_score"] = np.where(has_resale_data, with_resale, without_resale).round(1)

    cols = [
        "player_id", "name", "transfer_date", "from_club_name", "to_club_name",
        "position", "sub_position", "foot", "height_in_cm", "age_at_transfer",
        "transfer_fee", "market_value_in_eur", "value_before", "value_after", "value_peak",
        "fee_to_value_ratio", "club_quality_ratio",
        "from_total_market_value", "to_total_market_value",
        "from_domestic_competition_id", "to_domestic_competition_id",
        "pre_apps", "pre_minutes", "pre_goals", "pre_assists",
        "pre_ga_p90", "pre_goals_p90", "pre_mins_per_app",
        "post_apps", "post_minutes", "post_goals", "post_assists", "post_ga_p90",
        "team_games_in_tenure", "pct_team_games_played",
        "from_league_ga_baseline", "to_league_ga_baseline",
        "pre_ga_p90_vs_league", "post_ga_p90_vs_league", "expected_post_ga_p90_vs_league",
        "tenure_days", "still_at_club",
        "next_transfer_fee", "has_resale_data", "resale_weight",
        "has_fotmob_data", "has_rating_data", "has_attacking_data", "has_defensive_data", "has_possession_data",
    ] + FOTMOB_RAW_COLS + [
        "perf_level_pct", "perf_delta_pct", "value_growth_pct", "playing_time_pct", "value_for_money_pct",
        "rating_pct", "attacking_pct", "defensive_pct", "possession_pct", "resale_profit_pct",
        "success_score",
    ]
    out = df[cols].sort_values("transfer_date")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows to {OUT_PATH}")

    print("\nFinishing loan-spell dataset...")
    finish_loan_dataset(loans_df, league_position_baseline, position_fallback)


if __name__ == "__main__":
    main()
