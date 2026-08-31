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
SUB_POSITION_WEIGHTS = _score_weights_raw["_sub_positions"]
POSITION_WEIGHTS = {
    k: v for k, v in _score_weights_raw.items()
    if not k.startswith("_") and k != "resale_weight_curve"
}

LOAN_SCORE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_score_weights.json")
with open(LOAN_SCORE_WEIGHTS_PATH) as f:
    _loan_score_weights_raw = json.load(f)
LOAN_SUB_POSITION_WEIGHTS = _loan_score_weights_raw["_sub_positions"]
LOAN_POSITION_WEIGHTS = {k: v for k, v in _loan_score_weights_raw.items() if not k.startswith("_")}


def lookup_weights(df, position_weights, sub_position_weights):
    """
    Per-row weight lookup: a row's actual sub_position (see
    load_actual_sub_positions) selects its weights from
    sub_position_weights when a distinct profile exists for that specific
    sub_position; otherwise (an unlisted sub_position like Central
    Midfield/Centre-Forward, which fall back to the broad row on purpose -
    see data/score_weights.json's _sub_positions_comment - or a missing
    sub_position) it falls back to position_weights[row's broad position].
    Percentile ranking (perf_level_pct, compute_fotmob_component_pcts,
    etc.) is unaffected by any of this - it still groups by the broad
    position column only, so e.g. a Centre-Back is still ranked against
    every Defender, just weighted differently once ranked.
    """
    rows = [
        sub_position_weights.get(sub_position, position_weights[position])
        for position, sub_position in zip(df["position"], df["sub_position"])
    ]
    return pd.DataFrame(rows, index=df.index)

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
PRETRANSFER_FOTMOB_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pretransfer_fotmob_stats_cache.csv")

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


# Raw pre-transfer FotMob stats (see scripts/fetch_pretransfer_fotmob_stats.py)
# - the *predict model's* pre-transfer performance signal, unlike
# FOTMOB_RAW_COLS above which feeds the historical score's post-transfer
# components. Not percentile-ranked or league-baseline-adjusted the way
# the score's FotMob components are - the predict model is a tree
# ensemble that can learn its own splits/thresholds directly from raw
# per-90 numbers, so that machinery (built for combining components onto
# one 0-100 scale) isn't needed here. Which of these actually earn a
# place as a trained feature is decided empirically in train_model.py,
# same "test before keeping" discipline as every other feature change -
# see README.
PRETRANSFER_FOTMOB_RAW_COLS = [
    "pre_fotmob_rating", "pre_fotmob_total_minutes", "pre_fotmob_seasons_used", "pre_fotmob_total_matches",
    "pre_fotmob_goals_per_90", "pre_fotmob_expected_goals_per_90", "pre_fotmob_expected_assists_per_90",
    "pre_fotmob_won_contest", "pre_fotmob_total_att_assist",
    "pre_fotmob_total_tackle", "pre_fotmob_interception", "pre_fotmob_effective_clearance", "pre_fotmob_ball_recovery",
    "pre_fotmob_accurate_pass", "pre_fotmob_saves", "pre_fotmob__save_percentage", "pre_fotmob_goals_conceded",
]


def load_pretransfer_fotmob_stats():
    """
    Load the per-transfer pre-transfer-year FotMob stats built by
    scripts/fetch_pretransfer_fotmob_stats.py - the mirror image of
    load_fotmob_stats() above, for the year *before* the move instead of
    the tenure after it. Same degrade-gracefully behavior when the cache
    doesn't exist yet.
    """
    if not os.path.exists(PRETRANSFER_FOTMOB_STATS_PATH):
        print(f"  (no pre-transfer FotMob stats cache at {PRETRANSFER_FOTMOB_STATS_PATH} - run scripts/fetch_pretransfer_fotmob_stats.py to enable it; continuing without it)")
        return pd.DataFrame(columns=["player_id", "transfer_date"] + PRETRANSFER_FOTMOB_RAW_COLS)
    stats = pd.read_csv(
        PRETRANSFER_FOTMOB_STATS_PATH,
        usecols=["player_id", "transfer_date"] + PRETRANSFER_FOTMOB_RAW_COLS,
        parse_dates=["transfer_date"],
    )
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


def load_actual_sub_positions():
    """
    players.csv's own sub_position is a single, undated label - whatever
    Transfermarkt currently lists for that player, the same value
    regardless of which transfer or era is being scored. For a player who
    changed roles over their career (e.g. central midfield early on,
    pushed into a more defensive role later) that mislabels every older
    transfer with today's role instead of the one they actually had at
    the time.

    game_lineups.csv doesn't have that problem: one row per (game,
    player), across both starting_lineup and substitutes rows, with the
    sub-position they were actually fielded in for that specific match.
    This takes each player's single most-common fielded sub-position
    across their *entire* lineup history - a real, dated career summary
    rather than a today-only snapshot. Verified against players.csv's own
    label: for players with a real sample (>=10 lineup rows), the two
    agree 74.7% of the time - real drift, not a rounding error - and
    roughly a quarter of players never settle into one dominant role at
    all (<70% of their own lineup rows at their single most-common
    position).

    This is still career-wide, not per-tenure - see
    data/score_weights.json's _sub_positions_comment for why that
    (coarser but simpler, and immune to short-tenure/loan sample-size
    issues) tradeoff was chosen. Returns a Series indexed by player_id;
    load_players() falls back to players.csv's own sub_position for any
    player with no game_lineups rows at all.
    """
    lineups = pd.read_csv(
        os.path.join(RAW_DIR, "game_lineups.csv"),
        usecols=["player_id", "position"],
    ).dropna(subset=["position"])
    return lineups.groupby("player_id")["position"].agg(lambda s: s.value_counts().idxmax())


def load_players():
    """
    Load players.csv: one row per player with position, physical
    attributes, and current club/value. sub_position is overridden with
    each player's actual career-wide fielded sub-position from
    game_lineups.csv (see load_actual_sub_positions) wherever that's
    available - only falling back to players.csv's own (single, undated)
    sub_position label for a player with no lineup data at all.
    """
    df = pd.read_csv(
        os.path.join(RAW_DIR, "players.csv"),
        usecols=[
            "player_id", "name", "date_of_birth", "position",
            "sub_position", "foot", "height_in_cm",
            "current_club_id", "market_value_in_eur",
        ],
        parse_dates=["date_of_birth"],
    )
    actual_sub_position = load_actual_sub_positions()
    df["sub_position"] = df["player_id"].map(actual_sub_position).fillna(df["sub_position"])
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


def compute_value_growth_pct(value_before, value_peak, value_after):
    """
    Blends two different views of market value growth, each independently
    percentile-ranked against the whole population, rather than either
    alone: growth *relative to the player's own pre-transfer value* (a
    cheap breakout signing wins big here - 5m to 20m is 4x) and the
    *absolute euro gain* (a marathon-sized fee can still win here on a
    comparatively modest ratio - 75m to 110m is "only" 1.47x but a real
    35m paper gain). Ratio alone systematically buries already-expensive
    transfers: the bigger the starting value, the harder it is to move the
    ratio at all, even when the club's asset has appreciated by tens of
    millions and could plainly be resold at a profit. Absolute euros alone
    has the opposite bias against cheap signings, so the two are blended.
    Both value_peak and value_after are blended in (80/20, leaning heavily
    toward peak) rather than using either alone - see value_peak's own
    docstring for why peak matters and end-of-tenure alone would misread
    a long, valuable career as a decline. Peak gets the large majority of
    the weight rather than an even split: end-of-tenure value already has
    its own dedicated signal elsewhere in the score (resale_profit - what
    the club actually realized when they sold the player, when that's
    known), so leaning value_growth itself more heavily on peak avoids
    doubly punishing a player for a value decline off their peak that
    resale_profit already accounts for on its own terms.
    """
    ratio_peak_pct = percentile_rank(value_peak / value_before.clip(lower=1))
    ratio_end_pct = percentile_rank(value_after / value_before.clip(lower=1))
    ratio_combo = 0.8 * ratio_peak_pct + 0.2 * ratio_end_pct

    abs_gain_peak_pct = percentile_rank(value_peak - value_before)
    abs_gain_end_pct = percentile_rank(value_after - value_before)
    abs_combo = 0.8 * abs_gain_peak_pct + 0.2 * abs_gain_end_pct

    return 0.5 * ratio_combo + 0.5 * abs_combo


FEE_OVERPAY_FREE_PASS_RATIO = 1.3


def compute_fee_penalty_pct(transfer_fee, value_before):
    """
    How much a transfer's fee should count against value_for_money, as a
    0-100 percentile - but NOT a plain percentile_rank(fee / value_before)
    the way an earlier version worked, because that turns out too harsh
    for an ordinary premium: paying 1.2-1.5x a player's pre-transfer value
    is common (the dataset's own median fee/value ratio is 0.62x, 75th
    percentile only 1.17x), not some rare extreme, and clubs routinely pay
    a modest premium for plenty of transfers that still work out fine.

    Below FEE_OVERPAY_FREE_PASS_RATIO (1.3x), a fee counts as a completely
    normal premium and gets zero penalty, full stop - not "a small
    penalty," genuinely zero. Only the ~19% of transfers that actually
    exceed that ratio get ranked at all, and only against EACH OTHER
    (not the whole dataset) - a transfer with a small excess over 1.3x
    should be judged against how *other overpays* compare, not muddied by
    the 81% of transfers that never overpaid at all.

    This only works because it's NOT a monotonic transform fed into a
    single global percentile_rank(): percentile rank only depends on
    relative order, so scaling or clipping the ratio *before* one global
    rank (an earlier, wrong attempt at this) leaves every above-threshold
    transfer's rank essentially unchanged - the "free pass" population
    still occupies the same share of "everyone ranked below me" either
    way. Splitting into two literally different populations - a fixed 0
    for anyone under the threshold, a real percentile rank *within only
    the overpaid group* for anyone over it - is what actually creates a
    ratio just over 1.3x (small excess, ranks low within the overpaid
    group) and a ratio of 5x (huge excess, ranks high within it) landing
    in genuinely different places, instead of both merely being "above
    the line."

    Returns a Series aligned to transfer_fee's index, 0-100, 0 = fee
    fully justified regardless of size, 100 = the single worst overpay in
    the dataset.
    """
    ratio = transfer_fee.fillna(0) / value_before.clip(lower=1)
    has_overpay = ratio > FEE_OVERPAY_FREE_PASS_RATIO
    excess = (ratio - FEE_OVERPAY_FREE_PASS_RATIO).clip(lower=0)

    fee_penalty_pct = pd.Series(0.0, index=transfer_fee.index)
    fee_penalty_pct.loc[has_overpay] = percentile_rank(excess.loc[has_overpay])
    return fee_penalty_pct


VALUE_FOR_MONEY_PERFORMANCE_COMPONENTS = ["perf_level", "attacking", "defensive", "possession", "rating"]


def value_for_money_performance_proxy(df, w):
    """
    The "performance" side of value_for_money's fee-vs-output comparison
    (see main()) - not just perf_level_pct (goal contributions/90), which
    is only a real quality signal for attack-minded roles. A weighted
    blend of every on-pitch quality signal already computed for this row
    (perf_level_pct, attacking_pct, defensive_pct, possession_pct,
    rating_pct - VALUE_FOR_MONEY_PERFORMANCE_COMPONENTS), weighted by the
    row's own position/sub-position weight profile `w` (see
    lookup_weights) - the same weights the rest of the score already uses
    to decide how much each signal matters for this specific role, so a
    Centre-Back's "performance" leans on defensive_pct the way a winger's
    leans on attacking_pct/perf_level_pct, continuously rather than an
    all-or-nothing switch between two components.

    This started as a goalkeeper-only fix (perf_level_pct is meaningless
    for keepers: 82% have exactly 0 goal contributions, tied at the same
    percentile regardless of how well they actually played), then a
    defensive-vs-perf_level switch for defense-oriented roles generally
    (checked directly: perf_level_pct barely correlates with
    defensive_pct for Centre-Back (0.01), and is actually *negative* for
    Right-Back (-0.18), Left-Back (-0.14), and Defensive Midfield
    (-0.15) - goal contributions aren't just a weak proxy for defensive
    quality there, they're roughly uncorrelated-to-inversely-related with
    it). A hard switch between only two components still has the same
    flavor of problem one level up, though: it throws away
    attacking_pct/possession_pct/rating_pct entirely for a full-back
    whose game genuinely involves all four, and flips its answer sharply
    right at whatever weight threshold decides "dominant." Blending all
    of them, in the same proportion the rest of the score already trusts,
    avoids both problems at once.

    A component that's unknown for this row (no FotMob data for that
    specific bucket) drops its weight from both the numerator and the
    denominator, same renormalization pattern used everywhere else in
    this file, rather than guessing at a neutral value. perf_level_pct
    is always defined (Transfermarkt goal data has no coverage gaps the
    way FotMob does) and every non-goalkeeper position has SOME weight on
    it, so the denominator is never zero for an outfielder; a goalkeeper
    (weight 0 on perf_level/attacking) with no FotMob data at all for
    that tenure is the one case that can zero out entirely, so it falls
    back to plain perf_level_pct there, keeping value_for_money_pct
    defined for every row the way the rest of main() expects (unlike the
    four FotMob components themselves, value_for_money doesn't have a
    has_*_data flag / weight-drop path of its own).
    """
    pct = {
        "perf_level": df["perf_level_pct"], "attacking": df["attacking_pct"],
        "defensive": df["defensive_pct"], "possession": df["possession_pct"], "rating": df["rating_pct"],
    }
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for c in VALUE_FOR_MONEY_PERFORMANCE_COMPONENTS:
        known = pct[c].notna()
        numerator = numerator + np.where(known, w[c] * pct[c].fillna(0), 0)
        denominator = denominator + np.where(known, w[c], 0)
    proxy = numerator / denominator.replace(0, np.nan)
    return proxy.fillna(df["perf_level_pct"])


def fold_perf_level_into_attacking(df):
    """
    perf_level_pct (Transfermarkt goal contributions/90, covering every
    season back to FIRST_SEASON_YEAR) and attacking_pct (FotMob goals/xG/
    xA/chance creation, only from each league's own FotMob start season
    onward - see fetch_fotmob_stats.py) both measure the same underlying
    thing: attacking output. Weighting them as two fully independent
    components double-counts that signal for every transfer where both
    happen to be known.

    Where FotMob attacking data exists, this overwrites attacking_pct in
    place with the average of the two, so the richer FotMob signal
    (goals/xG/xA/chance creation) still gets folded in alongside the raw
    Transfermarkt goals+assists number rather than being thrown away.
    Where it doesn't (older transfers, or leagues FotMob doesn't cover
    that far back), attacking_pct is left untouched (still NaN) and
    perf_level keeps counting standalone - it's the only signal available,
    and it would be wrong to drop it just because *some other* transfer
    happens to have FotMob data.

    Returns the has_attacking_data mask; callers use it to zero out
    perf_level's own weight for these same rows and fold that weight into
    attacking's instead (see main()/finish_loan_dataset()), so the total
    weight budget for "attacking output" is unchanged either way - just
    concentrated in one bucket instead of split across two overlapping
    ones.
    """
    has_attacking = df["has_attacking_data"]
    df["attacking_pct"] = np.where(
        has_attacking, 0.5 * (df["perf_level_pct"] + df["attacking_pct"]), df["attacking_pct"],
    )
    return has_attacking


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
MIN_FOTMOB_LEAGUE_BASELINE_ROWS = 15


def compute_fotmob_league_baselines(df, stats):
    """
    For each (destination league, position) and each raw FotMob per-90
    stat, the minutes-weighted average across every matched tenure in the
    combined transfers+loans FotMob sample - used to adjust for genuine
    differences in playing style across leagues before ranking, the same
    idea compute_league_position_baselines already applies to goal
    contributions. Confirmed empirically: Bundesliga/Ligue 1/Denmark
    centre-backs average measurably more tackles+interceptions+
    clearances+recoveries per 90 than Premier League/La Liga ones (a
    ~20-point spread in average defensive_pct) - a real style difference
    (more transition-heavy play generates more defensive actions per
    player), not a quality one, that was previously silently baked into
    every FotMob-derived component.

    Unlike compute_league_position_baselines (built from the *entire*
    appearances.csv - every player, every league), this only has the
    FotMob sample itself to work with (~5,700 transfers/loans in
    data/raw/fotmob_stats_cache.csv) - well-covered leagues (Premier
    League: 100+ matched centre-backs alone) get a stable baseline, thin
    ones fall back to the position-wide average via
    MIN_FOTMOB_LEAGUE_BASELINE_ROWS, same fallback pattern as
    lookup_league_baseline.

    Returns {stat: (baseline_series, fallback_series)} - baseline_series
    indexed by (competition_id, position) with thin cells dropped,
    fallback_series indexed by position alone - both feed
    lookup_league_baseline per row, per stat.
    """
    baselines = {}
    for stat in stats:
        valid = df[stat].notna() & df["fotmob_total_minutes"].notna()
        sub = df.loc[valid, [stat, "fotmob_total_minutes", "to_domestic_competition_id", "position"]]
        weighted = sub[stat] * sub["fotmob_total_minutes"]

        grouped = pd.DataFrame({"weighted": weighted, "minutes": sub["fotmob_total_minutes"], "n": 1}).groupby(
            [sub["to_domestic_competition_id"], sub["position"]]
        ).sum()
        grouped = grouped[grouped["n"] >= MIN_FOTMOB_LEAGUE_BASELINE_ROWS]
        baseline = grouped["weighted"] / grouped["minutes"].clip(lower=1)

        pos_grouped = pd.DataFrame({"weighted": weighted, "minutes": sub["fotmob_total_minutes"]}).groupby(sub["position"]).sum()
        fallback = pos_grouped["weighted"] / pos_grouped["minutes"].clip(lower=1)

        baselines[stat] = (baseline, fallback)
    return baselines


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

    Each raw stat is first adjusted against its (league, position)
    baseline (see compute_fotmob_league_baselines) - a DIFFERENCE (raw
    stat minus baseline), not a ratio like perf_level's league adjustment
    uses, since several of these stats are negative by construction
    (fotmob_goals_conceded_inv) or already a percentage
    (fotmob__save_percentage), where a ratio's sign/scale gets confusing;
    a plain "N more/fewer than the league average" offset works
    uniformly across all of them. The adjusted value is then
    percentile-ranked *within position group*, like perf_level_pct, then
    (for the three multi-stat buckets) averaged across whichever of that
    bucket's raw stats are actually available for the row, so one missing
    sub-stat doesn't zero out the whole bucket. "Defensive" uses a
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
    league_baselines = compute_fotmob_league_baselines(df, all_stats)

    pct = {}
    for stat in all_stats:
        baseline, fallback = league_baselines[stat]
        stat_league_baseline = lookup_league_baseline(df["to_domestic_competition_id"], df["position"], baseline, fallback)
        stat_vs_league = df[stat] - stat_league_baseline
        pct[stat] = stat_vs_league.groupby(df["position"]).rank(pct=True) * 100

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

    df["value_growth_pct"] = compute_value_growth_pct(df["value_before"], df["value_peak"], df["value_after"])

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
        # missing that specific bucket, rather than guessed at. perf_level
        # and attacking are folded together first (see
        # fold_perf_level_into_attacking) since they'd otherwise double-
        # count the same attacking-output signal.
        w = lookup_weights(df, LOAN_POSITION_WEIGHTS, LOAN_SUB_POSITION_WEIGHTS)
        has_attacking = fold_perf_level_into_attacking(df)
        perf_level_weight = np.where(has_attacking, 0, w["perf_level"])
        attacking_weight = np.where(has_attacking, w["attacking"] + w["perf_level"], w["attacking"])

        known_score = (
            perf_level_weight * df["perf_level_pct"]
            + w["perf_delta"] * df["perf_delta_pct"]
            + w["value_growth"] * df["value_growth_pct"]
            + w["playing_time"] * df["playing_time_pct"]
        )
        known_weight = perf_level_weight + w["perf_delta"] + w["value_growth"] + w["playing_time"]

        fotmob_component_weight = {"rating": w["rating"], "attacking": attacking_weight, "defensive": w["defensive"], "possession": w["possession"]}
        for c in FOTMOB_COMPONENTS:
            comp_weight = fotmob_component_weight[c]
            comp_known = df[f"has_{c}_data"]
            known_score = known_score + np.where(comp_known, comp_weight * df[f"{c}_pct"].fillna(0), 0)
            known_weight = known_weight + np.where(comp_known, comp_weight, 0)

        # all 8 loan weights sum to 1.0 by construction (data/loan_score_weights.json), so known_weight is already the effective denominator
        df["loan_success_score"] = (known_score / known_weight).round(1)

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

    # See compute_value_growth_pct for the ratio/absolute-gain blend.
    df["value_growth_pct"] = compute_value_growth_pct(df["value_before"], df["value_peak"], df["value_after"])

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

    # Predict-model-only pre-transfer FotMob signal (see
    # PRETRANSFER_FOTMOB_RAW_COLS) - raw per-90 numbers, not percentile-
    # ranked, and not part of the success_score label at all (unlike the
    # post-transfer FotMob components just attached above). chances-
    # created is a season total on FotMob's own leaderboard, not a rate -
    # normalized to per-90 the same way attach_fotmob_components does for
    # the post-transfer side.
    df = df.merge(load_pretransfer_fotmob_stats(), on=["player_id", "transfer_date"], how="left")
    pre_minutes_per_90 = (df["pre_fotmob_total_minutes"] / 90).clip(lower=1)
    df["pre_fotmob_chances_created_p90"] = df["pre_fotmob_total_att_assist"] / pre_minutes_per_90

    # Weights vary by position - see data/score_weights.json for why.
    # Looked up here (rather than down by the weighted-sum below, where
    # this used to happen) because value_for_money_performance_proxy
    # needs it too - which "performance" signal counts for value_for_money
    # depends on the row's own weight profile.
    w = lookup_weights(df, POSITION_WEIGHTS, SUB_POSITION_WEIGHTS)

    # Value-for-money: did performance level justify what was paid relative
    # to the player's own market value at the time, rather than relative to
    # every other transfer's fee (which makes any nine-figure fee look
    # "expensive" even when it's a bargain for that specific player). See
    # compute_fee_penalty_pct for why a fee's penalty isn't a plain
    # percentile_rank(fee / value_before) - an ordinary premium (up to
    # 1.3x pre-transfer value) gets zero penalty, only genuine overpays
    # count, ranked against each other rather than the whole dataset. See
    # value_for_money_performance_proxy for why the "performance" side
    # isn't always perf_level_pct - computed here, after
    # attach_fotmob_components, because it needs defensive_pct, which
    # doesn't exist until that call has run.
    fee_penalty_pct = compute_fee_penalty_pct(df["transfer_fee"], df["value_before"])
    performance_proxy = value_for_money_performance_proxy(df, w)
    df["value_for_money_pct"] = percentile_rank(pd.Series(performance_proxy, index=df.index) - fee_penalty_pct)

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

    # w (each row's position/sub-position weight profile) was already
    # looked up above, before value_for_money. When resale_profit is
    # unknown for a transfer, its weight is dropped and the rest are
    # renormalized to still sum to 1, rather than filling in a fabricated
    # "neutral" score for data we don't actually have. Same pattern now
    # applies to each of the 4 FotMob components independently
    # (has_rating_data, has_attacking_data, ...): folded in first, below,
    # since they need to be settled BEFORE the resale-profit renormalization
    # runs on top of the result.
    other_weight_sum = 1 - w["resale_profit"]  # e.g. 0.92 - the reference weight left for everything except resale_profit

    # perf_level and attacking both measure attacking output (see
    # fold_perf_level_into_attacking) - fold them together first so their
    # weights combine into one bucket instead of double-counting the same
    # signal. perf_level_weight is 0 wherever that folding happened
    # (its weight moved into attacking_weight instead), so it's safe to
    # keep multiplying it by perf_level_pct unconditionally below.
    has_attacking = fold_perf_level_into_attacking(df)
    perf_level_weight = np.where(has_attacking, 0, w["perf_level"])
    attacking_weight = np.where(has_attacking, w["attacking"] + w["perf_level"], w["attacking"])

    known_score = (
        perf_level_weight * df["perf_level_pct"]
        + w["perf_delta"] * df["perf_delta_pct"]
        + w["value_growth"] * df["value_growth_pct"]
        + w["playing_time"] * df["playing_time_pct"]
        + w["value_for_money"] * df["value_for_money_pct"]
    )
    known_weight = perf_level_weight + w["perf_delta"] + w["value_growth"] + w["playing_time"] + w["value_for_money"]

    # Each FotMob component only contributes its weighted percentile to the
    # numerator - and its weight to the denominator - on rows where it's
    # actually known; fillna(0) on the percentile side is safe precisely
    # because the has_*_data mask already excludes it from the weight sum
    # too, so it can never silently count as "0, i.e. worst possible" for a
    # row that's simply missing data. attacking uses attacking_weight (its
    # own weight, plus perf_level's when folded in) rather than the flat
    # w["attacking"] the other three components use.
    fotmob_component_weight = {"rating": w["rating"], "attacking": attacking_weight, "defensive": w["defensive"], "possession": w["possession"]}
    for c in FOTMOB_COMPONENTS:
        comp_weight = fotmob_component_weight[c]
        comp_known = df[f"has_{c}_data"]
        known_score = known_score + np.where(comp_known, comp_weight * df[f"{c}_pct"].fillna(0), 0)
        known_weight = known_weight + np.where(comp_known, comp_weight, 0)

    # Rescale up to other_weight_sum: when everything above is known this
    # is a no-op (known_weight already sums to other_weight_sum); when
    # something's missing, this redistributes its share proportionally
    # across whatever else the row does have - same renormalization idea
    # as "without_resale" below, just one layer earlier.
    base_score = known_score / known_weight * other_weight_sum

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
    ] + FOTMOB_RAW_COLS + PRETRANSFER_FOTMOB_RAW_COLS + ["pre_fotmob_chances_created_p90"] + [
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
