"""
Build a per-transfer training dataset from the raw Transfermarkt CSVs
(dcaribou/transfermarkt-datasets, mirrored on Kaggle as davidcariboo/player-scores).

For every transfer we compute:
  - pre-transfer features: age, position, physical attributes, fee, market
    value, and performance in the player's final year at the old club
  - a post-hoc "success score" (0-100) blending performance level/delta,
    market value growth, playing time, and value-for-money, all measured
    over the player's *entire tenure* at the new club (from the transfer
    until their next departure, or "now" if they're still there) rather
    than a fixed first-year window. A fixed window either penalizes slow
    starters who took time to adapt, or misses a player who started hot
    and faded once the honeymoon period ended.

Only the pre-transfer features are used as model inputs; the success score
is the training label.
"""
import json
import os
import numpy as np
import pandas as pd

SCORE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "score_weights.json")
with open(SCORE_WEIGHTS_PATH) as f:
    POSITION_WEIGHTS = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

RAW_DIR = os.environ.get(
    "TRANSFERMARKT_RAW_DIR",
    os.path.expanduser(
        "~/.cache/kagglehub/datasets/davidcariboo/player-scores/versions/677"
    ),
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers_processed.csv")

PRE_WINDOW_DAYS = 365
MIN_APPS_PER_WINDOW = 10
MIN_DATE = pd.Timestamp("2013-01-01")
MAX_DATE = pd.Timestamp.today().normalize()
REFERENCE_NOW = pd.Timestamp.today().normalize()


def load_transfers():
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
    df = df.reset_index(drop=True)
    df["transfer_idx"] = df.index

    # Tenure at the new club runs until the player's next transfer (any
    # destination), or until "now" if they haven't moved again since.
    df = df.sort_values(["player_id", "transfer_date"])
    df["tenure_end"] = df.groupby("player_id")["transfer_date"].shift(-1)
    df["next_transfer_fee"] = df.groupby("player_id")["transfer_fee"].shift(-1)
    df["tenure_end"] = df["tenure_end"].fillna(REFERENCE_NOW)
    df = df.sort_values("transfer_idx").reset_index(drop=True)
    return df


def load_players():
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
    # total_market_value in clubs.csv is unpopulated in this dataset version, so
    # club quality is proxied below from the summed current market value of each
    # club's squad (players.csv), rather than read from clubs.csv directly.
    df = pd.read_csv(
        os.path.join(RAW_DIR, "clubs.csv"),
        usecols=["club_id", "domestic_competition_id"],
    )
    return df


def compute_club_value_proxy(players):
    proxy = (
        players.dropna(subset=["current_club_id"])
        .groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .rename("club_value_proxy")
    )
    proxy.index = proxy.index.astype(int)
    return proxy


def load_appearances():
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
    keys = list(zip(league_ids, positions))
    return pd.Series(
        [baseline.get(k, fallback.get(k[1], fallback.mean())) for k in keys],
        index=league_ids.index,
    )


def load_valuations():
    df = pd.read_csv(
        os.path.join(RAW_DIR, "player_valuations.csv"),
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    )
    return df.sort_values("date")


def compute_windowed_appearance_stats(transfers, appearances):
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


def percentile_rank(series):
    return series.rank(pct=True) * 100


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


def main():
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

    value_growth = df["value_after"] / df["value_before"].clip(lower=1)
    df["value_growth_pct"] = percentile_rank(value_growth)

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

    # Resale profit: did the buying club later resell the player for more
    # than they paid? A real, distinct signal from sporting performance - a
    # decent-but-unspectacular player who's later flipped for a profit is a
    # good outcome for the club even if he was never a star there. Only
    # counted when there's a genuine subsequent sale for a recorded fee
    # (~17% of transfers); the dataset doesn't distinguish loans from
    # permanent transfers, and most "next transfer, fee=0" cases are loans
    # rather than real free exits, so those are treated as unknown rather
    # than guessed at either way.
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
    # fabricated "neutral" score for data we don't actually have.
    w = df["position"].map(POSITION_WEIGHTS).apply(pd.Series)
    base_score = (
        w["perf_level"] * df["perf_level_pct"]
        + w["perf_delta"] * df["perf_delta_pct"]
        + w["value_growth"] * df["value_growth_pct"]
        + w["playing_time"] * df["playing_time_pct"]
        + w["value_for_money"] * df["value_for_money_pct"]
    )
    with_resale = base_score + w["resale_profit"] * df["resale_profit_pct"]
    without_resale = base_score / (1 - w["resale_profit"])
    df["success_score"] = np.where(has_resale_data, with_resale, without_resale).round(1)

    cols = [
        "player_id", "name", "transfer_date", "from_club_name", "to_club_name",
        "position", "sub_position", "foot", "height_in_cm", "age_at_transfer",
        "transfer_fee", "market_value_in_eur", "value_before", "value_after",
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
        "next_transfer_fee", "has_resale_data",
        "perf_level_pct", "perf_delta_pct", "value_growth_pct",
        "playing_time_pct", "value_for_money_pct", "resale_profit_pct",
        "success_score",
    ]
    out = df[cols].sort_values("transfer_date")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
