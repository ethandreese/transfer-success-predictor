import csv
import io
import json
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = FastAPI(title="Transfer Success Predictor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def fold_accents(value):
    """Lowercase with diacritics stripped, e.g. 'Dembélé' -> 'dembele', so a plain-ASCII search matches accented names."""
    if not isinstance(value, str):
        return value
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


CLUB_NAME_STRIP_TOKENS = {
    "fc", "cf", "cfc", "acf", "ssc", "uc", "bc", "as", "gd", "cd", "ud", "kv", "krc", "rc", "fk",
    "sc", "ac", "afc", "club", "de", "football", "calcio",
    "vfl", "vfb", "tsv", "sv", "ssv", "us",
    "spor", "kulubu", "jimnastik",
}

# Nickname/official-name pairs that share no common token after
# CLUB_NAME_STRIP_TOKENS stripping, so build_club_name_aliases' mechanical
# pass can't catch them on its own ("Man City" and "Manchester City" have
# no word in common at all). Each group here was individually verified
# against the actual data before being added - not assumed from football
# knowledge alone - by checking that every raw-name variant in the group
# shares the same domestic_competition_id (both directions) and a
# plausible, non-contradictory transfer_date range; see the PR/commit this
# shipped in for the full per-group check. That check also caught a genuine
# false lead: "Sporting" alone looks ambiguous (Sporting CP, Sporting
# Gijón, and Royal Charleroi Sporting Club are all real clubs nicknamed
# "Sporting"), but every single "Sporting" row in this dataset carries
# league PO1 - the same league as "Sporting CP" and neither of the other
# two - so it's included below, unlike a name that actually did straddle
# more than one plausible club (none found in this pass, but the check is
# what makes the inclusion safe, not the resemblance).
#
# Deliberately not exhaustive - there are ~700 club names in this dataset
# and this covers the ones spot-checked so far (see README's Known
# limitations for what's still open, e.g. any second-division or
# less-followed club whose nickname/official-name split was never looked
# at).
#
# A second full sweep (triggered by "Swansea"/"Swansea City" still showing
# as two rows) went through every pair of names in the dataset that share a
# whole word after folding, and added every verified match below the first
# 28 groups. Also checked and rejected in that sweep, same as "Racing"
# above - looks related, isn't: "Barcelona"/"RCD Espanyol Barcelona" (two
# different Barcelona clubs, not a spelling variant), "Arsenal"/"Arsenal
# Tula"/"Arsenal Kyiv" (unrelated clubs that happen to share the English
# club's name), "Rangers"/"Queens Park Rangers" (Glasgow Rangers, not QPR),
# "Sporting"/"Sporting Gijón" and "Sporting"/"Royal Charleroi Sporting Club"
# (every bare "Sporting" row is league PO1 - see the note on the existing
# "Sporting"/"Sporting CP" entry below - these two aren't), "Nacional"/"Atl.
# Nacional" (Portuguese CD Nacional vs. Colombian Atlético Nacional), and
# "Athletic"/"Wigan Athletic"/"Forfar Athletic" (every bare "Athletic" row
# is league ES1 - it's Athletic Bilbao, not Wigan's or Forfar's). Also
# "Krasnodar"/"Kuban Krasnodar" despite sharing a league and a city: FC
# Krasnodar (founded 2008) and the dissolved Kuban Krasnodar are two
# separate real clubs, the same trap as Manchester City/United or Dundee
# FC/United. And reserve/youth sides ("Benfica"/"Benfica B", "Barcelona"/
# "Barcelona B", "Tottenham"/"Tottenham U21", "Villarreal"/"FC Villarreal
# C", "Krasnodar"/"Krasnodar 2", "Utrecht"/"Utrecht U21", "FC Cartagena"/
# "FC Cartagena B") were left split on purpose - a club's B/youth team runs
# its own transfer history, not the first team's.
#
# One cluster came back genuinely contested rather than a clean
# same-club-or-not call: Ukrainian "Metalist Kharkiv" went bankrupt around
# 2016, and two separately-run organizations have since both laid claim to
# the name/legacy - the user confirmed merging only "Metalist Kharkiv" with
# "Metalist Kharkiv (- 2016)" (same name, just marking when that spelling
# stopped appearing) while leaving bare "Metalist" and "Metalist 1925" split,
# since which of those two is the "real" continuation is a live dispute, not
# something this data can settle.
#
# A third pass, this time by string-similarity rather than shared whole
# words (to catch abbreviation/typo-style variants like "Man City" that the
# second pass's word-sharing check would miss), found one more contested
# case with the exact same shape as Metalist: Belgian "Beerschot AC" (only
# ever mentioned in 2013) went bankrupt that year, and the club playing
# under the Beerschot name from 2018 onward ("Beerschot VA") is a later,
# separately-reformed entity - per user confirmation, only "Beerschot VA"
# and its punctuation variant "Beerschot V.A." are merged; "Beerschot AC" is
# left split. Also rejected in this pass: "Metalurg D."/"Metalurg Z." (two
# different Ukrainian clubs - Donetsk and Zaporizhzhia - that happen to
# share a country with only one top flight, so the league-match check that
# works elsewhere doesn't discriminate here) and "Al-Wehda"/"Al-Wahda"
# (likely a Saudi club and a UAE club respectively, not enough data on the
# second spelling to confirm either way).
CLUB_NICKNAME_GROUPS = [
    {"Man City", "Manchester City"},
    {"Man Utd", "Manchester United"},
    {"PSG", "Paris SG", "Paris Saint-Germain"},
    {"Bor. Dortmund", "Borussia Dortmund", "Dortmund"},
    {"Tottenham", "Tottenham Hotspur"},
    {"Newcastle", "Newcastle United"},
    {"West Ham", "West Ham United"},
    {"West Brom", "West Bromwich Albion"},
    {"Brighton", "Brighton & Hove Albion"},
    {"Leeds", "Leeds United"},
    {"Leicester", "Leicester City"},
    {"AS Monaco", "Monaco"},
    {"Lyon", "Olympique Lyon"},
    {"Marseille", "Olympique Marseille"},
    {"LOSC Lille", "Lille"},
    {"Nice", "OGC Nice"},
    {"Real Betis", "Real Betis Balompié"},
    {"Athletic Bilbao", "Athletic Club", "Athletic"},
    {"Ajax", "Ajax Amsterdam"},
    {"Feyenoord", "Feyenoord Rotterdam"},
    {"PSV", "PSV Eindhoven"},
    {"Benfica", "SL Benfica"},
    {"Espanyol", "RCD Espanyol Barcelona"},
    {"Hamburg", "Hamburger SV"},
    {"Zenit S-Pb", "AO FK Zenit Sankt-Peterburg"},
    {"Shakhtar D.", "FC Shakhtar Donetsk"},
    {"Sporting", "Sporting CP"},
    {"Inter", "Inter Milan"},
    {"Swansea", "Swansea City"},
    {"Cardiff", "Cardiff City"},
    {"Norwich", "Norwich City"},
    {"Wigan", "Wigan Athletic"},
    {"Huddersfield", "Huddersfield Town"},
    {"Wolves", "Wolverhampton Wanderers"},
    {"QPR", "Queens Park Rangers"},
    {"Nottingham Forest", "Nott'm Forest", "Nottm Forest"},
    {"Frankfurt", "Eintracht Frankfurt", "E. Frankfurt"},
    {"Mönchengladbach", "Borussia Mönchengladbach", "Bor. M'gladbach"},
    {"Hoffenheim", "TSG Hoffenheim", "TSG 1899 Hoffenheim"},
    {"Atlético", "Atlético de Madrid", "Atlético Madrid"},
    {"Bologna", "Bologna Football Club 1909"},
    {"Mainz", "1.FSV Mainz 05"},
    {"Lazio", "Società Sportiva Lazio S.p.A."},
    {"Panathinaikos", "Panathinaikos Athlitikos Omilos"},
    {"Willem II", "Willem II Tilburg"},
    {"Karabükspor", "Kardemir Karabükspor"},
    {"Ankaragücü", "MKE Ankaragücü"},
    {"Leipzig", "RB Leipzig"},
    {"Salzburg", "RB Salzburg"},
    {"Montpellier", "Montpellier HSC"},
    {"Twente FC", "FC Twente", "FC Twente Enschede"},
    {"Estoril", "Estoril Praia", "GD Estoril Praia"},
    {"Vitesse", "Vitesse Arnhem"},
    {"Excelsior", "Excelsior Rotterdam"},
    {"Panionios", "Panionios Athens"},
    {"SönderjyskE", "Sönderjyske", "Sönderjyske Fodbold"},
    {"Roda JC", "Roda JC Kerkrade"},
    {"Dnipro", "Dnipro Dnipropetrovsk (-2020)"},
    {"Kryvbas", "Kryvbas Kryvyi Rig"},
    {"Belenenses", "CF Os Belenenses"},
    {"Chornomorets", "Chornomorets Odesa"},
    {"Panetolikos", "Panetolikos Agrinio"},
    {"Platanias", "AO Platanias"},
    {"Guingamp", "EA Guingamp"},
    {"PAS Lamia", "PAS Lamia 1964"},
    {"Marítimo", "CS Marítimo"},
    {"Panthrakikos", "Panthrakikos Komotini"},
    {"Iraklis", "Iraklis Thessaloniki"},
    {"Pescara", "Delfino Pescara 1936"},
    {"De Graafschap", "De Graafschap Doetinchem"},
    {"FC Ingolstadt", "FC Ingolstadt 04"},
    {"Zirka", "Zirka Kropyvnytskyi"},
    {"Enisey", "Enisey Krasnoyarsk"},
    {"Dinamo Zagreb", "GNK Dinamo Zagreb"},
    {"Xanthi", "AO Xanthi"},
    {"Karagümrük", "Fatih Karagümrük"},
    {"Coimbra", "Académica Coimbra"},
    {"Basel", "FC Basel", "FC Basel 1893"},
    {"Red Star", "Red Star Belgrade"},
    {"Malmö", "Malmö FF"},
    {"Young Boys", "BSC Young Boys"},
    {"Greuther Fürth", "SpVgg Greuther Fürth"},
    {"Kallithea", "Athens Kallithea"},
    {"Rosenborg", "Rosenborg Ballklub", "Rosenborg BK"},
    {"Sarpsborg 08", "Sarpsborg 08 Fotballforening"},
    {"Ergotelis", "GS Ergotelis"},
    {"União Madeira", "CF União Madeira (-2021)"},
    {"Stal Kamyanske", "PFK Stal Kamyanske (-2018)"},
    {"Ingulets", "Ingulets Petrove"},
    {"Clermont Foot", "Clermont Foot 63"},
    {"Desna", "Desna Chernigiv"},
    {"Tambov", "PFK Tambov (-2021)"},
    {"Karpaty Lviv", "Karpaty Lviv (-2021)"},
    {"Anzhi", "Anzhi Makhachkala ( -2022)"},
    {"KSC Lokeren", "KSC Lokeren (- 2020)"},
    {"Gaziantepspor", "Gaziantepspor (- 2020)"},
    {"Mordovia", "Mordovia Saransk (-2020)"},
    {"Desportivo Aves", "Desportivo Aves (- 2020)"},
    {"Lierse SK", "Lierse SK (- 2018)"},
    {"Mouscron", "Royal Excel Mouscron (-2022)"},
    {"RAEC Mons", "RAEC Mons (- 2015)"},
    {"Tosno", "FC Tosno (-2018)"},
    {"SC Paderborn", "SC Paderborn 07"},
    {"Metalist Kharkiv", "Metalist Kharkiv (- 2016)"},
    {"Roma", "AS Roma", "Associazione Sportiva Roma"},
    {"Leverkusen", "Bayer 04 Leverkusen", "B. Leverkusen"},
    {"Parma", "Parma Calcio 1913"},
    {"Atromitos", "APS Atromitos Athinon"},
    {"PAOK", "PAOK Salonika"},
    {"Akhmat Grozny", "RFK Akhmat Grozny"},
    {"Alavés", "Deportivo Alavés"},
    {"Stade Brestois", "Stade Brestois 29"},
    {"Troyes", "ESTAC Troyes"},
    {"Ural", "Ural Yekaterinburg"},
    {"Salernitana", "US Salernitana 1919"},
    {"Khimki", "FC Khimki (-2025)"},
    {"Levadiakos", "APO Levadiakos", "APO Levadiakos Football Club"},
    {"Kuban Krasnodar", "Kuban Krasnodar (-2018)"},
    {"Dijon", "Dijon FCO"},
    {"Veres Rivne", "NK Veres Rivne"},
    {"SC Cambuur", "SC Cambuur Leeuwarden"},
    {"Évian", "Thonon Évian Grand Genève FC"},
    {"Partizan", "FK Partizan Belgrade"},
    {"Apollon Smyrnis", "Apollon Smyrni"},
    {"Ionikos Nikeas", "Ionikos Nikea"},
    {"FC Mariupol", "FK Mariupol", "FSC Mariupol"},
    {"Dynamo Moscow", "Dinamo Moscow"},
    {"Beerschot VA", "Beerschot V.A."},
    {"FC Helsingör", "FC Helsingør"},
    {"Niki Volou", "Niki Volos"},
    {"PFC Lviv", "PFK Lviv"},
    {"Yeni Malatyaspor", "Y. Malatyaspor"},
    {"GFC Ajaccio", "G. Ajaccio"},
    {"Aris Saloniki", "Aris Thessalonikis"},
    {"Arm. Bielefeld", "Arminia Bielefeld"},
    {"Sint-Truiden", "Sint-Truidense VV"},
    {"A.G.S Asteras Tripolis", "Asteras Tripoli"},
    {"Aalesund", "Aalesunds FK"},
]
CLUB_NICKNAME_KEYS = {name: f"nickname:{i}" for i, group in enumerate(CLUB_NICKNAME_GROUPS) for name in group}

# "Racing" alone and "Racing Club" would otherwise merge mechanically
# (CLUB_NAME_STRIP_TOKENS includes the generic word "club") - checked and
# rejected: "Racing Club" here is Racing Club de Avellaneda (Argentina,
# league ARG1), while every "Racing" row carries no league at all (an
# uncovered competition), so there's no actual evidence the two are the
# same club, unlike every other CLUB_NAME_STRIP_TOKENS merge (which never
# hit this problem - "club" just happened to be safe for every other name
# it stripped). Force-split rather than trust the generic rule here.
CLUB_NAME_FORCE_SPLIT = {"Racing"}


def build_club_name_aliases(name_mentions):
    """
    Map every to_club_name/from_club_name spelling that's genuinely the
    same real club under a different legal-entity marker, accent encoding
    ("FC Barcelona"/"Barcelona", "Fenerbahçe"/"Fenerbahce"), or verified
    nickname (CLUB_NICKNAME_GROUPS, "Man City"/"Manchester City") to one
    canonical spelling (whichever variant is *longest* - "Tottenham
    Hotspur" over "Tottenham", "Arsenal FC" over "Arsenal" - on the theory
    that the fuller name is the one a reader unfamiliar with the shorthand
    will recognize) - applied to transfers_df, loans_df, and
    comparables["meta"] at startup (see below) so a club's name reads the
    same everywhere on the site, not just on its own /clubs.html report
    card. Without this, transfers_processed.csv's raw Transfermarkt names
    silently split a club like Chelsea or Tottenham across two or more
    identities, each request only ever seeing part of its real history.

    name_mentions is every to_club_name/from_club_name value across both
    transfers_df and loans_df, concatenated - only used as a tiebreaker
    (more mentions wins) on the rare case two variants tie in length; the
    canonical choice itself is decided by length, not frequency.

    The mechanical part (CLUB_NAME_STRIP_TOKENS) is narrow on purpose:
    strips only a fixed set of generic club-entity tokens plus a leading
    "1.FC " prefix, and merges two names only when the *remainder* is
    identical - safe because two *different* real clubs essentially never
    collide once a generic token like "FC" is removed. Two real near-misses
    found doing that check, both excluded rather than merged: "SC Dnipro-1"
    and "Dnipro" (the historical "Dnipro Dnipropetrovsk" club, dissolved
    2020) look like spelling variants but are legally distinct clubs -
    excluded by leaving the "-1" digit as its own token rather than
    stripping bare "1" generically; and "Racing"/"Racing Club" (see
    CLUB_NAME_FORCE_SPLIT).

    CLUB_NICKNAME_GROUPS covers the nickname/official-name pairs that share
    no token at all and so can't be caught mechanically - each entry there
    was individually verified, not guessed from football knowledge alone
    (see that constant's own comment). Still not exhaustive across all
    ~700+ club names here - see README's Known limitations.
    """
    def normalize(name):
        if name in CLUB_NAME_FORCE_SPLIT:
            return f"force_split:{name}"
        if name in CLUB_NICKNAME_KEYS:
            return CLUB_NICKNAME_KEYS[name]
        key = fold_accents(name)
        key = re.sub(r"^1\.?\s*fc\s+", "", key)
        key = re.sub(r"[^a-z0-9 ]", " ", key)
        return " ".join(w for w in key.split() if w not in CLUB_NAME_STRIP_TOKENS)

    counts = name_mentions.value_counts()
    groups = {}
    for name in counts.index:
        groups.setdefault(normalize(name), []).append(name)

    aliases = {}
    for variants in groups.values():
        if len(variants) < 2:
            continue
        # The longest spelling, not the most frequent - "Tottenham Hotspur"
        # over "Tottenham", "Arsenal FC" over "Arsenal" - on the theory that
        # the fuller name is the one a reader unfamiliar with the shorthand
        # will actually recognize. Ties (none found in practice) fall back
        # to whichever variant has more mentions, purely for determinism.
        canonical = max(variants, key=lambda n: (len(n), counts[n]))
        for variant in variants:
            aliases[variant] = canonical
    return aliases


pipeline = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
comparables = joblib.load(os.path.join(MODEL_DIR, "comparables.joblib"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

transfers_df = pd.read_csv(os.path.join(DATA_DIR, "transfers_processed.csv"))
# predicted_score/surprise_delta (see scripts/compute_prediction_surprises.py)
# cover ~93% of transfers - the same required-feature dropna the deployed
# model itself applies during training excludes the rest (a missing origin
# league, fee, or height) - left as NaN for those via how="left" rather than
# dropping the row from transfers_df entirely, since every other endpoint
# still needs it.
prediction_surprises_df = pd.read_csv(os.path.join(DATA_DIR, "prediction_surprises.csv"))
transfers_df = transfers_df.merge(prediction_surprises_df, on=["player_id", "transfer_date"], how="left")
# Lets the Model vs Reality page sort by "most accurate predictions first"
# (order=asc on this column) alongside its overachiever/bust sorts on the
# signed surprise_delta.
transfers_df["abs_surprise_delta"] = transfers_df["surprise_delta"].abs()
loans_df = pd.read_csv(os.path.join(DATA_DIR, "loans_processed.csv"))
players_df = pd.read_csv(os.path.join(DATA_DIR, "players_lookup.csv"))
clubs_df = pd.read_csv(os.path.join(DATA_DIR, "clubs_lookup.csv"))

# Canonicalize club names sitewide - transfers_processed.csv/loans_processed.csv
# have no club_id, so the same real club shows up under several raw
# Transfermarkt spellings (see build_club_name_aliases). Counting mentions
# across both files (not just transfers_df) before picking each group's
# canonical spelling, and rewriting to_club_name/from_club_name in place
# here, before _to_club_fold/_from_club_fold below are computed from them -
# every endpoint that reads either column (Browse, Loans, Surprises,
# Predict's comparables, Club Report Cards) sees the same name for the
# same club, not just whichever one happened to land on this specific row.
CLUB_NAME_ALIASES = build_club_name_aliases(pd.concat([
    transfers_df["to_club_name"], transfers_df["from_club_name"],
    loans_df["to_club_name"], loans_df["from_club_name"],
]))
transfers_df["to_club_name"] = transfers_df["to_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))
transfers_df["from_club_name"] = transfers_df["from_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))
loans_df["to_club_name"] = loans_df["to_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))
loans_df["from_club_name"] = loans_df["from_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))
# comparables["meta"] (train_model.py's nearest-neighbors index, shown as
# "comparable historical transfers" on Predict/Compare) is a separate
# artifact built straight from the raw CSV, not derived from transfers_df -
# needs the same rewrite or its club names would be the only place on the
# site still showing the old, split spellings.
comparables["meta"]["to_club_name"] = comparables["meta"]["to_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))
comparables["meta"]["from_club_name"] = comparables["meta"]["from_club_name"].map(lambda n: CLUB_NAME_ALIASES.get(n, n))

players_df["_name_fold"] = players_df["name"].map(fold_accents)
clubs_df["_name_fold"] = clubs_df["name"].map(fold_accents)
transfers_df["_name_fold"] = transfers_df["name"].map(fold_accents)
transfers_df["_from_club_fold"] = transfers_df["from_club_name"].map(fold_accents)
transfers_df["_to_club_fold"] = transfers_df["to_club_name"].map(fold_accents)
loans_df["_name_fold"] = loans_df["name"].map(fold_accents)
loans_df["_from_club_fold"] = loans_df["from_club_name"].map(fold_accents)
loans_df["_to_club_fold"] = loans_df["to_club_name"].map(fold_accents)
competitions_df = pd.read_csv(os.path.join(DATA_DIR, "competitions_lookup.csv"))
_dup_names = competitions_df["name"][competitions_df["name"].duplicated(keep=False)]
competitions_df["display_name"] = competitions_df.apply(
    lambda r: f"{r['name']} ({r['country_name']})" if r["name"] in _dup_names.values and pd.notna(r["country_name"]) else r["name"],
    axis=1,
)
LEAGUE_NAMES = dict(zip(competitions_df["competition_id"], competitions_df["display_name"]))

league_baselines_df = pd.read_csv(os.path.join(DATA_DIR, "league_baselines.csv"))
LEAGUE_POSITION_BASELINE = {
    (r["competition_id"], r["position"]): r["ga_p90_baseline"] for _, r in league_baselines_df.iterrows()
}


def league_display_name(competition_id):
    """
    Look up a league's display name, or "Unknown league" for the rare club
    that isn't in clubs.csv at all (~7 in transfers_processed.csv, ~23 in
    the smaller loans_processed.csv - obscure clubs the dataset never
    populated a domestic_competition_id for), where competition_id itself
    is NaN. LEAGUE_NAMES.get(competition_id, competition_id) alone would
    return that same NaN back out (a float NaN never equals itself, so the
    dict lookup always misses), which isn't JSON-serializable and 500s any
    endpoint that returns it.
    """
    if pd.isna(competition_id):
        return "Unknown league"
    return LEAGUE_NAMES.get(competition_id, competition_id)


def league_ga_baseline(competition_id, position):
    """Goal contributions/90 baseline for this (league, position), falling back to the position's overall average."""
    return LEAGUE_POSITION_BASELINE.get(
        (competition_id, position),
        LEAGUE_POSITION_BASELINE.get(("_default", position), 0.3),
    )


MIN_CLUB_TRANSFERS = 5
MIN_CLUB_RESALES = 3


def build_club_report_cards(df):
    """
    One row per club name that appears as a buyer and/or seller among
    permanent transfers, aggregating both sides independently - a club's
    incoming and outgoing counts are usually very different, so each side
    gets its own sample size rather than one blended number. Club identity
    here is purely the from_club_name/to_club_name *string*:
    transfers_processed.csv has no club_id column at all, and its raw
    Transfermarkt names ("Man City") don't reliably match clubs_lookup.csv's
    own naming ("Manchester City") - a real, unresolved gap between the two
    files - so this deliberately never joins clubs_lookup.csv, rather than
    risk manufacturing false matches between two differently-spelled clubs.

    Resale profit (next_transfer_fee - transfer_fee, see build_dataset.py) is
    computed once per row for its *to_club_name* - the buyer, since
    next_transfer_fee is what a third club later paid *that* buyer for the
    same player (df.groupby("player_id")["transfer_fee"].shift(-1)), not
    anything about the row's from_club_name/seller. So both "recruitment"
    stats (avg_incoming_score, total_spent) and "buy-develop-resell" stats
    (resales_count, avg_resale_profit_pct) come from the same to_club_name
    group here - grouping resale by from_club_name instead (an earlier,
    wrong version of this function) attributed a club's *buyer's* eventual
    resale to the selling club, e.g. crediting Manchester United's later
    resale of a player to whichever club sold that player to United
    originally.

    The from_club_name side instead captures a different, genuinely
    seller-side signal: how departing players went on to perform at their
    *next* club (avg_departure_score) - a development/retention read (are
    the players we let go thriving elsewhere?), not a financial one.

    Precomputed once at startup into club_report_cards_df below; the
    /api/clubs/leaderboard endpoint only filters/sorts/paginates it.
    """
    def transfer_highlight(g, best):
        """
        The best (best=True) or worst transfer in this group, by
        success_score. Carries player_id alongside transfer_date so the
        frontend can open this exact transfer's full breakdown card via
        /api/transfers/detail, the same (player_id, transfer_date) key
        every other click-to-view-card page already uses.
        """
        r = g.loc[g["success_score"].idxmax() if best else g["success_score"].idxmin()]
        return {
            "player_id": int(r["player_id"]), "name": r["name"],
            "success_score": float(r["success_score"]), "transfer_date": str(r["transfer_date"])[:10],
        }

    def flip_highlight(resold, profit, best):
        """
        The most (best=True) or least profitable buy-then-resell in this
        already-resold-only group. `profit` is precomputed once by the
        caller and shared across both calls, rather than each call redoing
        the same subtraction. Carries player_id for the same click-to-view
        reason as transfer_highlight above - a flip is still just a row of
        transfers_df, the same table /api/transfers/detail looks up.
        """
        r = resold.loc[profit.idxmax() if best else profit.idxmin()]
        return {
            "player_id": int(r["player_id"]), "name": r["name"], "bought_from": r["from_club_name"],
            "fee_paid": float(r["transfer_fee"]) if pd.notna(r["transfer_fee"]) else 0.0,
            "fee_received": float(r["next_transfer_fee"]),
            "transfer_date": str(r["transfer_date"])[:10],
        }

    rows = {}
    for club, g in df.groupby("to_club_name"):
        mode = g["to_domestic_competition_id"].mode()
        resold = g[g["has_resale_data"]]
        profit = resold["next_transfer_fee"] - resold["transfer_fee"].fillna(0)
        rows[club] = {
            "club_name": club,
            "league_id": mode.iat[0] if not mode.empty else None,
            "transfers_in": len(g),
            "avg_incoming_score": float(g["success_score"].mean()),
            "total_spent": float(g["transfer_fee"].fillna(0).sum()),
            "best_signing": transfer_highlight(g, best=True),
            "worst_signing": transfer_highlight(g, best=False),
            "resales_count": len(resold),
            "avg_resale_profit_pct": float(resold["resale_profit_pct"].mean()) if len(resold) else None,
            "total_resale_profit": float(profit.sum()),
            "best_flip": flip_highlight(resold, profit, best=True) if len(resold) else None,
            "worst_flip": flip_highlight(resold, profit, best=False) if len(resold) else None,
        }

    for club, g in df.groupby("from_club_name"):
        # A club that's only ever bought, never sold (or vice versa) still
        # needs a row with the other side's fields defaulted - setdefault
        # rather than assuming every club showed up in the loop above.
        row = rows.setdefault(club, {
            "club_name": club, "league_id": None, "transfers_in": 0,
            "avg_incoming_score": None, "total_spent": 0.0,
            "best_signing": None, "worst_signing": None,
            "resales_count": 0, "avg_resale_profit_pct": None,
            "total_resale_profit": 0.0, "best_flip": None, "worst_flip": None,
        })
        if row["league_id"] is None:
            mode = g["from_domestic_competition_id"].mode()
            row["league_id"] = mode.iat[0] if not mode.empty else None
        row["transfers_out"] = len(g)
        row["avg_departure_score"] = float(g["success_score"].mean())
        row["best_departure"] = transfer_highlight(g, best=True)
        row["worst_departure"] = transfer_highlight(g, best=False)

    for row in rows.values():
        row.setdefault("transfers_out", 0)
        row.setdefault("avg_departure_score", None)
        row.setdefault("best_departure", None)
        row.setdefault("worst_departure", None)

    out = pd.DataFrame(rows.values())
    out["_name_fold"] = out["club_name"].map(fold_accents)
    return out


# transfers_df's to_club_name/from_club_name are already canonical (see
# CLUB_NAME_ALIASES above, applied right after load) - no remapping needed here.
club_report_cards_df = build_club_report_cards(transfers_df)


MIN_LEAGUE_TRANSFERS = 15
MIN_LEAGUE_YEAR_SAMPLE = 5
TREND_WINDOW_YEARS = 3


def build_league_trends(df):
    """
    For every destination league with at least MIN_LEAGUE_TRANSFERS
    permanent transfers, compare its earliest TREND_WINDOW_YEARS complete
    years against its most recent TREND_WINDOW_YEARS complete years - "is
    spending outpacing performance" needs two real eras to compare, not
    just one all-time average. The dataset's own most recent year is
    always excluded from that comparison (and from the by-year chart data)
    as an in-progress season - checked directly that it runs far below a
    normal year's transfer count for every league (e.g. ~12 vs ~90/year
    for the Premier League), which would otherwise show up as a sudden,
    misleading collapse at the end of every trend line.

    by_year only includes years with at least MIN_LEAGUE_YEAR_SAMPLE
    transfers in that specific league - even a "major" league has some
    thin early years, and a single-digit sample swings an average wildly
    (see MIN_LEAGUE_SAMPLE in train_model.py for the same idea applied to
    a whole-league baseline instead of one league-year).

    Precomputed once at startup into league_trends_df below; the
    /api/leagues/trends endpoint only filters/sorts it - each row already
    carries its own by_year series, so the frontend's trend chart needs no
    second request.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["transfer_date"]).dt.year
    current_year = int(df["year"].max())

    rows = []
    for league_id, g in df.groupby("to_domestic_competition_id"):
        if len(g) < MIN_LEAGUE_TRANSFERS:
            continue
        row = {
            "league_id": league_id,
            "transfers": len(g),
            "avg_score": float(g["success_score"].mean()),
            "avg_fee": float(g["transfer_fee"].fillna(0).mean()),
            "early_years": None, "recent_years": None,
            "early_avg_fee": None, "recent_avg_fee": None, "fee_growth_pct": None,
            "early_avg_score": None, "recent_avg_score": None, "score_change": None,
            "by_year": [],
        }

        complete = g[g["year"] < current_year]
        years = sorted(complete["year"].unique())
        if len(years) >= 2 * TREND_WINDOW_YEARS:
            # Below this, there isn't enough real history for two
            # non-overlapping eras to mean anything - the league still
            # gets a row (basic stats above), just with no trend to show.
            early_years, recent_years = years[:TREND_WINDOW_YEARS], years[-TREND_WINDOW_YEARS:]
            early = complete[complete["year"].isin(early_years)]
            recent = complete[complete["year"].isin(recent_years)]

            by_year = (
                complete.groupby("year")
                .agg(avg_score=("success_score", "mean"), avg_fee=("transfer_fee", lambda s: s.fillna(0).mean()), count=("success_score", "size"))
                .reset_index()
            )
            by_year = by_year[by_year["count"] >= MIN_LEAGUE_YEAR_SAMPLE]

            early_avg_fee, recent_avg_fee = early["transfer_fee"].fillna(0).mean(), recent["transfer_fee"].fillna(0).mean()
            early_avg_score, recent_avg_score = early["success_score"].mean(), recent["success_score"].mean()

            row.update({
                "early_years": f"{min(early_years)}–{max(early_years)}",
                "recent_years": f"{min(recent_years)}–{max(recent_years)}",
                "early_avg_fee": float(early_avg_fee),
                "recent_avg_fee": float(recent_avg_fee),
                "fee_growth_pct": float((recent_avg_fee - early_avg_fee) / early_avg_fee * 100) if early_avg_fee > 0 else None,
                "early_avg_score": float(early_avg_score),
                "recent_avg_score": float(recent_avg_score),
                "score_change": float(recent_avg_score - early_avg_score),
                "by_year": [
                    {"year": int(r["year"]), "avg_score": float(r["avg_score"]), "avg_fee": float(r["avg_fee"]), "count": int(r["count"])}
                    for _, r in by_year.iterrows()
                ],
            })
        rows.append(row)
    return pd.DataFrame(rows)


league_trends_df = build_league_trends(transfers_df)

# Every (player_id, name, _name_fold) mention across both transfers_df and
# loans_df, deduplicated to one row per player - built once here rather
# than in search_players_for_career itself, which used to redo this same
# concat+dedup over ~12,700 rows on every request (the Player Timelines
# search box fires it on nearly every keystroke).
PLAYER_CAREER_SEARCH_DF = pd.concat([
    transfers_df[["player_id", "name", "_name_fold"]],
    loans_df[["player_id", "name", "_name_fold"]],
]).drop_duplicates(subset="player_id")


NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

EXAMPLE_TRANSFER_KEYS = [
    ("Erling Haaland", "Man City"),
    ("Ousmane Dembélé", "Barcelona"),
    ("Cole Palmer", "Chelsea"),
    ("Jadon Sancho", "Man Utd"),
]

POSITION_PLURAL = {
    "Attack": "attackers", "Midfield": "midfielders",
    "Defender": "defenders", "Goalkeeper": "goalkeepers",
}


def resale_outcome_phrase(fee_paid, fee_received):
    """Plain-English verdict on a resale: profit, loss, or break-even, given what the club paid vs. what it later sold the player for."""
    fee_paid = 0 if pd.isna(fee_paid) else fee_paid
    if fee_received > fee_paid:
        return "a profitable flip for the club, regardless of on-pitch performance"
    if fee_received < fee_paid:
        return "sold for less than the club paid, a loss independent of on-pitch performance"
    return "resold for the same fee paid, breaking even"


def describe_resale_profit(r, eur_m):
    """Build the 'Resale profit' breakdown row's description: the fee paid/received and the outcome (profit/loss)."""
    outcome = resale_outcome_phrase(r["transfer_fee"], r["next_transfer_fee"])
    return f"Bought for {eur_m(r['transfer_fee'])}, later resold for {eur_m(r['next_transfer_fee'])}, {outcome}."


FOTMOB_COMPONENT_LABELS = {
    "rating": "FotMob rating",
    "attacking": "Attacking",
    "defensive": "Defending",
    "possession": "Possession",
}


def describe_fotmob_component(component, r, position_plural, is_loan=False):
    """
    Build one FotMob-derived breakdown row's description (rating/attacking/
    defensive/possession - see compute_fotmob_component_pcts in
    build_dataset.py for why they're kept separate rather than blended into
    one number). The underlying per-90 rates driving "attacking" aren't
    stored directly (only the raw season totals are), so chances-created/90
    is recomputed here the same way build_dataset.py derives it - from
    fotmob_total_att_assist and fotmob_total_minutes. Shared between
    describe_components() (permanent transfers) and
    describe_loan_components() (loans) - is_loan only changes the wording
    ("on loan at"/"other loan spells" vs. "at"/"other {position}"), purely
    cosmetic: unlike every other component, rating/attacking/defensive/
    possession are ranked against transfers and loans *combined* (see
    attach_fotmob_components in build_dataset.py), so the number itself
    means the same thing on both kinds of card - only the phrasing differs.

    Each bucket's _pct is an average of whichever of its underlying stats
    are actually available (see compute_fotmob_component_pcts), so a
    component can be "known" even when one specific stat behind it isn't -
    e.g. attacking_pct valid from goals/chances-created alone with xG/xA
    missing that season. parts() builds the description from only the
    sub-stats that are actually present, instead of formatting a NaN
    straight into the string ("nan xG/90").

    "attacking" is also folded together with the Transfermarkt goal-
    contributions number (see fold_perf_level_into_attacking in
    build_dataset.py) - whenever this row is shown at all, that folding
    happened, so post_ga_p90 is always included alongside the FotMob
    sub-stats here, not just when it happens to be missing.

    Returns {"description": ..., "stats": [...] or None} rather than a
    single string - two or more sub-stats read as a comma-separated wall of
    numbers as one sentence, so those render as a bulleted list in the
    frontend tooltip instead (see renderBreakdown() in app.js/compare.js/
    browse.js/loans.js), with "description" holding just the setup line.
    A single sub-stat (rating) has nothing to bullet, so it stays one plain
    sentence and "stats" is None - the frontend's cue to skip the list.
    """
    seasons = int(r["fotmob_seasons_used"])
    season_note = "1 season" if seasons == 1 else f"{seasons} seasons"
    minutes_per_90 = max(r["fotmob_total_minutes"] / 90, 1)
    chances_created_p90 = r["fotmob_total_att_assist"] / minutes_per_90

    def parts(*pairs):
        """pairs is (value, format-string) tuples - drop any whose value is NaN, keep the rest as a list of formatted strings."""
        return [fmt.format(v) for v, fmt in pairs if pd.notna(v)]

    if component == "rating":
        detail = parts((r["fotmob_rating"], "{:.2f} average match rating"))
    elif component == "attacking":
        detail = parts(
            (r["post_ga_p90"], "{:.2f} goal contributions/90"),
            (r["fotmob_goals_per_90"], "{:.2f} goals/90"),
            (r["fotmob_expected_goals_per_90"], "{:.2f} xG/90"),
            (r["fotmob_expected_assists_per_90"], "{:.2f} xA/90"),
            (chances_created_p90, "{:.1f} chances created/90"),
        )
    elif component == "defensive" and r["position"] == "Goalkeeper":
        detail = parts(
            (r["fotmob_saves"], "{:.1f} saves/90"),
            (r["fotmob__save_percentage"], "{:.0f}% save rate"),
            (r["fotmob_goals_conceded"], "{:.1f} goals conceded/90"),
        )
    elif component == "defensive":
        detail = parts(
            (r["fotmob_total_tackle"], "{:.1f} tackles/90"),
            (r["fotmob_interception"], "{:.1f} interceptions/90"),
            (r["fotmob_effective_clearance"], "{:.1f} clearances/90"),
            (r["fotmob_ball_recovery"], "{:.1f} recoveries/90"),
        )
    else:  # possession
        detail = parts(
            (r["fotmob_accurate_pass"], "{:.1f} accurate passes/90"),
            (r["fotmob_won_contest"], "{:.1f} dribbles/90"),
        )

    # "vs. other {position}" here (not "other loan spells"/"other transfers"
    # like the rest of each card's rows) because these four components are
    # ranked against transfers and loans combined - see
    # attach_fotmob_components in build_dataset.py.
    at_club = f"on loan at {r['to_club_name']}" if is_loan else f"at {r['to_club_name']}"
    if len(detail) > 1:
        return {
            "description": f"Over {season_note} {at_club} (ranked vs. other {position_plural}, league-adjusted):",
            "stats": detail,
        }
    return {
        "description": (
            f"{detail[0]}, averaged across {season_note} {at_club}, "
            f"ranked vs. other {position_plural} after adjusting for the league's own average"
        ),
        "stats": None,
    }


def ordinal(n):
    """1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th", ..., 11/12/13 -> "11th"/"12th"/"13th" (the 11-13 exception to the usual 1/2/3 pattern)."""
    n = int(round(n))
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def eur_m(v):
    """Format a euro amount for display, e.g. 50_000_000 -> "€50m", 300_000 -> "€0.3m", 2_049_250_000 -> "€2.05b", 0/NaN -> "free". No individual transfer fee reaches a billion, but a club's aggregate spend/resale-profit total (see build_club_report_cards) can."""
    if pd.isna(v) or v == 0:
        return "free"
    if abs(v) >= 1_000_000_000:
        return f"€{v / 1_000_000_000:.2f}b"
    millions = v / 1_000_000
    # Sub-million fees are common (e.g. a €300k sale) - one decimal place
    # keeps them from rounding down to a misleading "€0m".
    return f"€{millions:.1f}m" if millions < 1 else f"€{millions:.0f}m"


def describe_components(r):
    """
    Build the full "why this score" breakdown for one row of
    transfers_processed.csv: a list of {label, value, description} dicts,
    one per success-score component actually used for this transfer (4
    always, plus one of "G/A per 90"/"Attacking" - see below - plus up to
    3 more FotMob-derived rows - possession/defending/rating, each
    independently shown only when its own has_*_data flag is true - and
    "Resale profit" when has_resale_data is true - so 5 to 10 rows total).
    Fixed display order: Transfer fee, Value change, Resale profit, G/A
    per 90, G/A change, Attacking, Possession, Defending, FotMob rating,
    Playing time - each entry above simply drops out of that order when
    its own flag is false. Used by both /api/examples and
    /api/transfers/detail via build_transfer_card().

    "G/A per 90" and "Attacking" (the FotMob "attacking" bucket) are
    mutually exclusive, not both-or-neither: they measure the same
    underlying thing (attacking output), so build_dataset.py folds them
    into one weighted bucket instead of double-counting the signal (see
    fold_perf_level_into_attacking) - whichever one was actually used for
    this row's score is the one shown here. has_attacking_data is that
    same switch: true means the fold happened and "Attacking" (in the
    FotMob block below) carries the combined number; false means FotMob
    had nothing for this transfer and "G/A per 90" alone carries it, same
    as before FotMob data existed.
    """
    position_plural = POSITION_PLURAL.get(r["position"], r["position"])
    to_league = league_display_name(r["to_domestic_competition_id"])
    # Goal-contribution signals (G/A per 90, G/A change, the FotMob
    # "Attacking" bucket) are meaningless for a goalkeeper - virtually none
    # ever register a goal contribution, and their weight in the score is
    # already 0% for exactly that reason (see README's "Weights" section) -
    # so none of the three are worth a row here. "Defending" is relabeled
    # "Goalkeeping" for the same position, since describe_fotmob_component
    # already swaps in shot-stopping stats (saves, save %, goals conceded)
    # for that bucket rather than tackles/interceptions.
    is_goalkeeper = r["position"] == "Goalkeeper"
    fotmob_components = ("possession", "defensive", "rating") if is_goalkeeper else ("attacking", "possession", "defensive", "rating")
    return [
        {
            "label": "Transfer fee",
            "value": round(float(r["value_for_money_pct"]), 1),
            "description": (
                f"{eur_m(r['transfer_fee'])} fee vs. {eur_m(r['value_before'])} market value at the time, "
                f"weighed against on-pitch performance relative to what the fee implied"
            ),
        },
        {
            "label": "Value change",
            "value": round(float(r["value_growth_pct"]), 1),
            "description": (
                f"{eur_m(r['value_before'])} → peaked at {eur_m(r['value_peak'])} (now {eur_m(r['value_after'])})"
                if r["value_peak"] > r["value_after"] * 1.05
                else f"{eur_m(r['value_before'])} → {eur_m(r['value_after'])} market value"
            ),
        },
    ] + ([
        {
            "label": "Resale profit",
            "value": round(float(r["resale_profit_pct"]), 1),
            "description": describe_resale_profit(r, eur_m),
        },
    ] if bool(r["has_resale_data"]) else []) + ([] if is_goalkeeper or bool(r["has_attacking_data"]) else [
        {
            "label": "G/A per 90",
            "value": round(float(r["perf_level_pct"]), 1),
            "description": (
                f"{r['post_ga_p90']:.2f} goal contributions/90 at {r['to_club_name']} "
                f"({r['post_ga_p90_vs_league']:.1f}x the {to_league} average for {position_plural}), "
                f"ranked vs. other {position_plural}"
            ),
        },
    ]) + ([] if is_goalkeeper else [
        {
            "label": "G/A change",
            "value": round(float(r["perf_delta_pct"]), 1),
            "description": (
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x, beating the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high (some pullback from a peak is normal)"
                if r["post_ga_p90_vs_league"] >= r["expected_post_ga_p90_vs_league"] else
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x, below the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high"
            ),
        },
    ]) + [
        {
            "label": "Goalkeeping" if (component == "defensive" and is_goalkeeper) else FOTMOB_COMPONENT_LABELS[component],
            "value": round(float(r[f"{component}_pct"]), 1),
            **describe_fotmob_component(component, r, position_plural),
        }
        for component in fotmob_components
        if bool(r[f"has_{component}_data"])
    ] + [
        {
            "label": "Playing time",
            "value": round(float(r["playing_time_pct"]), 1),
            "description": (
                f"{int(r['post_apps'])} of {int(r['team_games_in_tenure'])} games "
                f"{r['to_club_name']} played during the tenure "
                f"({r['pct_team_games_played'] * 100:.0f}% - captures injuries/rotation, "
                f"blended with raw appearance count for sustained presence)"
            ),
        },
    ]

FEATURE_LABELS = {
    "age_at_transfer": "Age at transfer",
    "height_vs_position": "Height vs. position average",
    "pre_apps": "Recent appearances",
    "pre_minutes": "Recent minutes played",
    "pre_goals_p90": "Recent goals per 90",
    "pre_ga_p90": "Recent goal contributions per 90",
    "pre_mins_per_app": "Minutes per appearance",
    "log_transfer_fee": "Transfer fee",
    "log_value_before": "Market value before the move",
    "fee_to_value_ratio": "Fee relative to market value",
    "club_quality_ratio": "Step up/down in club quality",
    "log_from_club_value": "Origin club's squad value",
    "log_to_club_value": "Destination club's squad value",
    "position": "Position",
    "sub_position": "Specific role",
    "foot": "Preferred foot",
    "from_domestic_competition_id": "Origin league",
    "to_domestic_competition_id": "Destination league",
    "pre_fotmob_rating_pct": "Recent FotMob rating",
    "pre_fotmob_attacking_pct": "Recent attacking output",
    "pre_fotmob_defensive_pct": "Recent defensive work",
    "pre_fotmob_possession_pct": "Recent passing/possession",
    # The 12 raw pre_fotmob_* stats and the 4 has_pre_*_data flags don't
    # need an entry here - the model trains on the 4 composites above (see
    # train_model.py:PRETRANSFER_FOTMOB_COMPOSITES), and the raw stats only
    # ever appear as descriptive context underneath a composite's own
    # explanation row (see PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES),
    # never as their own standalone contribution.
}

# Which raw stats to show as descriptive context under each composite
# feature's explanation (see explain_prediction) - "72nd percentile among
# forwards" alone doesn't say what that means concretely, same reasoning as
# describe_fotmob_component's bulleted breakdown for the historical score.
# Purely descriptive now, not tied to the swap calculation the way the
# grouped-feature version of this used to be - each composite is already
# one real, independent model feature (see train_model.py), so
# explain_prediction's normal per-feature leave-one-out swap already gives
# an honest contribution without any special combining logic.
# defensive's raw stats are position-dependent (an outfield player's real
# tackles/interceptions vs. a goalkeeper's saves/save%/goals-conceded) -
# "goalkeeper_raw_features" picked instead of "raw_features" when the
# player being explained is a goalkeeper.
PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES = {
    "pre_fotmob_rating_pct": {"raw_features": ["pre_fotmob_rating"]},
    "pre_fotmob_attacking_pct": {
        "raw_features": ["pre_fotmob_expected_goals_per_90", "pre_fotmob_expected_assists_per_90", "pre_fotmob_chances_created_p90"],
    },
    "pre_fotmob_defensive_pct": {
        "raw_features": ["pre_fotmob_total_tackle", "pre_fotmob_interception", "pre_fotmob_effective_clearance", "pre_fotmob_ball_recovery"],
        "goalkeeper_label": "Recent shot-stopping",
        "goalkeeper_raw_features": ["pre_fotmob_saves", "pre_fotmob__save_percentage", "pre_fotmob_goals_conceded"],
    },
    "pre_fotmob_possession_pct": {"raw_features": ["pre_fotmob_accurate_pass", "pre_fotmob_won_contest"]},
}

# has_pre_*_data flags aren't shown as their own explanation row (data-
# quality flags, not football signal) - each composite's own row says
# plainly when it has no real underlying data instead (see
# explain_prediction). Keyed by composite feature name so
# resolve_reference/the description logic can look up the right flag for
# whichever composite it's currently handling.
PRETRANSFER_FOTMOB_HAS_DATA_FLAGS = {
    "pre_fotmob_rating_pct": "has_pre_rating_data",
    "pre_fotmob_attacking_pct": "has_pre_attacking_data",
    "pre_fotmob_defensive_pct": "has_pre_defensive_data",
    "pre_fotmob_possession_pct": "has_pre_possession_data",
}

# Short per-stat labels for a composite's descriptive bulleted breakdown
# (see explain_prediction) - FEATURE_LABELS' full "Recent xG per 90" is
# redundant once it's already a bullet under a "Recent attacking output"
# heading, so these stay terse.
PRETRANSFER_FOTMOB_STAT_LABELS = {
    "pre_fotmob_expected_goals_per_90": "xG",
    "pre_fotmob_expected_assists_per_90": "xA",
    "pre_fotmob_chances_created_p90": "Chances created",
    "pre_fotmob_accurate_pass": "Accurate passes",
    "pre_fotmob_won_contest": "Successful dribbles",
    "pre_fotmob_total_tackle": "Tackles",
    "pre_fotmob_interception": "Interceptions",
    "pre_fotmob_effective_clearance": "Clearances",
    "pre_fotmob_ball_recovery": "Recoveries",
    "pre_fotmob_saves": "Saves",
    "pre_fotmob__save_percentage": "Save rate",
    "pre_fotmob_goals_conceded": "Goals conceded",
}


PRETRANSFER_FOTMOB_PER90_FEATURES = {
    "pre_fotmob_expected_goals_per_90", "pre_fotmob_expected_assists_per_90", "pre_fotmob_chances_created_p90",
    "pre_fotmob_accurate_pass", "pre_fotmob_won_contest", "pre_fotmob_total_tackle", "pre_fotmob_interception",
    "pre_fotmob_effective_clearance", "pre_fotmob_ball_recovery", "pre_fotmob_saves", "pre_fotmob_goals_conceded",
}

LOG_FEATURES = {"log_transfer_fee", "log_value_before", "log_from_club_value", "log_to_club_value"}


def format_feature_value(feat, value, context=None):
    """
    Render one model feature's raw value in human-readable form for the
    prediction explanation (e.g. a log-transformed fee back to "€50.0m", a
    league code to its display name). `context` (the request's position and
    origin league) is only used for pre_ga_p90, to append a league-relative
    "(X.Xx their current league's average)" note - see league_ga_baseline.
    """
    if feat in LOG_FEATURES:
        value = np.expm1(value)
        if feat == "log_transfer_fee" and value == 0:
            return "free"
        return f"€{value / 1_000_000:.1f}m"
    if feat == "age_at_transfer":
        return f"{value:.1f} yrs"
    if feat == "height_vs_position":
        return f"{value:+.0f} cm"
    if feat == "pre_ga_p90":
        base = f"{value:.2f} per 90"
        if context:
            baseline = league_ga_baseline(context.get("from_domestic_competition_id"), context.get("position"))
            if baseline > 0:
                base += f" ({value / baseline:.1f}x their current league's average)"
        return base
    if feat == "pre_goals_p90":
        return f"{value:.2f} per 90"
    if feat == "pre_apps":
        return f"{value:.0f} apps"
    if feat == "pre_minutes":
        return f"{value:.0f} mins"
    if feat == "pre_mins_per_app":
        return f"{value:.0f} min/app"
    if feat in ("fee_to_value_ratio", "club_quality_ratio"):
        return f"{value:.2f}×"
    if feat in ("from_domestic_competition_id", "to_domestic_competition_id"):
        return LEAGUE_NAMES.get(value, value)
    if feat == "pre_fotmob_rating":
        return f"{value:.2f}"
    if feat == "pre_fotmob__save_percentage":
        return f"{value:.0f}%"
    if feat in PRETRANSFER_FOTMOB_PER90_FEATURES:
        return f"{value:.2f} per 90"
    return str(value)


def league_context_note(feat, actual_league, reference_league):
    """
    Explain WHY one league scores differently than another in a prediction
    explanation, instead of a bare "vs. a typical transfer's Premier
    League" swing that reads as "moving to Spain is inherently better".
    The real driver is almost entirely value_for_money, not on-pitch
    difficulty: Premier League clubs have historically paid a much larger
    premium over market value than clubs in every other major league (mean
    fee/value 1.55x vs. La Liga's 1.01x - see league_fee_ratio_baseline_to
    in train_model.py). Returns "" (falls back to the plain swing-only
    explanation) when either league is too thin a sample to trust - such
    leagues are simply absent from the baseline dicts (see
    MIN_LEAGUE_SAMPLE in train_model.py).
    """
    baseline = metadata[
        "league_success_baseline_to" if feat == "to_domestic_competition_id" else "league_success_baseline_from"
    ]
    if actual_league not in baseline or reference_league not in baseline:
        return ""
    verb = "to" if feat == "to_domestic_competition_id" else "leaving"
    note = (
        f": transfers {verb} {league_display_name(actual_league)} have historically averaged "
        f"{baseline[actual_league]} vs. {baseline[reference_league]} for {league_display_name(reference_league)}"
    )
    if feat == "to_domestic_competition_id":
        fee_baseline = metadata["league_fee_ratio_baseline_to"]
        if actual_league in fee_baseline and reference_league in fee_baseline:
            note += (
                f", largely reflecting fee premiums paid there "
                f"({fee_baseline[actual_league]:.2f}x market value on average vs. {fee_baseline[reference_league]:.2f}x)"
            )
    return note


CATEGORY_BASELINE_NOTE_TAIL = {
    "position": "a real baseline gap in this dataset, not a claim that one position is inherently a better transfer bet",
    "sub_position": "a real baseline gap in this dataset at a finer-grained role level, not a claim that one role is inherently a better bet",
    "foot": "a small, real gap in the data with no confirmed football mechanism behind it - worth reading skeptically",
}

# "left"/"right"/"both" read oddly as a bare noun ("left transfers") the way
# "Attack"/"Centre-Forward" do - display forms so category_baseline_note's
# "{X} transfers" phrasing reads naturally for foot too.
FOOT_DISPLAY = {"left": "Left-footed", "right": "Right-footed", "both": "Two-footed"}


def category_baseline_note(feat, actual_value, reference_value):
    """
    Add the real historical average success_score for a categorical
    feature's actual vs. reference value (position/sub_position/foot),
    instead of an unexplained swing against whichever category happens to
    be the dataset's mode (e.g. "Attack vs. a typical transfer's Defender"
    with nothing to say why). Unlike league_context_note, none of these
    three have a checked, verified mechanism behind the gap - position and
    sub_position plausibly trace back to the historical formula's own
    per-position weighting, but that isn't confirmed, and foot's gap has no
    football explanation found at all - so CATEGORY_BASELINE_NOTE_TAIL
    states the real number without inventing a cause, hedging explicitly
    for "foot" since that gap is smallest and least explicable. Returns ""
    when either category is too thin a sample to trust (missing from the
    baseline dict - see MIN_LEAGUE_SAMPLE in train_model.py).
    """
    baseline = metadata[f"{feat}_success_baseline"]
    if actual_value not in baseline or reference_value not in baseline:
        return ""
    display = FOOT_DISPLAY if feat == "foot" else {}
    actual_label = display.get(actual_value, actual_value)
    reference_label = display.get(reference_value, reference_value)
    return (
        f": {actual_label} transfers have historically averaged {baseline[actual_value]} "
        f"vs. {baseline[reference_value]} for {reference_label} transfers - {CATEGORY_BASELINE_NOTE_TAIL[feat]}"
    )


def club_value_rating_note(direction, actual_log_value, reference_log_value, reference_display):
    """
    Explain WHY origin/destination squad value moves the prediction,
    instead of a bare "€900m vs. a typical attacker's €172m" swing that
    never says what a bigger club value actually buys a transfer. Checked
    directly against every success_score component: log(club value)
    correlates with post-move rating_pct far more than with any other
    component (destination: r=0.29 vs. 0.23 for attacking, 0.22 for
    possession, -0.07 for defensive; origin: r=0.19, same pattern, weaker) -
    so the story is specifically about post-move rating, not attacking
    output or playing time. `club_value_rating_regression` (rating_pct ~
    log(club value), fit on the training data - see train_model.py) lets
    this quote the actual percentile gap implied by *this* prediction's own
    values rather than one baked-in number for everyone. actual_log_value/
    reference_log_value are already log1p-transformed (the model's own
    log_to_club_value/log_from_club_value feature), matching what the
    regression was fit against, so no conversion is needed here.
    Returns "" when the two implied percentiles are within 4 points of each
    other - not enough of a gap for the note to say anything a reader
    couldn't already tell from the swing itself.
    """
    reg = metadata["club_value_rating_regression"]["to" if direction == "to" else "from"]
    pred_actual = float(np.clip(reg["slope"] * actual_log_value + reg["intercept"], 0, 100))
    pred_reference = float(np.clip(reg["slope"] * reference_log_value + reg["intercept"], 0, 100))
    if abs(pred_actual - pred_reference) < 4:
        return ""
    if direction == "to":
        return (
            f": moving to a squad valued this highly has historically come with a stronger post-move "
            f"rating - signings there average around the {ordinal(pred_actual)} percentile, vs. the "
            f"{ordinal(pred_reference)} percentile at a club valued like {reference_display}, likely "
            f"reflecting the quality of teammates and system a wealthier club can offer"
        )
    return (
        f": players leaving a squad valued this highly have historically gone on to rate around the "
        f"{ordinal(pred_actual)} percentile at their new club, vs. the {ordinal(pred_reference)} percentile "
        f"for those leaving a club valued like {reference_display}, likely reflecting the caliber of "
        f"player a club that size already tends to have"
    )


class PredictRequest(BaseModel):
    """
    A hypothetical transfer to score: a player's pre-transfer profile
    (age/position/recent performance/market value) plus the destination
    club and fee. Mirrors NUMERIC_FEATURES/CATEGORICAL_FEATURES in
    train_model.py after the derived columns are added by build_feature_row.
    """
    age_at_transfer: float = Field(..., ge=15, le=42)
    height_in_cm: float = Field(..., ge=150, le=210)
    position: str
    sub_position: str
    foot: str
    # Optional, not required: a player currently at a club outside
    # LEAGUE_MAP (e.g. Messi at Inter Miami, Son at LAFC - both MLS) has no
    # reliable recent-performance data at all (see build_lookups.py and
    # search_players() above) - autofilled from players_lookup.csv's
    # recent_* columns when available and left unset otherwise, the same
    # pattern as pre_fotmob_* below. build_feature_row median-imputes a
    # missing one instead of a fabricated 0, which used to be sent here and
    # get treated as this player's genuine (terrible) recent form.
    pre_apps: float | None = Field(None, ge=0)
    pre_minutes: float | None = Field(None, ge=0)
    pre_goals_p90: float | None = Field(None, ge=0)
    pre_ga_p90: float | None = Field(None, ge=0)
    pre_mins_per_app: float | None = Field(None, ge=0)
    transfer_fee: float = Field(..., ge=0)
    value_before: float = Field(..., gt=0)
    from_domestic_competition_id: str
    to_domestic_competition_id: str
    from_total_market_value: float = Field(..., ge=0)
    to_total_market_value: float = Field(..., ge=0)
    # Pre-transfer FotMob per-90 stats (rating/attacking/possession/
    # defensive - see scripts/fetch_pretransfer_fotmob_stats.py). The model
    # doesn't train on these raw values directly - build_feature_row turns
    # them into 4 position-relative percentiles first (see
    # compute_pretransfer_fotmob_composites), the same rating/attacking/
    # defensive/possession split the historical score's post-transfer
    # components use. Optional: ~35-45% of transfers have no FotMob match
    # for the player's year before the move (an uncovered league, or a real
    # coverage gap - same ceilings as the post-transfer side), autofilled
    # from players_lookup.csv's recent_fotmob_* columns when available and
    # left unset otherwise - a bucket with no real underlying stat at all
    # gets the trained median composite rather than requiring the frontend
    # to know it.
    pre_fotmob_rating: float | None = None
    pre_fotmob_expected_goals_per_90: float | None = None
    pre_fotmob_expected_assists_per_90: float | None = None
    pre_fotmob_chances_created_p90: float | None = None
    pre_fotmob_accurate_pass: float | None = None
    pre_fotmob_won_contest: float | None = None
    pre_fotmob_total_tackle: float | None = None
    pre_fotmob_interception: float | None = None
    pre_fotmob_effective_clearance: float | None = None
    pre_fotmob_ball_recovery: float | None = None
    pre_fotmob_saves: float | None = None
    pre_fotmob__save_percentage: float | None = None
    pre_fotmob_goals_conceded: float | None = None


def pretransfer_percentile(value, stat, position):
    """
    One raw pre_fotmob_* value's percentile (0-100) within its position's
    training distribution - must mirror train_model.py's identically-named
    computation exactly, since this is what turns a live request's raw
    inputs into what the deployed model actually expects (see
    PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES/compute_pretransfer_fotmob_composites).
    None if there's no fitted table for this (stat, position) - a position
    with too few real training examples for a stable percentile curve - or
    the value itself is missing.
    """
    if value is None:
        return None
    breakpoints = metadata["pretransfer_percentile_tables"].get(stat, {}).get(position)
    if breakpoints is None:
        return None
    sign = -1 if stat in metadata["pretransfer_inverted_stats"] else 1
    return float(np.interp(sign * value, breakpoints, range(101)))


def compute_pretransfer_fotmob_composites(row):
    """
    Turn a request's raw pre_fotmob_* inputs into the 4 position-relative
    composites the model actually trains on (train_model.py's
    PRETRANSFER_FOTMOB_COMPOSITES) - average of whichever of a bucket's raw
    stats are real (via pretransfer_percentile), NaN-tolerant the same way
    compute_pretransfer_fotmob_composites in train_model.py is, imputed
    with the training median when a whole bucket has no real data.

    Returns (composites, has_real_data): the 4 composite values ready to
    drop into the feature row, and which buckets had at least one real
    (non-imputed) underlying stat - explain_prediction needs that to avoid
    a false swing, the same failure mode the raw-feature version of this
    had: comparing an imputed value against the wrong reference
    manufactures a "contribution" out of nothing (caught directly before
    the composite rework: an attacker with no defensive FotMob data was
    showing a real-looking "+3.7 Recent defensive work").
    """
    position = row["position"]
    medians = metadata["pretransfer_fotmob_composite_medians"]
    is_goalkeeper = position == "Goalkeeper"
    composites, has_real_data = {}, {}
    for composite_feat, spec in PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES.items():
        raw_feats = spec.get("goalkeeper_raw_features", spec["raw_features"]) if is_goalkeeper else spec["raw_features"]
        pcts = [p for p in (pretransfer_percentile(row.get(f), f, position) for f in raw_feats) if p is not None]
        has_real_data[composite_feat] = len(pcts) > 0
        composites[composite_feat] = sum(pcts) / len(pcts) if pcts else medians[composite_feat]
    return composites, has_real_data


RECENT_PERFORMANCE_FEATURES = ["pre_apps", "pre_minutes", "pre_goals_p90", "pre_ga_p90", "pre_mins_per_app"]

# The 3 of RECENT_PERFORMANCE_FEATURES that are structurally dependent on
# each other (minutes roughly equals apps times mins_per_app) - explained
# as one combined swap, not independently, to avoid producing a physically
# impossible synthetic row (see explain_prediction). pre_goals_p90/
# pre_ga_p90 are also related (ga_p90 = goals_p90 + assists_p90) but tested
# empirically without finding the same severity of issue, so they're left
# as independent explanation entries.
PLAYING_TIME_FEATURES = ["pre_apps", "pre_minutes", "pre_mins_per_app"]


def impute_recent_performance(row):
    """
    A player currently at a club outside LEAGUE_MAP (see build_lookups.py -
    e.g. Messi at Inter Miami, Son at LAFC, both MLS) has no reliable recent-
    performance data at all, so all 5 RECENT_PERFORMANCE_FEATURES arrive as
    None together (they're derived from one appearances rollup, not five
    independent ones - see build_lookups.py's build_players_lookup). Used to
    silently arrive as a fabricated 0 instead - a real, terrible "0 recent
    minutes played" signal for a player the model actually has no data on,
    which explain_prediction would then describe as if it meant something
    (and could show a backwards-looking "raising the score" swing purely as
    an artifact of where 0 happens to sit relative to the model's learned
    curve, not because 0 recent minutes is actually good).

    Missing features are filled with the exact same reference value
    resolve_reference (in explain_prediction) would independently pick for
    them - the position-conditional median, falling back to the flat one -
    so the leave-one-out swap for an imputed feature compares it against
    itself and reports an honest, guaranteed 0.0 contribution rather than a
    fabricated one from comparing two different baselines.

    Returns has_real_data: True if every one of the 5 fields was genuinely
    provided, False if this function had to impute all of them.
    """
    position = row["position"]
    position_reference = metadata["reference_values_by_position"].get(position, {})
    flat_reference = metadata["reference_values"]
    has_real_data = all(row.get(f) is not None for f in RECENT_PERFORMANCE_FEATURES)
    if not has_real_data:
        for f in RECENT_PERFORMANCE_FEATURES:
            position_value = position_reference.get(f)
            row[f] = position_value if position_value is not None else flat_reference[f]
    return has_real_data


def build_feature_row(req: PredictRequest) -> tuple[pd.DataFrame, dict]:
    """
    Turn a PredictRequest into the single-row DataFrame the model pipeline
    expects, computing the log/ratio/composite features it was trained on.
    Also returns which pre-transfer signals were genuinely computed from
    real data (as opposed to imputed with a reference value) - the 4
    pre-transfer FotMob composites (see compute_pretransfer_fotmob_composites)
    plus each of the 5 RECENT_PERFORMANCE_FEATURES names, all mapped to the
    one shared boolean impute_recent_performance returns (they're always
    missing together, not independently) - all consumed by
    explain_prediction the same way.
    """
    row = req.model_dump()
    has_real_recent_data = impute_recent_performance(row)
    row["log_transfer_fee"] = np.log1p(row["transfer_fee"])
    row["log_value_before"] = np.log1p(row["value_before"])
    row["log_from_club_value"] = np.log1p(row["from_total_market_value"])
    row["log_to_club_value"] = np.log1p(row["to_total_market_value"])
    row["fee_to_value_ratio"] = row["transfer_fee"] / max(row["value_before"], 1)
    row["club_quality_ratio"] = row["to_total_market_value"] / max(row["from_total_market_value"], 1)
    position_height_means = metadata["position_height_means"]
    row["height_vs_position"] = row["height_in_cm"] - position_height_means.get(
        row["position"], position_height_means["_default"]
    )

    composites, has_real_data = compute_pretransfer_fotmob_composites(row)
    row.update(composites)
    for composite_feat, flag in PRETRANSFER_FOTMOB_HAS_DATA_FLAGS.items():
        row[flag] = int(has_real_data[composite_feat])
    for feat in RECENT_PERFORMANCE_FEATURES:
        has_real_data[feat] = has_real_recent_data

    # Keep the raw pre_fotmob_* columns too, not just NUMERIC_FEATURES/
    # CATEGORICAL_FEATURES - the model pipeline only ever selects its own
    # named columns (extra ones are harmless), and explain_prediction needs
    # the raw values to describe a composite's underlying stats.
    df = pd.DataFrame([row])
    return df, has_real_data


def predict_marginalized_recent_performance(feature_row: pd.DataFrame, position: str) -> float:
    """
    The displayed success_score for a player with no real recent-
    performance data at all (see impute_recent_performance) - averages the
    model's prediction over every real same-position RECENT_PERFORMANCE_FEATURES
    combination in the training data (metadata["recent_performance_samples"],
    computed in train_model.py), instead of committing to feature_row's
    single median-point guess for those 5 features.

    A tree ensemble's response to a feature isn't linear, so E[f(X)] !=
    f(E[X]) - checked directly: for a real missing-data case, the median
    point predicted 55.6, but averaging over every real attacker's actual
    profile gave a mean of 57.0 with real spread (52.9-63.1 depending on
    which attacker's profile was used) - the median point isn't a neutral
    "no information" input, it's one specific (and here, pessimistic)
    guess. Whole real rows are used, not independently-resampled columns -
    RECENT_PERFORMANCE_FEATURES are structurally dependent on each other
    (see PLAYING_TIME_FEATURES), so resampling each column on its own would
    recreate the exact "impossible combination" bug already fixed for the
    leave-one-out explanation.

    One batched pipeline.predict() call over all samples at once (~35ms for
    2,585 rows, tested directly) rather than one call per sample - cheap
    enough to run on every request that needs it. Falls back to
    feature_row's own (median-imputed) prediction if this position somehow
    has no stored samples at all (defensive - every position has hundreds
    in practice).
    """
    samples = metadata["recent_performance_samples"].get(position)
    if not samples:
        return float(pipeline.predict(feature_row)[0])
    batch = pd.concat([feature_row] * len(samples), ignore_index=True)
    for feat in RECENT_PERFORMANCE_FEATURES:
        batch[feat] = [s[feat] for s in samples]
    return float(pipeline.predict(batch).mean())


def explain_prediction(feature_row: pd.DataFrame, base_score: float, real_data_flags: dict = {}, top_k: int = 5):
    """
    Approximate per-feature contributions by swapping one feature at a time
    to its "typical transfer" reference value (median/mode from training
    data) and seeing how much the prediction moves. A positive contribution
    means the actual value pushed the score up relative to a typical
    transfer; negative means it pulled the score down. This is a simple,
    transparent stand-in for a proper SHAP explanation.

    real_data_flags (see build_feature_row) says which pre-transfer signals
    were genuinely computed from real data rather than reference-imputed -
    the 4 pre-transfer FotMob composites, plus each of the 5
    RECENT_PERFORMANCE_FEATURES names mapped to one shared boolean (they're
    always missing together, not independently - see
    impute_recent_performance). An imputed value must be compared against
    that *same* reference, not some other one, or the swap manufactures a
    contribution out of the gap between two different baselines instead of
    a real signal (the exact failure mode that motivated moving to
    position-relative composites for FotMob in the first place - see
    train_model.py:PRETRANSFER_FOTMOB_COMPOSITES).
    """
    reference = metadata["reference_values"]
    position = feature_row["position"].iloc[0]
    context = {
        "position": position,
        "from_domestic_competition_id": feature_row["from_domestic_competition_id"].iloc[0],
    }

    # log_transfer_fee and fee_to_value_ratio are both 0 for free transfers
    # (out-of-contract moves, academy graduates), which are >50% of the
    # dataset - so their overall "typical" reference is 0, and comparing an
    # actual fee against "a typical €0m" is misleading. When the transfer
    # being explained itself has a real fee, compare it against the
    # typical *paid* transfer instead; a free transfer still compares
    # against the overall reference, which correctly reflects that being
    # free is itself common.
    is_paid_transfer = feature_row["log_transfer_fee"].iloc[0] > 0
    paid_reference = metadata["reference_values_paid"]

    # A striker's typical goal contributions, or a goalkeeper's typical
    # height, look nothing like the whole population's - compare these
    # against the same-position median instead of a flat one (see
    # reference_values_by_position in train_model.py).
    position_conditional = set(metadata["position_conditional_features"])
    position_reference = metadata["reference_values_by_position"].get(position, {})

    # A €100m fee for a player already valued at €70m isn't remarkable -
    # a flat "typical paid fee" (~€6m) makes any big-money move for an
    # already-valuable player look like a wild outlier. Compare the actual
    # fee against what's typically paid for a player valued this highly
    # instead (fee_regression: log(fee) ~ log(value), fit on paid
    # transfers - see train_model.py), using *this* transfer's own
    # value_before.
    fee_reg = metadata["fee_regression"]
    log_value_before = feature_row["log_value_before"].iloc[0]
    expected_log_fee = fee_reg["intercept"] + fee_reg["slope"] * log_value_before

    pretransfer_fotmob_composite_medians = metadata["pretransfer_fotmob_composite_medians"]

    def resolve_reference(feat):
        """The reference ("typical") value + label for one feature."""
        if feat in pretransfer_fotmob_composite_medians and not real_data_flags.get(feat, True):
            # This value is itself the bucket's median (build_feature_row
            # imputed it - no real underlying stat for this bucket) -
            # comparing it against any other reference would swap two
            # different baselines against each other and manufacture a
            # contribution out of nothing. Comparing the exact same median
            # against itself guarantees a true, honest zero.
            return pretransfer_fotmob_composite_medians[feat], "a typical transfer's"
        if feat == "log_transfer_fee" and is_paid_transfer:
            return expected_log_fee, "what's typically paid for a similarly-valued player:"
        if feat in position_conditional and feat in position_reference and pd.notna(position_reference[feat]):
            # pd.notna guards a real gap: a GK-only stat (e.g. saves) has no
            # meaningful median for outfield positions at all (virtually no
            # attacker/midfielder has FotMob save data), so
            # reference_values_by_position stores NaN there rather than a
            # fabricated number - falls through to the flat reference below,
            # which is always a real finite value (see train_model.py's
            # median-imputation for pre_fotmob_* features).
            return position_reference[feat], f"a typical {POSITION_PLURAL.get(position, position).rstrip('s')}'s"
        if is_paid_transfer and feat in paid_reference:
            return paid_reference[feat], "a typical paid transfer's"
        return reference[feat], "a typical transfer's"

    def swap_and_score(feats_to_values):
        """Predict with the given {feature: reference_value} substitutions applied all at once; returns the contribution (base_score - modified_score)."""
        modified = feature_row.copy()
        for feat, value in feats_to_values.items():
            modified[feat] = value
        modified_score = float(pipeline.predict(modified)[0])
        return round(base_score - modified_score, 1)

    is_goalkeeper = position == "Goalkeeper"
    has_data_flag_names = set(PRETRANSFER_FOTMOB_HAS_DATA_FLAGS.values())
    typical_pos_label = f"a typical {POSITION_PLURAL.get(position, position).rstrip('s')}'s"

    contributions = []

    # pre_apps/pre_minutes/pre_mins_per_app are structurally dependent
    # (minutes roughly equals apps times mins_per_app) - swapping just one
    # to its reference while the other two stay at the transfer's real
    # values can produce a physically impossible row (e.g. "0 apps, 1979
    # minutes, 0 min/app") that no real transfer ever has, which the model
    # then extrapolates unpredictably at. Checked directly: for a real
    # player with 0 recent apps/minutes/mins_per_app, swapping pre_minutes
    # alone showed a backwards +3.1 ("raising the score" for having no
    # recent playing time), while swapping all three together to one
    # consistent "typical" combination showed the correct, honest -4.2 -
    # and the three separate single-swap contributions don't even sum
    # close to that joint number, confirming they aren't independent.
    # Swapped together here for the explanation only, as one combined
    # "Recent playing time" entry with a bulleted breakdown of the real
    # numbers - the model itself is untouched, and was never shown an
    # inconsistent combination like this in training, only real ones.
    playing_time_refs = {f: resolve_reference(f)[0] for f in PLAYING_TIME_FEATURES}
    playing_time_contribution = swap_and_score(playing_time_refs)
    playing_time_direction = "raising" if playing_time_contribution >= 0 else "lowering"
    if not real_data_flags.get("pre_apps", True):
        # Not "raising/lowering the score by 0.0 pts" here, unlike every
        # other no-data message below - that framing would be misleading
        # now: the *displayed* score already comes from
        # predict_marginalized_recent_performance (see predict()), which
        # measurably shifts the score from what a single median guess would
        # give (checked directly: 55.6 -> 57.0 for a real case), just not in
        # a way this leave-one-out swap can attribute to a specific point
        # value the way it does for every feature that has one.
        playing_time_stats = None
        playing_time_detail = (
            "No recent performance data available for this player (outside the tracked leagues) - "
            f"the score above already averages the prediction across many real {POSITION_PLURAL.get(position, position).lower()}' "
            "actual recent-performance profiles, rather than guessing a single typical one"
        )
    else:
        playing_time_stats = [
            f"{FEATURE_LABELS[f]}: {format_feature_value(f, feature_row[f].iloc[0], context)}"
            for f in PLAYING_TIME_FEATURES
        ]
        playing_time_detail = (
            f"vs. {typical_pos_label} recent playing time, {playing_time_direction} the score by {abs(playing_time_contribution)} pts"
        )
    contributions.append({
        "feature": "pre_playing_time",
        "label": "Recent playing time",
        "contribution": playing_time_contribution,
        "actual_value": None,
        "typical_value": None,
        "stats": playing_time_stats,
        "detail": playing_time_detail,
    })

    for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if feat in has_data_flag_names:
            continue  # a data-quality flag, not a football signal - the composite feature's own detail (below) already says when it has no real data
        if feat in PLAYING_TIME_FEATURES:
            continue  # already handled together above - see playing_time_contribution
        reference_value, typical_label = resolve_reference(feat)
        actual_value = feature_row[feat].iloc[0]
        contribution = swap_and_score({feat: reference_value})
        direction = "raising" if contribution >= 0 else "lowering"

        if feat in PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES:
            # One real, independent model feature now (unlike the old raw-
            # stat version, which had to swap several correlated features
            # together to get an honest combined contribution - see
            # train_model.py:PRETRANSFER_FOTMOB_COMPOSITES) - the swap above
            # already gives the right number on its own. What's still worth
            # building here: a labeled bulleted breakdown of the underlying
            # raw stats as *descriptive* context, since "72nd percentile
            # among forwards" alone doesn't say what that means concretely
            # (same idea as describe_fotmob_component's breakdown for the
            # historical score).
            spec = PRETRANSFER_FOTMOB_COMPOSITE_RAW_FEATURES[feat]
            raw_feats = spec.get("goalkeeper_raw_features", spec["raw_features"]) if is_goalkeeper else spec["raw_features"]
            label = spec.get("goalkeeper_label", FEATURE_LABELS[feat]) if is_goalkeeper else FEATURE_LABELS[feat]
            has_real_data = real_data_flags.get(feat, True)
            typical_pos_label = "a typical goalkeeper's" if is_goalkeeper else f"a typical {POSITION_PLURAL.get(position, position).rstrip('s')}'s"
            if has_real_data and len(raw_feats) > 1:
                # Multiple sub-stats: a bulleted list, same as
                # describe_fotmob_component's own multi-stat rows. The
                # composite itself can have real data even when one specific
                # raw sub-stat doesn't - e.g. an attacker's clearance count -
                # same reasoning as describe_fotmob_component's own parts()
                # helper, which this mirrors: only the sub-stats actually
                # present get a bullet, rather than formatting a None
                # straight into the string (previously a 400 - predict()'s
                # try/except turns the resulting TypeError into an HTTP
                # error - on any transfer where this happened, caught
                # testing Compare by hand).
                stats = [
                    f"{PRETRANSFER_FOTMOB_STAT_LABELS[rf]}: {format_feature_value(rf, feature_row[rf].iloc[0], context)}"
                    for rf in raw_feats
                    if pd.notna(feature_row[rf].iloc[0])
                ]
                detail = f"{ordinal(actual_value)} percentile vs. {typical_pos_label} recent numbers, {direction} the score by {abs(contribution)} pts"
            elif has_real_data:
                # A single sub-stat (rating) has nothing to bullet - fold it
                # into one plain sentence instead, same as
                # describe_fotmob_component does for its own rating row.
                stats = None
                raw_value = format_feature_value(raw_feats[0], feature_row[raw_feats[0]].iloc[0], context)
                detail = f"{raw_value} ({ordinal(actual_value)} percentile vs. {typical_pos_label} recent numbers), {direction} the score by {abs(contribution)} pts"
            else:
                stats = None
                detail = f"No recent FotMob data available for this player - using a league-typical value, {direction} the score by {abs(contribution)} pts"
            contributions.append({
                "feature": feat,
                "label": label,
                "contribution": contribution,
                "actual_value": None,
                "typical_value": None,
                "stats": stats,
                "detail": detail,
            })
            continue

        if feat in RECENT_PERFORMANCE_FEATURES and not real_data_flags.get(feat, True):
            # actual_value already equals reference_value here (build_feature_row
            # imputed it with this exact reference - see impute_recent_performance),
            # so this swap's own contribution is guaranteed 0.0 - the plain
            # swing-only detail below would read as a technically-true but useless
            # "0.31 vs. a typical midfielder's 0.31", instead of saying plainly
            # that this player has no real recent-performance data at all (a club
            # outside LEAGUE_MAP - see build_lookups.py). Not "raising/lowering the
            # score by 0.0 pts" either, unlike a normal 0-contribution feature -
            # that would misleadingly imply this feature played no part in the
            # displayed score, when the *displayed* score actually comes from
            # predict_marginalized_recent_performance (see predict()), which
            # averages over real values of this feature rather than guessing one.
            contributions.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "contribution": contribution,
                "actual_value": None,
                "typical_value": None,
                "stats": None,
                "detail": (
                    "No recent performance data available for this player (outside the "
                    "tracked leagues) - the score above already averages the prediction across "
                    f"many real {POSITION_PLURAL.get(position, position).lower()}' actual values"
                ),
            })
            continue

        actual_display = format_feature_value(feat, actual_value, context)
        typical_display = format_feature_value(feat, reference_value, context)

        # Per-feature context note - the same idea as the historical score's
        # bespoke component descriptions (describe_components), so a
        # prediction explanation says *why* a swing happens, not just that
        # it does. Most features need nothing extra; a few (league,
        # fee-to-value, club-quality, height) are opaque or misleading
        # without it - see league_context_note for the motivating case.
        if feat == "height_vs_position":
            vs_clause = "vs. the position average"
        elif feat in ("to_domestic_competition_id", "from_domestic_competition_id"):
            vs_clause = f"vs. {typical_label} {typical_display}{league_context_note(feat, actual_value, reference_value)}"
        elif feat in ("log_to_club_value", "log_from_club_value"):
            club_side = "to" if feat == "log_to_club_value" else "from"
            note = club_value_rating_note(club_side, actual_value, reference_value, typical_display)
            vs_clause = f"vs. {typical_label} {typical_display}{note}"
        elif feat == "fee_to_value_ratio":
            vs_clause = (
                f"vs. {typical_label} {typical_display}: paying up to ~1.3x market value counts as a "
                f"normal premium in the historical scoring; only fees further above that actually count against a transfer"
            )
        elif feat == "club_quality_ratio":
            vs_clause = f"vs. {typical_label} {typical_display} (destination squad value ÷ origin squad value)"
        elif feat in ("position", "sub_position", "foot"):
            vs_clause = f"vs. {typical_label} {typical_display}{category_baseline_note(feat, actual_value, reference_value)}"
        else:
            vs_clause = f"vs. {typical_label} {typical_display}"

        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "contribution": contribution,
            "actual_value": actual_display,
            "typical_value": typical_display,
            "stats": None,
            "detail": f"{actual_display} {vs_clause}, {direction} the score by {abs(contribution)} pts",
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_k]


def find_comparables(feature_row: pd.DataFrame, k: int = 5):
    """
    Look up the k most similar historical transfers to `feature_row` using
    the nearest-neighbors index built in train_model.py (Euclidean distance
    over the scaled numeric features). Used both to show "most similar
    historical transfers" and, via their success_score spread, as the
    predicted score_range in /api/predict.
    """
    x = feature_row[comparables["features"]]
    x_scaled = comparables["scaler"].transform(x)
    dist, idx = comparables["index"].kneighbors(x_scaled, n_neighbors=k)
    rows = comparables["meta"].iloc[idx[0]]
    out = []
    for _, r in rows.iterrows():
        out.append({
            "name": r["name"],
            "transfer_date": str(r["transfer_date"])[:10],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "success_score": float(r["success_score"]),
        })
    return out


def build_transfer_card(r):
    """
    Build the JSON shape shared by /api/examples and /api/transfers/detail
    for one row of transfers_processed.csv: identity/route, the score, and
    the full describe_components() breakdown.
    """
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "transfer_date": str(r["transfer_date"])[:10],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "success_score": float(r["success_score"]),
        "pre_ga_p90": round(float(r["pre_ga_p90"]), 2),
        "post_ga_p90": round(float(r["post_ga_p90"]), 2),
        "tenure_days": int(r["tenure_days"]),
        "still_at_club": bool(r["still_at_club"]),
        "breakdown": describe_components(r),
    }


def describe_loan_components(r):
    """
    Build the "why this score" breakdown for one row of loans_processed.csv:
    2 components always, plus one of "G/A per 90"/"Attacking" - see
    describe_components(), the same fold applies here - plus up to 3 more
    FotMob-derived rows (possession/defending/rating, each independently
    shown only when its own has_*_data flag is true - so 3 to 7 rows
    total). Fixed display order: Value change, G/A per 90, G/A change,
    Attacking, Possession, Defending, FotMob rating, Playing time - same
    order as describe_components() minus Transfer fee and Resale profit,
    neither of which apply to a loan (see below). Unlike
    describe_components(), there's no "resale profit" row - a loan doesn't
    end in a sale of its own - and no "value for money"/"Transfer fee" row
    - most loans carry no real fee, see data/loan_score_weights.json.
    """
    position_plural = POSITION_PLURAL.get(r["position"], r["position"])
    to_league = league_display_name(r["to_domestic_competition_id"])
    # See describe_components() for why these three drop out, and
    # "Defending" relabels to "Goalkeeping", for a goalkeeper.
    is_goalkeeper = r["position"] == "Goalkeeper"
    fotmob_components = ("possession", "defensive", "rating") if is_goalkeeper else ("attacking", "possession", "defensive", "rating")
    return [
        {
            "label": "Value change",
            "value": round(float(r["value_growth_pct"]), 1),
            "description": (
                f"{eur_m(r['value_before'])} → peaked at {eur_m(r['value_peak'])} (now {eur_m(r['value_after'])}) during the loan"
                if r["value_peak"] > r["value_after"] * 1.05
                else f"{eur_m(r['value_before'])} → {eur_m(r['value_after'])} market value during the loan"
            ),
        },
    ] + ([] if is_goalkeeper or bool(r["has_attacking_data"]) else [
        {
            "label": "G/A per 90",
            "value": round(float(r["perf_level_pct"]), 1),
            "description": (
                f"{r['post_ga_p90']:.2f} goal contributions/90 while on loan at {r['to_club_name']} "
                f"({r['post_ga_p90_vs_league']:.1f}x the {to_league} average for {position_plural}), "
                f"ranked vs. other loan spells"
            ),
        },
    ]) + ([] if is_goalkeeper else [
        {
            "label": "G/A change",
            "value": round(float(r["perf_delta_pct"]), 1),
            "description": (
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x on loan, beating the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high (some pullback from a peak is normal)"
                if r["post_ga_p90_vs_league"] >= r["expected_post_ga_p90_vs_league"] else
                f"Started at {r['pre_ga_p90_vs_league']:.1f}x league average, now at "
                f"{r['post_ga_p90_vs_league']:.1f}x on loan, below the ~{r['expected_post_ga_p90_vs_league']:.1f}x "
                f"expected for a player starting that high"
            ),
        },
    ]) + [
        {
            "label": "Goalkeeping" if (component == "defensive" and is_goalkeeper) else FOTMOB_COMPONENT_LABELS[component],
            "value": round(float(r[f"{component}_pct"]), 1),
            **describe_fotmob_component(component, r, position_plural, is_loan=True),
        }
        for component in fotmob_components
        if bool(r[f"has_{component}_data"])
    ] + [
        {
            "label": "Playing time",
            "value": round(float(r["playing_time_pct"]), 1),
            "description": (
                f"{int(r['post_apps'])} of {int(r['team_games_in_tenure'])} games "
                f"{r['to_club_name']} played during the loan "
                f"({r['pct_team_games_played'] * 100:.0f}% - usually the central question a loan gets "
                f"judged on, blended with raw appearance count)"
            ),
        },
    ]


def build_loan_card(r):
    """
    Build /api/loans/detail's full-card JSON for one row of
    loans_processed.csv: identity/route, the score, the full
    describe_loan_components() breakdown, and whether this loan later
    converted to a permanent transfer (see find_loan_conversion) - the
    loans page's click-to-view modal.
    """
    conversion = find_loan_conversion(r)
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "transfer_date": str(r["transfer_date"])[:10],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "loan_success_score": float(r["loan_success_score"]),
        "pre_ga_p90": round(float(r["pre_ga_p90"]), 2),
        "post_ga_p90": round(float(r["post_ga_p90"]), 2),
        "tenure_days": int(r["tenure_days"]),
        "still_on_loan": bool(r["still_on_loan"]),
        "converted_to_permanent": conversion is not None,
        "conversion_transfer_date": conversion["transfer_date"] if conversion else None,
        "conversion_success_score": conversion["success_score"] if conversion else None,
        "breakdown": describe_loan_components(r),
    }


@app.get("/api/health")
def health():
    """Liveness check plus a dump of the deployed model's training metadata (feature lists, test metrics)."""
    return {"status": "ok", "model_metadata": metadata}


@app.get("/api/examples")
def examples():
    """
    Return the curated homepage cards (EXAMPLE_TRANSFER_KEYS) as full
    transfer cards. EXAMPLE_TRANSFER_KEYS is written in whichever short
    form reads clearly to a maintainer ("Man City") - resolved through
    CLUB_NAME_ALIASES before matching, since transfers_df's own
    to_club_name is already canonicalized to (usually longer) spellings
    like "Manchester City" - without this, the lookup would silently drop
    an entry every time the canonical-name rule picks a different variant.
    """
    out = []
    for name, to_club in EXAMPLE_TRANSFER_KEYS:
        canonical_club = CLUB_NAME_ALIASES.get(to_club, to_club)
        match = transfers_df[
            (transfers_df["name"] == name) & (transfers_df["to_club_name"] == canonical_club)
        ]
        if match.empty:
            continue
        out.append(build_transfer_card(match.iloc[-1]))
    return out


@app.get("/api/transfers/detail")
def transfer_detail(player_id: int, transfer_date: str):
    """Look up one specific historical transfer by (player_id, transfer_date) and return its full card - used by the browse page's click-to-view modal."""
    match = transfers_df[
        (transfers_df["player_id"] == player_id) & (transfers_df["transfer_date"] == transfer_date)
    ]
    if match.empty:
        raise HTTPException(status_code=404, detail="transfer not found")
    return build_transfer_card(match.iloc[0])


RECENT_PERFORMANCE_COLUMNS = [
    "recent_apps", "recent_minutes", "recent_goals", "recent_assists",
    "recent_ga_p90", "recent_goals_p90", "recent_mins_per_app",
]


def serialize_player_rows(rows):
    """
    players_lookup.csv rows -> JSON-safe records for the Predict/Compare
    forms: recent_fotmob_* columns and RECENT_PERFORMANCE_COLUMNS are
    genuinely numeric (unlike the other columns here, which are safely
    blanket-filled with "" for a missing string field) and feed straight
    into PredictRequest's Optional[float] pre_fotmob_*/pre_apps/pre_minutes/
    etc. fields - filling a missing one with "" would send the frontend a
    string that's neither a valid float nor JSON null, breaking the
    request. Left as real NaN, then swapped to None (valid JSON null)
    below instead - a fabricated "" or 0 would misrepresent "no data" as a
    real value. RECENT_PERFORMANCE_COLUMNS is genuinely NaN for a player at
    a club outside LEAGUE_MAP (e.g. Messi at Inter Miami, Son at LAFC -
    both MLS) - see build_lookups.py - rather than always having a real
    number the way it used to before that was fixed, which is exactly why
    this needs the same numeric-safe handling recent_fotmob_* already had.

    Shared by /api/players/search (many rows at once) and
    /api/players/{player_id} (exactly one) below, so both return byte-for-
    byte the same shape - Compare's shareable-link restore (GET one player
    by id) has to reconstruct exactly what the search-driven selection flow
    would have given buildPayload().
    """
    numeric_cols = [c for c in rows.columns if c.startswith("recent_fotmob")] + RECENT_PERFORMANCE_COLUMNS
    records = rows.drop(columns=numeric_cols).fillna("").to_dict(orient="records")
    numeric_records = rows[numeric_cols].astype(object).where(rows[numeric_cols].notna(), None).to_dict(orient="records")
    for record, numeric_record in zip(records, numeric_records):
        record.update(numeric_record)
    return records


@app.get("/api/players/search")
def search_players(q: str, limit: int = 10):
    """Accent-insensitive substring search over players_lookup.csv, for the prediction form's player autocomplete - see serialize_player_rows() for the numeric-safe record shape."""
    if len(q) < 2:
        return []
    limit = max(1, min(limit, 50))
    mask = players_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = players_df[mask].head(limit).drop(columns=["_name_fold"])
    return serialize_player_rows(rows)


@app.get("/api/players/career-search")
def search_players_for_career(q: str, limit: int = 10):
    """
    Accent-insensitive player-name search over every player with at least
    one scored permanent transfer or loan (the union of transfers_df and
    loans_df), for the Player Timelines page's search box. Deliberately
    broader than /api/players/search: that one is filtered to
    players_lookup.csv, which only covers players above a market-value
    threshold (see build_lookups.py) - fine for the Predict form's
    autofill, but it would silently hide most retired or lower-value
    players from a page whose entire point is showing real career history
    (~59% of players with a scored transfer/loan aren't in that lookup at
    all - checked directly).
    """
    if len(q) < 2:
        return []
    limit = max(1, min(limit, 50))
    mask = PLAYER_CAREER_SEARCH_DF["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    matches = PLAYER_CAREER_SEARCH_DF[mask].head(limit)
    return matches[["player_id", "name"]].to_dict(orient="records")


# Registered after /api/players/search and /api/players/career-search
# above (and before /api/players/{player_id}/career below, though that one
# can't actually collide - see next comment) deliberately: FastAPI/
# Starlette matches routes by trying each registered path pattern in
# order, and a bare {player_id} segment (no :int converter in the path
# itself - the `player_id: int` type hint only validates *after* a route
# already matched) matches any single path segment as a string, including
# literally "search" or "career-search". Registered first, this route
# would have swallowed every request meant for those two search endpoints
# and 422'd on trying to parse "search"/"career-search" as an int -
# breaking the Predict/Compare player autocomplete and the Player
# Timelines search entirely. /api/players/{player_id}/career is safe
# either way - it has an extra /career segment a bare {player_id} pattern
# never matches - but kept below this one anyway for readability, longest
# and most specific path last.
@app.get("/api/players/{player_id}")
def get_player(player_id: int):
    """
    One players_lookup.csv row by id, in exactly the shape
    /api/players/search already returns (see serialize_player_rows()) -
    lets Compare's shareable-link restore reconstruct a scenario's player
    selection from a URL alone (?player_a=<id>), the same way the existing
    /api/clubs/{club_id} already does for a destination club.
    """
    rows = players_df[players_df["player_id"] == player_id].drop(columns=["_name_fold"])
    if rows.empty:
        raise HTTPException(status_code=404, detail="player not found")
    return serialize_player_rows(rows)[0]


@app.get("/api/players/{player_id}/career")
def player_career(player_id: int):
    """
    Every scored permanent transfer and loan spell for one player, in
    chronological order - the data behind the Player Timelines page's
    career chart. Combines transfers_df and loans_df (tagging each stop
    with its own "type") since many real careers include both, e.g. a
    young player loaned out several times before a permanent breakthrough
    move - score is normalized to one shared "score" field either way
    (success_score / loan_success_score) so the frontend chart doesn't
    need to know which table a given stop came from, only how to route a
    click on it to the right detail endpoint.
    """
    permanent = transfers_df[transfers_df["player_id"] == player_id]
    loan_rows = loans_df[loans_df["player_id"] == player_id]
    if permanent.empty and loan_rows.empty:
        raise HTTPException(status_code=404, detail="no scored transfers or loans found for this player")

    stops = []
    for _, r in permanent.iterrows():
        stops.append({
            "type": "permanent",
            "transfer_date": str(r["transfer_date"])[:10],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "score": float(r["success_score"]),
        })
    for _, r in loan_rows.iterrows():
        stops.append({
            "type": "loan",
            "transfer_date": str(r["transfer_date"])[:10],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "score": float(r["loan_success_score"]),
        })
    stops.sort(key=lambda s: s["transfer_date"])

    name = (permanent["name"] if not permanent.empty else loan_rows["name"]).iloc[0]
    player_row = players_df[players_df["player_id"] == player_id]
    # Most players with real scored history are missing from players_lookup.csv
    # (see search_players_for_career) - position/current_club are a nice-to-have
    # header, not required, so this degrades to just the name and timeline.
    position = str(player_row["position"].iloc[0]) if not player_row.empty and pd.notna(player_row["position"].iloc[0]) else None
    current_club = str(player_row["current_club_name"].iloc[0]) if not player_row.empty and pd.notna(player_row["current_club_name"].iloc[0]) else None

    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "current_club": current_club,
        "stops": stops,
    }


@app.get("/api/clubs/search")
def search_clubs(q: str, limit: int = 10):
    """Accent-insensitive substring search over clubs_lookup.csv, for the destination-club autocomplete."""
    if len(q) < 2:
        return []
    limit = max(1, min(limit, 50))
    mask = clubs_df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)
    rows = clubs_df[mask].head(limit).drop(columns=["_name_fold"])
    return rows.fillna("").to_dict(orient="records")


# Each sort field's own (minimum-sample column, threshold) - an average
# (avg_incoming_score, avg_resale_profit_pct) is misleading from a handful
# of transfers, so ranking by one of those two filters out clubs below the
# threshold entirely rather than showing a noisy number. A raw total
# (total_spent, transfers_in/out) has no such problem - one real transfer is
# one real data point, not a noisy average - so those get no filter.
CLUB_SORT_FIELDS = {
    "avg_incoming_score": ("transfers_in", MIN_CLUB_TRANSFERS),
    "avg_resale_profit_pct": ("resales_count", MIN_CLUB_RESALES),
    "avg_departure_score": ("transfers_out", MIN_CLUB_TRANSFERS),
    "total_spent": (None, 0),
    "total_resale_profit": (None, 0),
    "transfers_in": (None, 0),
    "transfers_out": (None, 0),
}


@app.get("/api/clubs/leaderboard")
def clubs_leaderboard(
    league: str | None = None,
    q: str | None = None,
    sort: str = "avg_incoming_score",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
):
    """
    Paginated, filterable, sortable ranking of every club that's bought or
    sold at least one scored permanent transfer (see
    build_club_report_cards) - a club's "recruitment report card". Each row
    already carries its own best/worst signing and best/worst resale, so the
    frontend's click-to-view-card modal needs no second request.
    """
    df = club_report_cards_df
    if league:
        df = df[df["league_id"] == league]
    if q:
        df = df[df["_name_fold"].str.contains(fold_accents(q), na=False, regex=False)]

    sort_field = sort if sort in CLUB_SORT_FIELDS else "avg_incoming_score"
    min_field, min_count = CLUB_SORT_FIELDS[sort_field]
    if min_field:
        df = df[df[min_field] >= min_count]
    df = df.dropna(subset=[sort_field])
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = []
    for _, r in page.iterrows():
        results.append({
            "club_name": r["club_name"],
            "league": league_display_name(r["league_id"]),
            "transfers_in": int(r["transfers_in"]),
            "avg_incoming_score": None if pd.isna(r["avg_incoming_score"]) else round(float(r["avg_incoming_score"]), 1),
            "total_spent": float(r["total_spent"]),
            "best_signing": r["best_signing"],
            "worst_signing": r["worst_signing"],
            "resales_count": int(r["resales_count"]),
            "avg_resale_profit_pct": None if pd.isna(r["avg_resale_profit_pct"]) else round(float(r["avg_resale_profit_pct"]), 1),
            "total_resale_profit": float(r["total_resale_profit"]),
            "best_flip": r["best_flip"],
            "worst_flip": r["worst_flip"],
            "transfers_out": int(r["transfers_out"]),
            "avg_departure_score": None if pd.isna(r["avg_departure_score"]) else round(float(r["avg_departure_score"]), 1),
            "best_departure": r["best_departure"],
            "worst_departure": r["worst_departure"],
        })
    return {"total": total, "limit": limit, "offset": offset, "results": results}


@app.get("/api/clubs/leaderboard/filters")
def clubs_leaderboard_filters():
    """List the distinct primary leagues present in club_report_cards_df, for the clubs page's league filter dropdown."""
    league_ids = club_report_cards_df["league_id"].dropna().unique().tolist()
    leagues = sorted(
        ({"id": lid, "name": LEAGUE_NAMES.get(lid, lid)} for lid in league_ids),
        key=lambda x: x["name"],
    )
    return {"leagues": leagues}


LEAGUE_TREND_SORT_FIELDS = {"transfers", "avg_score", "avg_fee", "fee_growth_pct", "score_change"}


@app.get("/api/leagues/trends")
def leagues_trends(sort: str = "transfers", order: str = "desc"):
    """
    Every league with at least MIN_LEAGUE_TRANSFERS transfers (see
    build_league_trends), sorted for the League Trends page - a couple
    dozen leagues at most, so unlike the other leaderboards this returns
    every row rather than paginating. Sorting by fee_growth_pct or
    score_change drops leagues with no trend data (not enough real
    history for the early/recent era comparison) rather than showing them
    at an arbitrary position.
    """
    df = league_trends_df
    sort_field = sort if sort in LEAGUE_TREND_SORT_FIELDS else "transfers"
    df = df.dropna(subset=[sort_field]).sort_values(sort_field, ascending=(order == "asc"))

    results = []
    for _, r in df.iterrows():
        results.append({
            "league_id": r["league_id"],
            "league": league_display_name(r["league_id"]),
            "transfers": int(r["transfers"]),
            "avg_score": round(float(r["avg_score"]), 1),
            "avg_fee": float(r["avg_fee"]),
            "early_years": r["early_years"],
            "recent_years": r["recent_years"],
            "early_avg_fee": None if pd.isna(r["early_avg_fee"]) else float(r["early_avg_fee"]),
            "recent_avg_fee": None if pd.isna(r["recent_avg_fee"]) else float(r["recent_avg_fee"]),
            "fee_growth_pct": None if pd.isna(r["fee_growth_pct"]) else round(float(r["fee_growth_pct"]), 1),
            "early_avg_score": None if pd.isna(r["early_avg_score"]) else round(float(r["early_avg_score"]), 1),
            "recent_avg_score": None if pd.isna(r["recent_avg_score"]) else round(float(r["recent_avg_score"]), 1),
            "score_change": None if pd.isna(r["score_change"]) else round(float(r["score_change"]), 1),
            "by_year": r["by_year"],
        })
    return {"total": len(results), "results": results}


@app.get("/api/clubs/{club_id}")
def get_club(club_id: int):
    """Look up one club by id - used to fetch a selected player's *current* club details (for the "origin club" side of a prediction)."""
    row = clubs_df[clubs_df["club_id"] == club_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="club not found")
    return row.iloc[0].drop("_name_fold").fillna("").to_dict()


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Score a hypothetical transfer: run the model, clip to [0, 100], and
    attach a likely score_range (min/max among the nearest comparable
    historical transfers - a single point estimate would overstate how
    confident a R^2~0.10 model can be), the top-5 feature explanation, and
    the comparables themselves.

    When there's no real recent-performance data at all (see
    impute_recent_performance), the *displayed* score comes from
    predict_marginalized_recent_performance instead of a plain
    pipeline.predict() on feature_row's single median-point guess for those
    5 features - see that function for why. explain_prediction is
    deliberately still built from median_point_score, not the marginalized
    one: every one of its per-feature swaps compares against feature_row's
    own prediction, so feeding it a different base_score than what
    feature_row itself predicts would shift every single feature's
    contribution by the exact same constant amount (base_score minus
    feature_row's real prediction) - caught directly while verifying this,
    every contribution in a missing-data explanation was inflated by the
    same ~+1.5, not just the recent-performance entries' own. The
    breakdown explains the median-point prediction (self-consistent, as
    always); only the headline number is upgraded to the marginalized one -
    the two already don't sum to exactly the same thing for any nonlinear-
    model prediction here, so this doesn't introduce a new kind of gap,
    just widens the existing one slightly for this one case.
    """
    try:
        feature_row, real_data_flags = build_feature_row(req)
        position = feature_row["position"].iloc[0]
        median_point_score = float(pipeline.predict(feature_row)[0])
        if real_data_flags.get("pre_apps", True):
            raw_score = median_point_score
        else:
            raw_score = predict_marginalized_recent_performance(feature_row, position)
        score = max(0.0, min(100.0, raw_score))
        comps = find_comparables(feature_row)
        explanation = explain_prediction(feature_row, median_point_score, real_data_flags)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    comp_scores = [c["success_score"] for c in comps]
    score_range = [round(min(comp_scores), 1), round(max(comp_scores), 1)] if comp_scores else [score, score]
    return {
        "success_score": round(score, 1),
        "score_range": score_range,
        "comparable_transfers": comps,
        "explanation": explanation,
        "model_test_mae": metadata["test_mae"],
        "model_test_r2": metadata["test_r2"],
    }


class CompareScenario(BaseModel):
    """One hypothetical transfer plus its display label, for CompareRequest.scenarios below."""
    request: PredictRequest
    label: str = "Option"


class CompareRequest(BaseModel):
    """2-4 hypothetical transfers to score side by side, for the compare page. No single "delta" field here (unlike the old two-scenario-only shape) - it doesn't generalize past a pair, so the frontend ranks `results` itself instead."""
    scenarios: list[CompareScenario] = Field(min_length=2, max_length=4)


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Score every scenario via predict() and return them together, for the compare page."""
    return {"results": [{**predict(s.request), "label": s.label} for s in req.scenarios]}


TRANSFER_SORT_FIELDS = {
    "success_score", "transfer_date", "age_at_transfer", "transfer_fee", "tenure_days", "to_club_name",
}


@app.get("/api/filters")
def get_filters():
    """List the distinct positions and destination leagues present in transfers_processed.csv, for the browse page's filter dropdowns."""
    positions = sorted(transfers_df["position"].dropna().unique().tolist())
    league_ids = transfers_df["to_domestic_competition_id"].dropna().unique().tolist()
    leagues = sorted(
        ({"id": lid, "name": LEAGUE_NAMES.get(lid, lid)} for lid in league_ids),
        key=lambda x: x["name"],
    )
    return {"positions": positions, "leagues": leagues}


def filter_transfers(position=None, league=None, q=None, min_fee=None, max_fee=None, min_age=None, max_age=None):
    """
    Shared position/league/q/fee-range/age-range filtering for
    /api/transfers and /api/transfers/export, so the exported CSV always
    matches exactly what the table's current filters show rather than a
    second, driftable reimplementation of the same filters. A fee-range
    bound naturally excludes a free/undisclosed-fee transfer too (NaN
    compares False against either bound) - it genuinely can't be judged
    as inside or outside a fee range with no known fee.
    """
    df = transfers_df
    if position:
        df = df[df["position"] == position]
    if league:
        df = df[df["to_domestic_competition_id"] == league]
    if q:
        q_fold = fold_accents(q)
        mask = (
            df["_name_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_to_club_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_from_club_fold"].str.contains(q_fold, na=False, regex=False)
        )
        df = df[mask]
    if min_fee is not None:
        df = df[df["transfer_fee"] >= min_fee]
    if max_fee is not None:
        df = df[df["transfer_fee"] <= max_fee]
    if min_age is not None:
        df = df[df["age_at_transfer"] >= min_age]
    if max_age is not None:
        df = df[df["age_at_transfer"] <= max_age]
    return df


def transfer_row_dict(r):
    """One transfers_processed.csv row as a plain dict - shared by /api/transfers' JSON list and /api/transfers/export's CSV (via rows_to_csv), so both always describe a transfer identically."""
    fee = r["transfer_fee"]
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "position": r["position"],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "to_league": league_display_name(r["to_domestic_competition_id"]),
        "transfer_date": str(r["transfer_date"])[:10],
        "age_at_transfer": round(float(r["age_at_transfer"]), 1),
        "transfer_fee": None if pd.isna(fee) else float(fee),
        "tenure_days": int(r["tenure_days"]),
        "still_at_club": bool(r["still_at_club"]),
        "success_score": float(r["success_score"]),
    }


def rows_to_csv(rows):
    """A list of flat dicts (see transfer_row_dict/loan_row_dict) as a CSV text blob for an export endpoint's response body - one header row from the first result's keys (every row already shares the same fields in the same order), empty string for zero matching rows rather than a headerless file."""
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


@app.get("/api/transfers")
def list_transfers(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "success_score",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
    min_fee: float | None = None,
    max_fee: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
):
    """Paginated, filterable, sortable listing of every scored transfer, for the browse page's table."""
    df = filter_transfers(position, league, q, min_fee, max_fee, min_age, max_age)

    sort_field = sort if sort in TRANSFER_SORT_FIELDS else "success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = [transfer_row_dict(r) for _, r in page.iterrows()]
    return {"total": total, "limit": limit, "offset": offset, "results": results}


@app.get("/api/transfers/export")
def export_transfers(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "success_score",
    order: str = "desc",
    min_fee: float | None = None,
    max_fee: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
):
    """Every transfer matching the current filters (no /api/transfers-style pagination cap) as a downloadable CSV - "export what you're looking at" for Browse, reusing filter_transfers() so the file can never silently diverge from what the table shows."""
    df = filter_transfers(position, league, q, min_fee, max_fee, min_age, max_age)
    sort_field = sort if sort in TRANSFER_SORT_FIELDS else "success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))
    csv_text = rows_to_csv([transfer_row_dict(r) for _, r in df.iterrows()])
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transfers.csv"},
    )


SURPRISE_SORT_FIELDS = {"surprise_delta", "abs_surprise_delta", "success_score", "predicted_score", "transfer_date", "age_at_transfer"}


@app.get("/api/surprises")
def list_surprises(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "surprise_delta",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
):
    """
    Paginated, filterable, sortable listing of every transfer with a held-
    out model prediction (predicted_score/surprise_delta - see
    scripts/compute_prediction_surprises.py), ranked by surprise_delta
    (success_score minus predicted_score) by default: order=desc surfaces
    the biggest overachievers (a model that never saw this transfer's
    outcome predicted a flop from pre-transfer data alone, the player
    thrived anyway); order=asc surfaces the biggest busts. sort=
    abs_surprise_delta&order=asc instead surfaces the model's most accurate
    calls - the transfers where the held-out prediction landed closest to
    what actually happened.
    """
    df = transfers_df[transfers_df["predicted_score"].notna()]
    if position:
        df = df[df["position"] == position]
    if league:
        df = df[df["to_domestic_competition_id"] == league]
    if q:
        q_fold = fold_accents(q)
        mask = (
            df["_name_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_to_club_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_from_club_fold"].str.contains(q_fold, na=False, regex=False)
        )
        df = df[mask]

    sort_field = sort if sort in SURPRISE_SORT_FIELDS else "surprise_delta"
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = []
    for _, r in page.iterrows():
        results.append({
            "player_id": int(r["player_id"]),
            "name": r["name"],
            "position": r["position"],
            "from_club": r["from_club_name"],
            "to_club": r["to_club_name"],
            "to_league": league_display_name(r["to_domestic_competition_id"]),
            "transfer_date": str(r["transfer_date"])[:10],
            "age_at_transfer": round(float(r["age_at_transfer"]), 1),
            "success_score": float(r["success_score"]),
            "predicted_score": float(r["predicted_score"]),
            "surprise_delta": float(r["surprise_delta"]),
        })
    return {"total": total, "limit": limit, "offset": offset, "results": results}


LOAN_SORT_FIELDS = {"loan_success_score", "transfer_date", "age_at_transfer", "tenure_days", "to_club_name"}


@app.get("/api/loans/filters")
def get_loan_filters():
    """List the distinct positions and loan-destination leagues present in loans_processed.csv, for the loans page's filter dropdowns."""
    positions = sorted(loans_df["position"].dropna().unique().tolist())
    league_ids = loans_df["to_domestic_competition_id"].dropna().unique().tolist()
    leagues = sorted(
        ({"id": lid, "name": LEAGUE_NAMES.get(lid, lid)} for lid in league_ids),
        key=lambda x: x["name"],
    )
    return {"positions": positions, "leagues": leagues}


def find_loan_conversion(loan_row):
    """
    Whether this loan later led to a permanent transfer - the same player
    bought permanently by the same loan club, from the same original
    owner, at some point after the loan started. Requires both club names
    to match exactly (not just the player), since a real loan-to-buy is
    specifically "the club that had them on loan later bought them" -
    a player who was loaned to club Y and later permanently joined a
    third club Z is a different, unrelated transfer, not a conversion of
    this loan. Returns {"transfer_date", "success_score"} for the
    earliest such transfer, or None.

    A loan that converts with no separate recorded event (see Known
    limitations in the README - Transfermarkt sometimes shows a
    conversion as one continuous loan record, never a distinct permanent
    transfer) is invisible to this - it only catches a conversion that
    actually shows up as its own row in transfers_processed.csv.
    """
    candidates = transfers_df[
        (transfers_df["player_id"] == loan_row["player_id"])
        & (transfers_df["from_club_name"] == loan_row["from_club_name"])
        & (transfers_df["to_club_name"] == loan_row["to_club_name"])
        & (transfers_df["transfer_date"] > loan_row["transfer_date"])
    ]
    if candidates.empty:
        return None
    nearest = candidates.sort_values("transfer_date").iloc[0]
    return {"transfer_date": str(nearest["transfer_date"])[:10], "success_score": float(nearest["success_score"])}


def filter_loans(position=None, league=None, q=None, min_age=None, max_age=None, min_duration=None, max_duration=None):
    """Shared position/league/q/age-range/duration-range filtering for /api/loans and /api/loans/export - see filter_transfers()'s identical reasoning for why this is factored out rather than duplicated per endpoint."""
    df = loans_df
    if position:
        df = df[df["position"] == position]
    if league:
        df = df[df["to_domestic_competition_id"] == league]
    if q:
        q_fold = fold_accents(q)
        mask = (
            df["_name_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_to_club_fold"].str.contains(q_fold, na=False, regex=False)
            | df["_from_club_fold"].str.contains(q_fold, na=False, regex=False)
        )
        df = df[mask]
    if min_age is not None:
        df = df[df["age_at_transfer"] >= min_age]
    if max_age is not None:
        df = df[df["age_at_transfer"] <= max_age]
    if min_duration is not None:
        df = df[df["tenure_days"] >= min_duration]
    if max_duration is not None:
        df = df[df["tenure_days"] <= max_duration]
    return df


def loan_row_dict(r):
    """One loans_processed.csv row as a plain dict - shared by /api/loans' JSON list and /api/loans/export's CSV (via rows_to_csv). converted_to_permanent is looked up per row (see find_loan_conversion) - cheap enough at list-page size (one filtered scan of transfers_df per loan row, ~25-100 of them per request) that it doesn't need precomputing at startup the way league_trends_df does."""
    conversion = find_loan_conversion(r)
    return {
        "player_id": int(r["player_id"]),
        "name": r["name"],
        "position": r["position"],
        "from_club": r["from_club_name"],
        "to_club": r["to_club_name"],
        "to_league": league_display_name(r["to_domestic_competition_id"]),
        "transfer_date": str(r["transfer_date"])[:10],
        "age_at_transfer": round(float(r["age_at_transfer"]), 1),
        "tenure_days": int(r["tenure_days"]),
        "still_on_loan": bool(r["still_on_loan"]),
        "loan_success_score": float(r["loan_success_score"]),
        "converted_to_permanent": conversion is not None,
        "conversion_transfer_date": conversion["transfer_date"] if conversion else None,
        "conversion_success_score": conversion["success_score"] if conversion else None,
    }


@app.get("/api/loans")
def list_loans(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "loan_success_score",
    order: str = "desc",
    limit: int = 25,
    offset: int = 0,
    min_age: float | None = None,
    max_age: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
):
    """Paginated, filterable, sortable listing of every scored loan spell, for the loans page's table."""
    df = filter_loans(position, league, q, min_age, max_age, min_duration, max_duration)

    sort_field = sort if sort in LOAN_SORT_FIELDS else "loan_success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset:offset + limit]

    results = [loan_row_dict(r) for _, r in page.iterrows()]
    return {"total": total, "limit": limit, "offset": offset, "results": results}


@app.get("/api/loans/export")
def export_loans(
    position: str | None = None,
    league: str | None = None,
    q: str | None = None,
    sort: str = "loan_success_score",
    order: str = "desc",
    min_age: float | None = None,
    max_age: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
):
    """Every loan matching the current filters (no /api/loans-style pagination cap) as a downloadable CSV - "export what you're looking at" for Loans, reusing filter_loans() so the file can never silently diverge from what the table shows."""
    df = filter_loans(position, league, q, min_age, max_age, min_duration, max_duration)
    sort_field = sort if sort in LOAN_SORT_FIELDS else "loan_success_score"
    df = df.sort_values(sort_field, ascending=(order == "asc"))
    csv_text = rows_to_csv([loan_row_dict(r) for _, r in df.iterrows()])
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=loans.csv"},
    )


@app.get("/api/loans/detail")
def loan_detail(player_id: int, transfer_date: str):
    """Look up one specific loan spell by (player_id, transfer_date) and return its full card - used by the loans page's click-to-view modal."""
    match = loans_df[
        (loans_df["player_id"] == player_id) & (loans_df["transfer_date"] == transfer_date)
    ]
    if match.empty:
        raise HTTPException(status_code=404, detail="loan not found")
    return build_loan_card(match.iloc[0])


def binned_trend(df, col, q):
    """
    Split df[col] into q roughly-equal-sized buckets (pandas qcut, so each
    bucket gets a comparable sample rather than a fixed-width range that
    could leave a bucket with a handful of transfers) and return each
    bucket's median `col` value, average success_score, and sample size -
    the points a fee-vs-score or age-vs-score chart draws its trend line
    through. `duplicates="drop"` collapses buckets that end up sharing an
    edge (e.g. many transfers at the exact same low fee), so q is an upper
    bound on the number of points returned, not a guarantee.
    """
    d = df[[col, "success_score"]].dropna()
    buckets = pd.qcut(d[col], q=q, duplicates="drop")
    grouped = (
        d.groupby(buckets, observed=True)
        .agg(x=(col, "median"), avg_score=("success_score", "mean"), n=("success_score", "size"))
        .reset_index(drop=True)
        .sort_values("x")
    )
    return [
        {"x": float(r["x"]), "avg_score": round(float(r["avg_score"]), 1), "n": int(r["n"])}
        for _, r in grouped.iterrows()
    ]


def numeric_column(series, ndigits=None):
    """
    A float column as a plain JSON-safe list, for the /api/analytics scatter
    payload: NaN -> None (DataFrame.where(cond, None) on a float64 column
    casts None right back to NaN to keep the column's dtype, so it can't be
    used for this - see the test this was caught by), everything else -> a
    real Python float, optionally rounded, since a scatter chart never needs
    more precision than that and unrounded fees/values roughly double the
    payload size.

    Never rounds a genuinely-positive value down to exactly 0 - transfer_fee/
    market_value_in_eur use 0 as a real "free transfer"/no-value sentinel
    (distinct from the None above), which the frontend's `> 0` filters rely
    on to separate "has this number" from "doesn't". Rounding a real €200 fee
    down to €0 at the nearest-€1k precision used here would silently and
    wrongly read on the frontend as free rather than merely imprecise - not
    reachable with today's data (the smallest real fee is €20k) but nothing
    else guards against it, so a future lower-fee row would corrupt the
    chart silently rather than just losing precision.
    """
    result = []
    for v in series:
        if pd.isna(v):
            result.append(None)
            continue
        rounded = round(v, ndigits) if ndigits is not None else v
        if v > 0 and rounded <= 0:
            rounded = 10 ** -ndigits
        result.append(float(rounded))
    return result


@app.get("/api/analytics/trends")
def get_analytics_trends():
    """
    Just fee_trend/age_trend (see binned_trend() above and /api/analytics
    below, which computes the same two series as part of its Analytics-page
    payload) - a separate, much lighter endpoint for the Predict page's
    "where does this land" marker, which only needs these two small series,
    not /api/analytics' full ~8,300-row scatter payload (columnar and
    rounded, but still a few hundred KB - wasteful to ship to a page that
    never draws a scatter plot at all).
    """
    df = transfers_df
    return {
        "fee_trend": binned_trend(df[df["transfer_fee"] > 0], "transfer_fee", q=10),
        "age_trend": binned_trend(df, "age_at_transfer", q=12),
    }


@app.get("/api/analytics")
def get_analytics():
    """
    Aggregate data for the Analytics page's four charts, computed fresh on
    every call (transfers_df is small enough - ~8,300 rows - that there's
    no need to precompute at startup the way league_trends_df is).

    - scatter: every permanent transfer's player_id/transfer_date/name/
      position/fee/market value/age/score, columnar (one array per field,
      not one object per row) - cuts the JSON payload roughly in half by
      not repeating eight field names 8,300 times. player_id/transfer_date
      let the frontend open a clicked point's full /api/transfers/detail
      card, the same way every other list page on the site does. Numeric
      values are rounded before serializing (fee/market value to the
      nearest €1k, age/score to 1 decimal) via numeric_column() since the
      frontend only plots them, never needs full precision. Fee and market
      value are left null for the transfers missing them (the same
      ~1.6%/39% gaps documented on prediction_surprises.csv's merge above)
      rather than dropped, so the age-vs-score chart - which needs neither
      - still gets every transfer; each chart filters out its own nulls
      client-side.
    - fee_trend/age_trend: binned_trend() over fee>0 transfers (fee can't
      sit on a log axis at 0) and all transfers respectively - the line
      overlaid on those two scatter charts. Also available alone, without
      the scatter payload, via /api/analytics/trends.
    - by_year: transfer count and average fee (free/unknown transfers
      counted as €0, same convention as league_trends_df's avg_fee) per
      year, excluding the current in-progress year - same reasoning as
      league_trends_df's `current_year` handling above, an in-progress
      year runs far below a normal year's count and would read as a
      sudden collapse rather than the incomplete data it is.
    """
    df = transfers_df

    scatter = {
        "player_id": df["player_id"].astype(int).tolist(),
        "transfer_date": [str(v)[:10] for v in df["transfer_date"]],
        "name": df["name"].tolist(),
        "position": df["position"].tolist(),
        "transfer_fee": numeric_column(df["transfer_fee"], ndigits=-3),
        "market_value_in_eur": numeric_column(df["market_value_in_eur"], ndigits=-3),
        "age_at_transfer": numeric_column(df["age_at_transfer"], ndigits=1),
        "success_score": numeric_column(df["success_score"], ndigits=1),
    }

    fee_trend = binned_trend(df[df["transfer_fee"] > 0], "transfer_fee", q=10)
    age_trend = binned_trend(df, "age_at_transfer", q=12)

    year_df = df.copy()
    year_df["year"] = pd.to_datetime(year_df["transfer_date"]).dt.year
    current_year = int(year_df["year"].max())
    by_year_grouped = (
        year_df[year_df["year"] < current_year]
        .groupby("year")
        .agg(transfers=("success_score", "size"), avg_fee=("transfer_fee", lambda s: s.fillna(0).mean()))
        .reset_index()
    )
    by_year = [
        {"year": int(r["year"]), "transfers": int(r["transfers"]), "avg_fee": round(float(r["avg_fee"]))}
        for _, r in by_year_grouped.iterrows()
    ]

    return {
        "scatter": scatter,
        "fee_trend": fee_trend,
        "age_trend": age_trend,
        "by_year": by_year,
    }


class NoCacheStaticFiles(StaticFiles):
    """
    StaticFiles that tells the browser never to cache a response at all
    (Cache-Control: no-store) rather than trusting a cached copy without
    even asking - the default (no explicit Cache-Control, just an ETag/
    Last-Modified pair) lets browsers apply heuristic caching, so editing
    style.css or app.js during development doesn't show up until a hard
    refresh, which is exactly what happened testing the modal-width
    change - even a brand-new tab kept serving the pre-edit CSS.

    no-cache (revalidate-before-use, but still cacheable) was tried first
    and wasn't reliable enough in practice - real-world browser/extension/
    proxy behavior around conditional-GET revalidation is inconsistent
    enough that a genuinely stale copy kept surfacing anyway (diagnosed
    directly: document.querySelector('script[src*="app.js"]').src showed
    no ?v= query string at all in an affected tab - a copy old enough to
    predate cache-busting being added in the first place). no-store is the
    unambiguous version - the browser is told not to persist the response
    at all, so there's nothing left to serve stale. The ?v=N query-string
    bump on every <link>/<script> tag (see app/static/*.html) stays too,
    as a second, independent safeguard - belt and suspenders for a bug
    class that kept recurring with just one fix in place.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/", NoCacheStaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
