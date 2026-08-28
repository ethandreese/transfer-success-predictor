"""
Backfill a real loan/free/paid transfer_type for every candidate transfer,
by reading it straight from transfermarkt's live transferHistory API instead
of the packaged Kaggle/GitHub dataset.

Why this is needed: the packaged dataset (dcaribou/transfermarkt-datasets)
parses each transfer's raw fee text down to a single number in its ETL, and
its parsing rule maps *any* non-numeric fee string - including "loan
transfer", "End of loan", and "Loan fee: EUR X" - to a flat 0, identical to a
genuine free transfer (Bosman move, academy graduate, released player). So
loans and free transfers are indistinguishable in transfers.csv even though
the live API's raw fee text distinguishes them clearly. See README for the
full investigation (verified against Jadon Sancho's real transfer history).

This script re-fetches each player's transfer history from
https://www.transfermarkt.co.uk/ceapi/transferHistory/list/{player_id} - an
unofficial, undocumented endpoint (it's what transfermarkt's own site calls,
not a published third-party API) - and classifies each transfer's raw fee
text into transfer_type: "paid", "free", "loan", or "unknown" (unparseable,
e.g. "-" or "?"). Results are appended incrementally to a resumable cache CSV
so a long run (tens of thousands of players, one request each) can be killed
and restarted without losing progress or re-hitting players already done.
build_dataset.py joins this cache onto transfers.csv by
(player_id, transfer_date, from_club_id, to_club_id) and drops loan rows
before any other computation, so a loan spell no longer gets scored as if it
were a permanent-transfer decision.

Usage:
    ./.venv/bin/python scripts/fetch_transfer_types.py [--limit N]
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd

RAW_DIR = os.environ.get(
    "TRANSFERMARKT_RAW_DIR",
    os.path.expanduser("~/.cache/kagglehub/datasets/davidcariboo/player-scores/versions/677"),
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transfer_types_cache.csv")
MIN_DATE = pd.Timestamp("2013-01-01")
MAX_DATE = pd.Timestamp.today().normalize()

API_URL = "https://www.transfermarkt.co.uk/ceapi/transferHistory/list/{player_id}"
USER_AGENT = "transfer-success-predictor research script (contact: ethandreese@gmail.com)"
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3

FIELDS = ["player_id", "transfer_date", "from_club_id", "to_club_id", "fee_raw", "transfer_type"]


def classify_fee(fee_raw):
    """
    Classify a raw transfermarkt fee string into paid/free/loan/unknown.
    Real examples seen from the live API: "loan transfer", "End of loan",
    'Loan fee:<br /><i class="normaler-text">EUR5.90m</i>', "free transfer",
    "EUR85.00m", "-", "?".
    """
    if not fee_raw:
        return "unknown"
    text = re.sub("<[^>]+>", "", fee_raw).strip().lower()
    if "loan" in text:
        return "loan"
    if text == "free transfer":
        return "free"
    if text.startswith("€") or text.startswith("eur"):
        return "paid"
    return "unknown"


def club_id_from_href(href):
    """Extract the numeric club id from a transfermarkt club href like '/dortmund/transfers/verein/16/saison_id/2022'."""
    match = re.search(r"/verein/(\d+)", href or "")
    return int(match.group(1)) if match else None


def fetch_player_transfers(player_id):
    """Fetch and classify one player's transfer history. Returns a list of row dicts, or None if every retry failed."""
    url = API_URL.format(player_id=player_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            rows = []
            for t in data.get("transfers", []):
                date = t.get("dateUnformatted")
                # "0000-00-00" is a known placeholder the live API returns
                # for a small number of transfers with no real recorded date
                # (dcaribou's own ETL filters the same sentinel out) - it can
                # never join onto transfers.csv (which has no such rows), so
                # skip it here rather than caching an unusable row.
                if not date or date == "0000-00-00":
                    continue
                rows.append({
                    "player_id": player_id,
                    "transfer_date": date,
                    "from_club_id": club_id_from_href((t.get("from") or {}).get("href")),
                    "to_club_id": club_id_from_href((t.get("to") or {}).get("href")),
                    "fee_raw": t.get("fee"),
                    "transfer_type": classify_fee(t.get("fee")),
                })
            return rows
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED player {player_id}: {e}")
                return None


def candidate_player_ids():
    """Unique player ids across every candidate transfer in transfers.csv, matching build_dataset.load_transfers()'s own filters (date range, no-op moves)."""
    df = pd.read_csv(
        os.path.join(RAW_DIR, "transfers.csv"),
        usecols=["player_id", "transfer_date", "from_club_id", "to_club_id"],
        parse_dates=["transfer_date"],
    )
    df = df.dropna(subset=["transfer_date", "from_club_id", "to_club_id"])
    df = df[(df["transfer_date"] >= MIN_DATE) & (df["transfer_date"] <= MAX_DATE)]
    df = df[df["from_club_id"] != df["to_club_id"]]
    return sorted(df["player_id"].astype(int).unique())


def load_done_player_ids():
    """Player ids already present in OUT_PATH, so a re-run resumes instead of re-fetching everyone from scratch."""
    if not os.path.exists(OUT_PATH):
        return set()
    done = pd.read_csv(OUT_PATH, usecols=["player_id"])
    return set(done["player_id"].unique())


def main():
    """Fetch and cache transfer types for every not-yet-done candidate player, printing progress every 200 players."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N not-yet-done players (for testing)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    player_ids = candidate_player_ids()
    done = load_done_player_ids()
    todo = [p for p in player_ids if p not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(player_ids):,} candidate players, {len(done):,} already cached, {len(todo):,} to fetch", flush=True)

    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        start = time.time()
        for i, player_id in enumerate(todo):
            rows = fetch_player_transfers(player_id)
            if rows:
                writer.writerows(rows)
                f.flush()
            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(todo) - i - 1) / rate if rate > 0 else float("nan")
                print(f"  {i + 1:,}/{len(todo):,} fetched ({rate:.1f}/s, ~{remaining / 60:.0f} min left)", flush=True)
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Done. Cache at {OUT_PATH}")


if __name__ == "__main__":
    main()
