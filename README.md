# Transfer Success Predictor ⚽

[![Tests](https://github.com/ethandreese/transfer-success-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/ethandreese/transfer-success-predictor/actions/workflows/tests.yml)

Predicts how a football (soccer) transfer is likely to go, trained on real
historical transfer data rather than hand-picked examples like Haaland→City
or Dembélé→Barça.

**Contents:** [Project layout](#project-layout) · [Running it](#running-it) ·
[Testing](#testing) · [How it works](#how-it-works) · [Pages](#pages) ·
[Known limitations](#known-limitations) · [Project history](#project-history)

## Project layout

```
scripts/fetch_transfer_types.py           # (optional) backfills data/raw/transfer_types_cache.csv from transfermarkt's live API
scripts/fetch_fotmob_stats.py             # (optional) backfills data/raw/fotmob_stats_cache.csv from FotMob's public stat leaderboards
scripts/fetch_pretransfer_fotmob_stats.py # (optional) backfills data/raw/pretransfer_fotmob_stats_cache.csv - the predict model's pre-transfer FotMob signal
scripts/fetch_current_fotmob_stats.py     # (optional) backfills data/raw/current_fotmob_stats_cache.csv - live predictions' "current form" FotMob signal
scripts/build_dataset.py   # raw Transfermarkt CSVs -> data/transfers_processed.csv + data/loans_processed.csv
scripts/train_model.py     # trains the deployed model + comparable-transfers index
scripts/compute_prediction_surprises.py  # 5-fold held-out predictions for every transfer -> data/prediction_surprises.csv (the Model vs Reality page)
scripts/build_lookups.py   # small player/club search tables for the web app
app/main.py                 # FastAPI backend (serves the API + the static frontend)
app/static/                 # vanilla HTML/CSS/JS frontend (index=home/predict/browse/loans/compare/surprises/clubs/player/leagues/analytics/about)
data/                        # committed: small derived CSVs only (~8MB total)
tests/                       # pytest suite - runs against committed artifacts only
```

The raw Transfermarkt CSVs (~730MB) are **not** committed — they're
downloaded locally via [kagglehub](https://github.com/Kaggle/kagglehub) and
only used to regenerate `data/*.csv`.

## Running it

The trained model and processed data are already committed to the repo,
so if `.venv/` already exists (it's gitignored, but you may have set it
up already), just start the server:

```bash
./.venv/bin/uvicorn app.main:app --reload
```

Then open http://localhost:8000.

**First-time setup** (no `.venv/` yet):

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload
```

**Regenerating `data/*.csv` and the model from scratch** is only needed if
you want to rebuild from the raw Transfermarkt dataset (e.g. after it's
updated, or after changing the scoring formula) — not needed to just run
the app:

```bash
./.venv/bin/python -c "import kagglehub; kagglehub.dataset_download('davidcariboo/player-scores')"
./.venv/bin/python scripts/fetch_transfer_types.py           # optional but recommended - see below
./.venv/bin/python scripts/fetch_fotmob_stats.py              # optional but recommended - see below
./.venv/bin/python scripts/fetch_pretransfer_fotmob_stats.py  # optional but recommended - see below
./.venv/bin/python scripts/build_dataset.py
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/fetch_current_fotmob_stats.py      # optional but recommended - see below
./.venv/bin/python scripts/build_lookups.py                   # run again to merge in the current-FotMob snapshot just fetched
./.venv/bin/python scripts/train_model.py
./.venv/bin/python scripts/compute_prediction_surprises.py
```

`fetch_transfer_types.py` re-fetches every candidate player's real transfer
history from transfermarkt's live API to tell loans apart from free
transfers — one request per player (~23k), politely rate-limited, so it
takes a few hours and is safe to interrupt and re-run (it resumes from
`data/raw/transfer_types_cache.csv` rather than starting over). It's
optional: skip it and `build_dataset.py` still runs, just without loan
detection — every zero-fee transfer (including loans) is treated as a
permanent transfer, and `data/loans_processed.csv` comes out empty.

`fetch_fotmob_stats.py` pulls FotMob's season stat leaderboards for the 23
leagues in `LEAGUE_MAP` and stitches each tenure — both permanent
transfers and loan spells, in one pass — into
`data/raw/fotmob_stats_cache.csv`. A few thousand requests across ~14
seasons x 23 leagues, politely rate-limited, resumable from
`data/raw/fotmob_season_cache/` (gitignored) rather than refetching every
league-season from scratch. Also optional: skip it and `build_dataset.py`
still runs, just without the four FotMob components — each one's weight
is dropped and the other weights renormalized for every transfer, the
same as when resale data is unknown.

`fetch_pretransfer_fotmob_stats.py` and `fetch_current_fotmob_stats.py`
feed the *predict model* (not the historical score). Both reuse
`fetch_fotmob_stats.py`'s season-fetch/cache and matching functions
directly (imported, not duplicated), so if that cache is already warm
these run in seconds, not hours. Both are optional the same way: skip
either and `train_model.py`/`build_lookups.py` still run, just with every
`pre_fotmob_*` feature (or every searched player's `recent_fotmob_*`
autofill) falling back to the median/`None` a missing match already
degrades to.

`compute_prediction_surprises.py` is **not** optional like the four above -
`app/main.py` loads `data/prediction_surprises.csv` unconditionally at
startup (the Model vs Reality page's data source). It reuses
`train_model.py`'s exact feature-engineering functions and feature lists,
swapping 5-fold cross-validation in for that script's one temporal split,
so every transfer gets a `predicted_score` from a model that never saw that
transfer's own outcome during fitting - unlike `model.joblib` itself, which
is refit on the full dataset for serving accurate live predictions, at the
cost of its own predictions on historical transfers being partly circular
(it saw the answer). Runs in seconds; doesn't touch `model.joblib`.

## Testing

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/pytest
```

The suite runs entirely against the committed `data/*.csv` and
`app/model/*.joblib` artifacts (no raw-dataset dependency), covering:
`tests/test_score_formula.py` (position weights sum to 1.0, every processed
transfer's `success_score` matches its stored sub-components recomputed
through `score_weights.json`, no nulls/out-of-range values) and
`tests/test_app.py` (every API endpoint, including accent-insensitive
search, the compare/predict flows, `/api/surprises`' sort directions and
held-out-only filtering, `/api/clubs/leaderboard`'s minimum-sample
thresholds and name-alias merging, `/api/players/{id}/career`'s
transfer+loan merge and graceful handling of a player missing from
`players_lookup.csv`, and `/api/leagues/trends`' minimum-sample
thresholds, disjoint early/recent windows, and exclusion of the current
in-progress year).

## How it works

**Data.** [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)
(mirrored on Kaggle as [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores)),
~50k players, ~175k transfers, ~1.9M appearances, ~656k market valuations.

**Success score (the training label, 0–100).** Computed from what
happened *after* the move, measured over the player's **entire tenure**
at the new club (transfer date until their next departure, or "now" if
they're still there) — not just year one, so a slow starter who adapted
late and a hot starter who faded aren't both mis-scored by a fixed
first-year window.

Built from ten sub-metrics, each converted to a percentile rank (so no
single stat's raw scale dominates), then blended with **weights that vary
by position and sub-position** (`data/score_weights.json`):

- **Performance level & change** — goal contributions/90 at the new club,
  adjusted for how hard it is to score in that specific league, then
  ranked within the player's position. "Change" compares against a
  regression-based *expectation* given their pre-transfer level, not a
  raw before/after difference — otherwise an already-elite player has
  nowhere to go but "down" on paper even while staying elite. Folded
  together with the FotMob "attacking" bucket below when both exist, so
  the two don't double-count the same signal.
- **Market value growth** — an 80/20 blend of growth *relative to the
  player's own starting value* and the *absolute euro gain*, each itself
  weighted 80/20 toward the tenure's peak value over its end-of-tenure
  value (a long career's market value naturally declines with age by the
  time a player leaves, which shouldn't read as failure).
- **Playing time** — 60% percent of the team's actual games played
  (catches injuries/rotation a raw count hides), 40% raw appearance
  count, with a forgiveness curve for a normal amount of missed games
  before ranking (even clearly-elite players miss a median of ~12 games
  a season). Weighted highest for goalkeepers, where winning the single
  starting job is an unusually clean signal.
- **Value for money** — fee vs. a position-weighted blend of every
  on-pitch signal, judged against the player's own market value at
  signing rather than the whole dataset's fee distribution. A premium up
  to 1.3x pre-transfer value counts as normal and gets zero penalty; only
  real overpays are ranked, against each other. Its weight decays with
  tenure length — a long, clearly successful career is no longer judged
  much on whether the initial fee looked reasonable.
- **Resale profit** — did the buying club later resell the player for a
  profit? Only counted when a real subsequent sale exists (~25% of
  transfers); weight decays with tenure length, the same idea as value
  for money.
- **Four FotMob-derived components** (rating, attacking, defensive,
  possession) — kept separate rather than blended into one number, so an
  attack-minded fullback and a purely defensive one don't collapse to the
  same score. League-adjusted and ranked within position; each
  component's weight is dropped and the rest renormalized whenever
  FotMob has no data for that specific bucket.

Weight shifts from goal-based components (heavy for attackers, ~0 for
goalkeepers, since goal contributions are meaningless for keepers) into
the FotMob defensive/attacking/possession buckets as a position relies
less on scoring. A handful of sub-positions (Defensive Midfield,
Centre-Back, full-backs, wingers) get their own weight row shifting
further toward defending/possession or attacking specifically, and the
four outfield defensive sub-positions weight FotMob's overall `rating`
at 30% instead of the usual 20% — checked directly, `rating` correlates
with the final score far more than raw defensive-action counting does
for those roles specifically (elite defenders often need to make fewer
visible defensive actions, not more).

**Loans are detected and scored separately**, not folded into the
permanent-transfer formula. The packaged dataset has no loan/permanent
flag at all — its ETL parses "loan transfer" and "€0" down to the same
flat fee — so loans are identified by re-fetching each transfer's real
fee text from Transfermarkt's own transfer-history API
(`scripts/fetch_transfer_types.py`) and scored on
`data/loan_score_weights.json` instead: no value-for-money or resale
component (most loans carry no real fee and don't end in a sale), and
playing time weighted much more heavily — whether the loan delivered
game time is usually the central question it's judged on. A loan spell
with zero appearances is kept, not filtered out, since that's a real
(bad) outcome the Loans tab exists to surface.

**The FotMob components** fill a real gap in the base Transfermarkt data,
which has no column for defensive output at all (tackles, saves, clean
sheets). `scripts/fetch_fotmob_stats.py` pulls FotMob's public season
leaderboards for the 23 leagues that show up as a transfer destination
and stitches each tenure's *entire* window (matching the rest of the
score's philosophy), correctly handling multi-season tenures and
same-league mid-season moves (FotMob otherwise attributes an entire
season to whichever club a player is registered at when fetched, not
split by stint). See [Known limitations](#known-limitations) for its
real coverage ceilings.

**Model.** A ridge-regularized linear regression trained on pre-transfer-
only features (age, position, physical attributes, fee, market value,
prior-year performance including FotMob rating/xG/xA/passing/defensive
output, and origin/destination club & league strength) — nothing about
what happened after the move. Evaluated on a temporal holdout (trained on
transfers before mid-2023, tested on transfers since): **MAE ≈ 12.26
points** on the 0–100 scale, **R² ≈ 0.211**, vs. ≈14.37 MAE for always
predicting the average. That's a modest but real signal — predicting a
player's *entire future tenure* from pre-transfer stats alone is
genuinely hard, since multi-year outcomes depend heavily on injuries,
tactics, and squad fit no pre-transfer number can see. (The model was
originally a tuned `GradientBoostingRegressor`; a later investigation
found a plain `Ridge` beat it and every other alternative tried — see
[Project history](#project-history).)

**Explainability.** For a known historical transfer, the app shows the
real 5-10 component breakdown with concrete numbers behind each one
("0.92 goal contributions/90 at Barcelona, ranked vs. other attackers").
For a hypothetical prediction, it shows each feature's contribution via a
leave-one-out swap against a "typical transfer" reference — but the
reference itself is contextual, not one flat number for the whole
dataset: position-conditional medians for stats that vary by role, a
paid-vs-free-specific fee reference (over half of all transfers are
free, which would otherwise drag the "typical fee" to €0), a fee
expectation regression against the player's own market value, and short
explanatory notes wherever a bare "X vs. typical Y" swap would otherwise
read as a claim the data doesn't actually support (e.g. the Premier
League's lower average score is almost entirely a fee-premium effect,
not an on-pitch one — the note says so explicitly).

## Pages

The nav groups every page under one of three dropdowns (**Predict**,
**Browse**, **Insights**) plus **Home** and **About** on their own - nine
pages was too many flat links, so `/` itself became a real landing page
rather than doubling as the predict form (see Project history).

- **`/`** — the home page: a short pitch, the curated showcase cards (moved
  here from the old `/`), and a link to every other page grouped the same
  way as the nav.
- **`/predict.html`** — predict a hypothetical transfer: search a real
  player, pick a destination club, see a predicted score, a likely range
  (min/max among the 5 most similar real transfers, since a single point
  estimate overstates how confident a model this size can be), a "why this
  score" breakdown, and the nearest historical comparables. A "Where this
  lands" section plots the prediction as a marker on the sitewide fee-vs-
  score and age-vs-score trend lines (`/api/analytics/trends` - the same
  `fee_trend`/`age_trend` series Analytics draws, without its much larger
  scatter payload), so the number isn't shown divorced from how every
  other real transfer at a similar fee/age actually went. The fee field
  also has a slider alongside the number input - dragging it (or typing a
  new number) debounce-re-predicts live once a first prediction already
  exists, so trying a range of fees doesn't need a re-click of "Predict
  success" for each one. Age has no equivalent slider - a player's age at
  a hypothetical transfer isn't something to explore a range of the way a
  fee is, so it stays a plain editable field.
- **`/compare.html`** — set up 2-4 hypothetical transfers side by side
  (same player to different clubs, or entirely different players) and see
  every prediction, range, and top factors together, with the highest
  score highlighted and a plain-English verdict ("X scores highest at Y,
  Z points ahead of the next best"). Starts with two columns; "+ Add
  another option" reveals a 3rd and 4th. A finished comparison syncs
  every active scenario (player/club/fee/age) to the URL, so it's
  bookmarkable and shareable - opening that link resolves every player
  and club by id (`/api/players/{id}`, `/api/clubs/{id}`) and re-runs the
  comparison automatically. A stale or mistyped link (an id that no
  longer resolves) leaves the form untouched rather than populating
  whichever scenarios happened to succeed while another sat blank.
- **`/browse.html`** — every scored transfer (~8,300), filterable by
  position, destination league, fee range, and age range, searchable by
  player/club name, sortable by score/date/fee/age, paginated. Click any
  row to open that transfer's full card (score + breakdown) in a modal.
  Every filter/search/sort selection syncs to the URL, and an "Export
  these results as CSV" link (`/api/transfers/export`, uncapped - it
  ignores pagination and returns every matching row) always reflects the
  current filters.
- **`/loans.html`** — every scored loan spell (~4,300), same browse/filter/
  search/click-to-view-card/URL-sync/CSV-export experience as
  `/browse.html` (filterable by age range and loan duration rather than
  fee, since most loans carry no real fee), but scored on the loan-specific
  formula above (no fee/resale rows in the breakdown, and duration shown
  in months rather than years). A loan whose player later signed
  permanently for the same club they were on loan at (detected by matching
  player + from/to club against a later row in the transfers data - see
  `find_loan_conversion()`) gets a "✓ Permanent" badge, in both the table
  and the detail card, linking to that conversion's date and score.
- **`/surprises.html`** ("Model vs Reality") — every scored permanent
  transfer with a held-out prediction (~93% of them - see
  `compute_prediction_surprises.py`), ranked by how far the real outcome
  diverged from what a model that never saw that transfer's own result
  would have guessed from pre-transfer data alone: the biggest
  overachievers, the biggest busts, or (via the Show filter) the calls it
  got closest to right. Same filter/search/click-to-view-card experience as
  Browse, plus the model's own number shown alongside the real one.
- **`/clubs.html`** — every club that's bought or sold at least one scored
  permanent transfer, ranked as a recruiter: incoming transfers and their
  average score, total spent, buy-develop-resell profit on the subset it
  later sold on again, and how players it let go performed at their *next*
  club. A club needs at least 5 transfers (3 for resale-profit ranking)
  before a rate-based ranking includes it. Click a club for its full report
  card (best/worst signing, best/worst flip, best/worst departure) - each
  of those six highlights is itself clickable, drilling into that specific
  transfer's own full breakdown card (with a link back to the club view) -
  and a link to its complete transfer history on Browse.
- **`/player.html`** — search a player to see their whole scored career as
  a timeline: one point per permanent transfer or loan, positioned by its
  real date (not just evenly spaced) and colored by score, connected in
  chronological order. A loan renders as a hollow ring instead of a solid
  dot - same color scale, visually distinct without a second legend. Click
  any point (or the table row below it) for that stop's full breakdown
  card. Search covers every player with a scored transfer or loan, not
  just the smaller set the Predict page's autocomplete can see (see Known
  limitations).
- **`/leagues.html`** — every league with at least 15 scored permanent
  transfers, ranked by transfer volume, average score/fee, or the change
  between its earliest and most recent 3 complete years (the current,
  in-progress year is excluded from every average). Click a league for a
  chart indexing average fee and average success score to the same
  early-period baseline, plus a plain-language verdict sentence - answers
  "are fees inflating faster than performance?" directly rather than
  leaving two differently-scaled numbers for the reader to compare
  themselves.
- **`/analytics.html`** — four hand-drawn SVG charts over every scored
  permanent transfer (`/api/analytics`, computed fresh per request - the
  dataset's small enough that there's no need to precompute at startup):
  fee vs. success score and age vs. success score (both scatter plots with
  a binned trend line overlaid), fee vs. market value (log-log scatter
  with a y=x reference line, dots still colored by outcome), and transfer
  count vs. average fee by year (both indexed to the first year = 100, the
  same trick League Trends' own chart uses to share one axis between two
  differently-scaled series). No table or filters, but every chart is
  click/tap-interactive: a scatter point opens that transfer's full
  `/api/transfers/detail` card, and a trend-line/yearly point shows a
  small tooltip with its exact numbers (see `showChartTooltip()` in
  `analytics.js` - a native `<title>` hover tooltip is also present as a
  free bonus on desktop, but never fires on a touch device, so it isn't
  the thing actually relied on).
- **`/about.html`** — what the site does and how the numbers are computed,
  in plain language.

## Known limitations

- **FotMob coverage is real but partial.** ~67% of scored permanent
  transfers (~39% of loans) get at least one of the four FotMob
  components; coverage per bucket runs 54-67% even among those. Only the
  23 leagues in `LEAGUE_MAP` are covered at all, each from whatever
  season FotMob's own history starts for that league (2016/17 for the
  big five, as late as 2021/22 for Greece), and Ukraine's Premier League
  has no defensive/possession/rating data on FotMob at all. A transfer
  missing a bucket just has that component's weight dropped and the rest
  renormalized — never guessed at.
- **FotMob/Transfermarkt player and club matching is name-based** (the
  two sites share no id), which is inherently approximate. Extensively
  debugged for identity-collision bugs (see [Project history](#project-history)),
  but the least-scrutinized of the 23 leagues likely still has a
  somewhat weaker match rate than the handful that were checked by hand.
- **Only transfers with ≥10 appearances in both the year before and the
  whole tenure after are scored** (~8,358 of ~112k permanent-transfer
  candidates), which skews the dataset toward established first-team
  players over fringe moves. Loans use a looser bar — only the pre-loan
  side needs ≥10 appearances, since a loan with zero game time afterward
  is a real outcome worth showing, not missing data.
- **The packaged `transfers.csv` has real gaps** (a handful of famous
  moves, like Eden Hazard's 2019 Real Madrid transfer, were entirely
  missing). `data/manual_transfers.csv` is a hand-verified backfill,
  built by cross-checking `appearances.csv`'s club-history against
  `transfers.csv` and re-fetching real gaps from Transfermarkt's own API
  — see [Project history](#project-history) for the full trail.
- **Loan detection depends on a one-time scrape** of Transfermarkt's live,
  unofficial transfer-history API — about 98% of candidates match a
  fetched record; the rest default to "unknown" and are scored as a
  permanent transfer. A loan that converts to a permanent deal at the
  same club, with no separate recorded event for the conversion, still
  scores as one continuous loan rather than splitting at the conversion
  point.
- **Sub-position weighting uses a player's single most-common fielded
  position across their entire career**, not the specific window of any
  one transfer — a player who genuinely changed roles mid-career has
  every transfer weighted by whichever role dominates their overall
  history.
- **Predicting a new hypothetical transfer is meaningfully less reliable**
  than the historical scores shown for known transfers, since the model
  only ever sees pre-transfer information by construction.
- **The Model vs Reality page's `predicted_score` isn't the deployed
  model.** It comes from 5-fold cross-validation (see
  `compute_prediction_surprises.py`), so every transfer's number is honest
  in the sense that matters for ranking "how surprising was this" (no
  prediction ever saw its own outcome) - but each fold is a slightly
  different fit than `model.joblib`, which is refit on the *full* dataset
  for serving live predictions. Don't expect it to match a live
  `/api/predict` call for the same inputs exactly.
- **Every club name on the site is identified purely by its name string** -
  `transfers_processed.csv`/`loans_processed.csv` have no club_id at all,
  and never join `clubs_lookup.csv` (whose own naming, "Manchester City",
  doesn't reliably match the raw Transfermarkt names here either, e.g. "Man
  City" - a separate, unresolved gap between the two files).
  `build_club_name_aliases` merges same-club spellings two ways:
  mechanically, for a generic legal-entity marker or accent encoding ("FC
  Barcelona"/"Barcelona", "Fenerbahçe"/"Fenerbahce", "Genoa"/"Genoa CFC",
  "Beşiktaş Jimnastik Kulübü"/"Besiktas" - ~110+ clusters, hand-checked one
  by one before shipping, across both transfers and loans); and via
  `CLUB_NICKNAME_GROUPS`, a curated list of nickname/official-name pairs
  that share no token at all ("Man City"/"Manchester City",
  "PSG"/"Paris Saint-Germain", "Bor. Dortmund"/"Borussia
  Dortmund"/"Dortmund", "Sporting"/"Sporting CP", "Swansea"/"Swansea City",
  "Wolves"/"Wolverhampton Wanderers", and ~130 more) - each entry
  individually verified against the real data (matching
  `domestic_competition_id` on both sides, a non-contradictory
  `transfer_date` range) rather than assumed from football knowledge alone,
  which is exactly what caught that "Sporting" alone is safe to fold into
  Sporting CP specifically (every row carries league `PO1`, never Sporting
  Gijón's `ES1` or Royal Charleroi's `BE1`) rather than the genuinely
  ambiguous case it looks like at a glance - and, in the other direction,
  that a mechanical strip of the generic word "club" would have wrongly
  merged "Racing" with the unrelated Argentine "Racing Club" (`ARG1`),
  force-split instead (`CLUB_NAME_FORCE_SPLIT`). A second full sweep (see
  the comment above `CLUB_NICKNAME_GROUPS` in `app/main.py`), triggered by
  "Swansea"/"Swansea City" still showing as two rows, checked every pair of
  names in the dataset sharing a whole word and turned up ~95 more genuine
  merges, plus several look-alike traps rejected the same way "Racing" was:
  "Barcelona"/"RCD Espanyol Barcelona" (two different Barcelona clubs),
  "Rangers"/"Queens Park Rangers" (Glasgow Rangers, not QPR), "Krasnodar"/
  "Kuban Krasnodar" (FC Krasnodar and the dissolved Kuban Krasnodar are two
  different clubs from the same city), and a club's B/reserve/youth side
  ("Benfica"/"Benfica B" and six more), which runs its own transfer history
  and was left split on purpose. One cluster - Ukrainian "Metalist Kharkiv",
  which went bankrupt around 2016 with two organizations since laying claim
  to the name - came back a genuine, still-contested identity dispute rather
  than a clean call; per user confirmation, only "Metalist Kharkiv" and its
  dissolution-marked spelling are merged, and bare "Metalist"/"Metalist
  1925" are left split. A third pass matched on string similarity instead of
  shared whole words (to catch abbreviation/typo-style pairs the second
  pass's method would miss, like "Y. Malatyaspor"/"Yeni Malatyaspor" or a
  genuine data-entry inconsistency like "FC Helsingör"/"FC Helsingør") and
  found 16 more genuine merges, plus the same kind of rejections: "Atalanta"/
  "Atlanta" (Italy's Atalanta BC vs. MLS's Atlanta United), "Metalurg D."/
  "Metalurg Z." (two different Ukrainian clubs that just share a
  single-top-flight country), and another contested-history case - Belgian
  "Beerschot AC" (bankrupt 2013) left split from the later-reformed
  "Beerschot VA", the same shape as the Metalist call. Applied once at startup to
  `to_club_name`/`from_club_name` in both `transfers_df` and `loans_df`
  (and to `comparables["meta"]`, the nearest-neighbors index behind
  Predict's "similar historical transfers"), so every page - Browse, Loans,
  Model vs Reality, Predict, Compare, and Club Report Cards - shows the
  same name for the same club. Neither list is exhaustive across all ~700+
  club names here - a club whose nickname/official-name split was never
  spot-checked still shows up as two or more separate identities everywhere
  it appears, each missing part of the real history.
- **`players_lookup.csv` covers far fewer players than have real scored
  history** - it's filtered to a market-value threshold for the Predict
  form's autofill (see `build_lookups.py`), which excludes ~59% of every
  player with a scored transfer or loan (checked directly), mostly
  retired or lower-value players. `/player.html`'s search
  (`/api/players/career-search`) works around this by searching
  `transfers_df`/`loans_df` directly instead, but a career page for one of
  those players still can't show a position or current club in its header
  - both come from `players_lookup.csv` and are simply absent for players
  missing from it.
- **League Trends buckets by calendar year, not by season**, and only ever
  compares the destination league's own earliest/most recent 3-year
  windows against each other - it doesn't control for anything else that
  changed across a league's whole football economy over a decade (transfer
  windows shifting, financial fair play rules, a different mix of buying
  clubs), so "fees grew faster than success" is a real, checked pattern in
  this data, not a claim about *why*. Only 14 of the dataset's 23 leagues
  have enough transfers (≥15) to appear at all, and only those 14 also
  happen to have enough year-span (≥6 complete years) for a trend - true
  for every league today, but `build_league_trends` still handles a future
  league that clears the first bar without the second, showing its basic
  stats with no chart rather than disappearing it outright.
- **League-adjustment feeds the historical label, not the predict
  model's own features** — tested directly as additional model inputs
  (league baselines, a league-adjusted performance number) and it didn't
  beat the simpler feature set on the holdout, so it's kept for
  explanation display only (`data/league_baselines.csv`), not fed to the
  model itself.

## Project history

A condensed changelog of notable fixes and investigations, newest first
within each group. Full reasoning and numbers for anything here are in
the git history.

**Predict model & backend**
- Fixed `/api/predict` (and everything built on it - Compare, the
  examples cards) 400ing whenever a composite FotMob feature (e.g.
  "defensive") had real data overall but one specific raw sub-stat was
  individually missing - e.g. an attacker with real tackle/interception/
  recovery numbers but no recorded clearances at all, since attackers
  rarely attempt any. `explain_prediction`'s bulleted-breakdown builder
  tried to number-format that one `None` straight into the string and
  crashed; `describe_fotmob_component()` (the historical-score version of
  the same breakdown) already handled this correctly by only listing
  sub-stats that are actually present, so the pre-transfer version now
  does the same. Not a rare edge case - partial FotMob bucket coverage
  (~54-67%, per Known limitations) is the common case, not the exception.
  Caught testing 3-way Compare by hand.
- Switched the predict model from `GradientBoostingRegressor` to a plain
  `Ridge` regression after benchmarking it against every tree-based
  alternative tried (HistGradientBoosting, RandomForest, ExtraTrees, and
  GBR variants) — Ridge won on every one of 5 different temporal splits.
  MAE 12.40→12.26, R² 0.185→0.211. Fixed an overfit-coefficient issue for
  thin-data leagues along the way (`OneHotEncoder(min_frequency=30)`).
- Investigated feature interactions, a quadratic age term, polynomial
  features, robust-loss regression, and nonlinear alternatives
  (KernelRidge, SVR) as follow-ups to the Ridge switch; none justified
  their added complexity over the plain model.
- Investigated several new feature ideas (destination-club league
  form/position, manager tenure at signing, reconstructed squad
  age/nationality mix, player/destination nationality fit) and two
  new-data-acquisition angles (a newer Kaggle dataset version, external
  sources like Wikipedia pageviews and Transfermarkt injury history);
  none produced a gain worth shipping.
- Found `fee_to_value_ratio` was computed from a column missing for
  38.7% of transfers, silently excluding them from model training —
  fixing it recovered 41% of usable training data (MAE 12.65→12.40, R²
  0.159→0.185).
- Investigated a more historically-accurate, per-transfer-date club-value
  feature (reconstructed from contemporaneous roster valuations); it
  measurably hurt the model despite being more accurate, so the simpler
  flat proxy was kept.
- Fixed the loan pipeline's early filter referencing the wrong
  (permanent-transfer) position-weight keys — harmless today only by
  coincidence, fixed for future-proofing.
- Fixed an unbounded `limit` query param on the player/club search
  endpoints, and deduplicated a club-value-proxy computation that had
  drifted into two independently-maintained copies.

**Data pipeline & scoring formula**
- Backfilled thousands of missing transfers into
  `data/manual_transfers.csv` via Transfermarkt's live transfer-history
  API, after discovering real gaps in the packaged `transfers.csv`
  (starting from Eden Hazard's entirely-missing 2019 Real Madrid move).
  Fixed several related bugs along the way (loan/permanent
  misclassification, `tenure_end` defaulting to "today" for players
  whose next real move wasn't recorded). Current scored counts: 8,358
  permanent transfers, 4,306 loans.
- Found and fixed several FotMob player/club identity-matching bugs
  (fuzzy-match collisions, ambiguous shared club-name words, an
  asymmetric similarity-ratio bug) that were silently attributing a
  tenure to a different real player's or club's stats.
- Fixed FotMob season-stitching contamination at both ends of a tenure —
  a mid-season arrival or departure could leak a different club's season
  into the aggregate (confirmed on Memphis Depay, Danny Ings, Alexander
  Isak, among others).
- Fixed Norway/Sweden silently returning the wrong (always-current)
  FotMob season due to a single-year vs. split-year season-label
  mismatch.
- Fixed a home/away misattribution bug in `appearances.csv` affecting a
  handful of players' pre-transfer appearance counts.
- Fixed mangled league display names (title-cased URL slugs like "Pko Bp
  Ekstraklasa") across 18 competitions.
- Corrected two inaccurate claims on `about.html` (a missing
  `resale_profit` component in its own description; an overstated
  symmetric playing-time rule that doesn't apply to loans).

**Frontend**
- Added fee/age range filters and a "Export these results as CSV" link
  (`/api/transfers/export`, `/api/loans/export` - uncapped, ignores
  pagination) to Browse and Loans, plus a "✓ Permanent" badge on any loan
  whose player later signed permanently for the same club, in both the
  table and the detail card. The conversion check
  (`find_loan_conversion()`) matches a loan row against a later transfer
  row on player id + exact from/to club, taking the earliest such match;
  verified against a real example (Timur Suleymanov's 2023 Pari NN → Loko
  Moscow loan converting to a permanent transfer in mid-2024). All three
  filters, the export link, and the new range inputs stay in sync with
  each other and the URL through the same `currentFilterParams()`/
  `syncURL()` pattern the existing search/position/league filters already
  used. The fee inputs' label and displayed value didn't update on a
  currency change at first - the same bug Predict's fee field already had
  and fixed (see `updateFeeCurrencyDisplay` below); ported the same fix to
  Browse, keeping `state.minFeeEurM`/`maxFeeEurM` as the real
  currency-independent bounds and only converting for display.
- Extended Compare from a fixed A/B pair to 2-4 scenarios: "+ Add another
  option" reveals a 3rd/4th column (each removable), the results grid
  and "Top factors" breakdowns are now built dynamically per result
  instead of two hardcoded columns, the highest score gets a highlighted
  border, and the old single pairwise "delta" sentence became a
  `verdictSentence()` that ranks all of them and names the winner plus
  its gap to the runner-up. `/api/compare` changed shape to match (a
  `scenarios: [{request, label}, ...]` list, 2-4 of them, instead of
  hardcoded `a`/`b`/`label_a`/`label_b`) - a breaking change with no
  other consumer to worry about. The URL-sync/restore feature below
  generalized the same way: `syncURL()` writes every scenario that was
  actually in the last comparison (dropping a stale 3rd/4th scenario's
  params if the next comparison only has two), and `restoreFromURL()`
  reveals however many c/d columns a link specifies before resolving
  them.
- Made a finished Compare comparison bookmarkable and shareable: both
  scenarios' player/club/fee/age sync to the URL after a successful
  compare, and opening that link resolves everything and re-runs the
  comparison automatically. Needed a new `/api/players/{player_id}`
  endpoint (players_lookup.csv has no by-id lookup otherwise, only
  `/api/players/search`) - registered *after* the existing
  `/api/players/search` and `/api/players/career-search` routes
  specifically, since a bare `{player_id}` path segment matches any
  string and, registered first, would have swallowed every request meant
  for those two and 422'd trying to parse "search"/"career-search" as an
  int, breaking both search endpoints entirely. Also caught while testing
  by hand: resolving each comparison scenario's player+club and applying
  it to the form were the same step, so a stale link where only one
  scenario's id failed to resolve still populated the *other* one,
  leaving one column filled in normally and the other blank with no
  visible reason Compare never ran. Fixed by resolving both scenarios
  first and only applying either one once both are confirmed to exist.
- Added a "Where this lands" section to Predict, plotting the current
  prediction as a marker against the sitewide fee/age trend lines from a
  new, lighter `/api/analytics/trends` endpoint (just `fee_trend`/
  `age_trend`, factored out of `/api/analytics` so Predict doesn't have
  to ship that page's much larger scatter payload just for two small
  series). Also added a fee slider next to the fee number input,
  debounce-re-predicting live as it's dragged once a first prediction
  already exists - deliberately fee-only, not age too, since a player's
  age at a hypothetical transfer isn't something to explore a range of.
  Caught while testing it by hand: since age has no live re-predict, the
  age marker chart was reading the age field live at render time, so
  editing age without re-submitting and then triggering any re-render
  (e.g. a currency change) plotted the new, unsubmitted age against the
  *old* score - fixed by snapshotting the exact age/fee a prediction was
  computed from (`state.lastPredictInputs`) instead of re-reading the
  form.
- Synced Browse/Loans/Model vs Reality/Club Report Cards/League Trends'
  search/filter/sort/page state to the URL (via `history.replaceState`),
  so a filtered view is finally bookmarkable and shareable instead of
  always resetting on reload - see `readURLParams()`/`writeURLParams()`
  in `settings.js`.
- Removed the Analytics page's success-score-by-position box plot (down
  to four charts) and replaced every chart's hover-only `<title>`
  tooltip - the trend-line markers, the yearly points on the market-over-
  time chart - with a real click/tap-triggered tooltip (`#chart-tooltip`,
  positioned in JS near the click point via `showChartTooltip()` in
  `analytics.js`). `<title>` alone never fires on a touch device (there's
  no hover state to trigger it) and has a real delay even on desktop;
  `<title>` is kept alongside the new `data-tooltip` attribute as a free
  bonus for a patient mouse user, but it's no longer the thing relied on.
- Fixed every hand-drawn chart on the site (Player Timelines, League
  Trends, Analytics) rendering tiny and stranded in a large dead gap
  below a narrow viewport - each `<svg>` set `width="100%"` but a fixed
  pixel `height`, which only matches its `viewBox`'s aspect ratio at
  exactly the viewBox's own width; narrower than that, the browser scales
  the chart down to fit the width while the element's box keeps the old
  fixed height. `.timeline-chart-wrap svg { width: 100%; height: auto; }`
  in `style.css` fixes all of them from one place, no per-chart JS
  changes needed. Also made the Analytics page's three scatter charts
  (fee vs. score, age vs. score, fee vs. market value) clickable - each
  point now opens its full `/api/transfers/detail` card, the same as
  every other list page, via one delegated click listener per chart
  rather than one per point; a larger invisible circle on top of each
  small visible dot gives touch a real target to hit. And restyled the
  fee/age trend line from a flat `var(--text)` line (black in light
  theme, competing with the scoreColor'd dots for attention) to a dashed
  light-blue line (`--trend-line`) with small hoverable markers at each
  underlying bucket, reading as its own series rather than a slash across
  the chart.
- Added an Analytics page (`/analytics.html` + `/api/analytics`): five
  hand-drawn SVG charts (fee vs. score, age vs. score, score by position,
  fee vs. market value, market volume over time) in the same no-library
  inline-SVG style as Player Timelines' career chart and League Trends'
  index chart, rather than pulling in a charting dependency for one page.
  Caught along the way: `DataFrame.where(cond, None)` on a float64 column
  casts `None` straight back to `NaN` to preserve the column's dtype
  instead of promoting it to `object` - silently reintroducing the exact
  raw-NaN-leak bug a `/api/leagues/trends` test already guards against on
  a different endpoint. Fixed by converting each scatter column
  explicitly (`None if pd.isna(v) else float(v)`, per value) instead of
  trusting a whole-DataFrame `.where()`, and added the equivalent test for
  this endpoint.
- Renamed the "Biggest Surprises" page to "Model vs Reality" and added a
  "Most accurate predictions first" option to its Show filter (sorting by
  `abs_surprise_delta` - the new backend field is `|surprise_delta|`,
  computed once alongside the existing merge in `app/main.py`), since the
  old name undersold a page that can now also surface the model's closest
  calls, not just its biggest misses.
- Ran a third club-name sweep, this time matching on string similarity
  (`difflib`) instead of shared whole words, to catch the abbreviation/typo
  pairs the second sweep's method structurally couldn't ("Y.
  Malatyaspor"/"Yeni Malatyaspor" share no whole word; neither do "Aalesund"/
  "Aalesunds FK" once the trailing "s" is accounted for). Added 16 more
  verified `CLUB_NICKNAME_GROUPS` entries, one genuine data-entry
  inconsistency ("FC Helsingör"/"FC Helsingør" - two different Nordic
  letters for the same Danish club), and rejected "Atalanta"/"Atlanta"
  (Italy's Atalanta BC vs. MLS's Atlanta United - different leagues),
  "Metalurg D."/"Metalurg Z." (two different Ukrainian clubs sharing a
  single-top-flight country, so the usual league check can't tell them
  apart), and "Al-Wehda"/"Al-Wahda" (likely different Saudi/UAE clubs).
  Found a second Metalist-shaped case too: Belgian "Beerschot AC" only
  appears in 2013, the year the club went bankrupt - flagged to the user,
  who confirmed merging just the later "Beerschot VA"/"Beerschot V.A."
  spellings and leaving "Beerschot AC" split.
- Ran a second full club-name sweep after a user report that "Swansea" and
  "Swansea City" still showed as two rows on Club Report Cards. Checked
  every pair of names in the dataset sharing a whole word (not just the
  original ~103 clusters) and added ~35 more mechanical `CLUB_NAME_STRIP_TOKENS`
  (generic markers like "AS", "Calcio", "Spor Kulübü") and ~95 more
  `CLUB_NICKNAME_GROUPS` entries, each individually verified the same way as
  the original set. Also caught and rejected several look-alike traps the
  same way "Racing"/"Racing Club" was originally: "Barcelona"/"RCD Espanyol
  Barcelona" (two different clubs), "Rangers"/"Queens Park Rangers" (Glasgow
  Rangers, not QPR), "Krasnodar"/"Kuban Krasnodar" (two different clubs from
  the same city, the dissolved one marked with a trailing year), and seven
  club-vs-its-own-B/youth-team pairs, left split since a reserve side runs
  its own transfer history. One cluster - Ukrainian "Metalist Kharkiv",
  which has a real, still-contested ownership dispute after its 2016
  bankruptcy - was flagged to the user rather than guessed; only the
  confirmed-safe half of that cluster was merged.
- Made Club Report Cards' six highlights (best/worst signing, best/worst
  flip, best/worst departure) clickable - each one now drills into that
  specific transfer's own full breakdown card in the same modal, with a
  "← Back" link to return to the club view. Required adding `player_id` to
  `build_club_report_cards`' highlight dicts in `app/main.py`, since they
  previously only carried enough to describe themselves, not enough to
  look themselves up via `/api/transfers/detail`.
- Restructured the nav into three dropdowns (Predict, Browse, Insights)
  plus Home and About, and split `/` into a real home page (the curated
  showcase cards, moved here, plus a linked directory of every page) with
  the predict form moved to its own `/predict.html` - nine flat nav links
  had become too many for one row. `settings.js` gained the shared
  click-to-open/close dropdown behavior (`initNavDropdowns`) every page
  loads; `app.js` dropped the now-dead `loadExamples`/`renderBreakdown`
  it no longer needed once the showcase cards left the predict page.
- Added League Trends (`/leagues.html` + `/api/leagues/trends`): every
  major league's average fee and average success score, comparing its
  earliest against its most recent 3 complete years (its current
  in-progress year, checked directly to run far below a normal season's
  volume, is excluded from every average). Indexes both series to the same
  early-period baseline so a euro amount and a 0-100 score can share one
  honest chart. Real result across nearly every league: fees have grown
  30-320% while average success score has barely moved (often within a
  couple of points) - a genuine "spending is outpacing performance"
  pattern in this data, not an assumption going in.
- Added Player Timelines (`/player.html` + `/api/players/{id}/career` +
  `/api/players/career-search`): search any player to see their whole
  scored career as one chronological, real-date-scaled chart - permanent
  transfers as solid dots, loans as hollow rings, both colored by score.
  Added a dedicated search endpoint rather than reusing the Predict form's
  player autocomplete, since that one is filtered to `players_lookup.csv`'s
  market-value threshold and would have silently hidden ~59% of players
  with real scored history (checked directly) - mostly retired or
  lower-value players, exactly the kind of long, uneven career this page
  exists to show.
- Added Club Report Cards (`/clubs.html` + `/api/clubs/leaderboard`): every
  club ranked as a recruiter (incoming transfers, spend, buy-develop-resell
  profit, how departing players did elsewhere). Found and fixed a real bug
  along the way: an early version attributed a club's *buyer's* eventual
  resale profit to whichever club had sold it that player originally (e.g.
  crediting Real Madrid's sale of Ronaldo to Juventus with Juventus's own
  later resale of him) - resale profit belongs to the buyer
  (`to_club_name`), not the seller, since `next_transfer_fee` is what a
  third club later paid *that buyer*. Also added `build_club_name_aliases`
  to merge ~75 same-club name variants that differ only by a generic
  legal-entity marker or accent encoding ("FC Barcelona"/"Barcelona"), then
  extended it with `CLUB_NICKNAME_GROUPS` - ~28 hand-verified
  nickname/official-name pairs sharing no token at all ("Man
  City"/"Manchester City", "PSG"/"Paris Saint-Germain", "Bor.
  Dortmund"/"Borussia Dortmund"/"Dortmund") - each checked against the real
  data (matching league both sides, non-contradictory dates) rather than
  assumed, which is what confirmed "Sporting" is safe to fold into Sporting
  CP specifically rather than the ambiguous case ("Sporting" could also
  mean Sporting Gijón or Royal Charleroi) it looks like on name alone.
  `transfers_processed.csv` has no club_id, so without either fix, several
  major clubs' report cards were silently missing part of their real
  history.
- Moved club-name canonicalization to run once at startup on
  `transfers_df`/`loans_df`/`comparables["meta"]` directly (previously only
  applied inside Club Report Cards' own aggregation) - Browse, Loans,
  Biggest Surprises, and Predict/Compare's comparables now all show the
  same name for the same club too, not just `/clubs.html`. Also caught and
  force-split a false merge the mechanical pass would otherwise have made
  once loans data joined the count: "Racing" and the unrelated Argentine
  "Racing Club" share the generic word "club" but not a league.
- Changed the canonical-name pick from "whichever spelling has the most
  transfer mentions" to "whichever spelling is longest" ("Tottenham
  Hotspur" over "Tottenham", "Arsenal FC" over "Arsenal") - a reader
  unfamiliar with a shorthand is more likely to recognize the fuller name.
- Added billions formatting (`eur_m()`/`formatMoney()`, e.g. `"€2049m"` ->
  `"€2.05b"`) - no individual transfer fee reaches a billion, but a club's
  aggregate spend or resale-profit total on `/clubs.html` regularly does.
- Added a "Biggest Surprises" page (`/surprises.html` + `/api/surprises`):
  every transfer ranked by how far its real outcome diverged from a 5-fold
  cross-validated, held-out model prediction - the biggest overachievers
  and busts a pre-transfer-only model didn't see coming.
- Added keyboard navigation and ARIA roles to the player/club search
  autocomplete on Predict and Compare (previously mouse-only); fixed
  missing input IDs on Compare; moved the compare-delta conclusion below
  the evidence it's based on; blocked predicting a transfer to a
  player's own current club.
- Fixed the transfer-fee input not updating when currency changed, and
  silently mis-scoring a foreign-currency-typed fee as EUR.
- Extended accessibility work to Browse/Loans: keyboard-openable table
  rows, and a shared focus-trap/dialog helper reused across all three of
  the site's modals.
- Fixed a CSS Grid layout bug where selecting a player or club visually
  stretched the unrelated sibling field beside it.
