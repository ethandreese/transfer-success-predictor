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
scripts/compute_prediction_surprises.py  # 5-fold held-out predictions for every transfer -> data/prediction_surprises.csv (the Biggest Surprises page)
scripts/build_lookups.py   # small player/club search tables for the web app
app/main.py                 # FastAPI backend (serves the API + the static frontend)
app/static/                 # vanilla HTML/CSS/JS frontend (index/browse/loans/compare/surprises/clubs)
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
startup (the Biggest Surprises page's data source). It reuses
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
held-out-only filtering, and `/api/clubs/leaderboard`'s minimum-sample
thresholds and name-alias merging).

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

- **`/`** — predict a hypothetical transfer: search a real player, pick a
  destination club, see a predicted score, a likely range (min/max among
  the 5 most similar real transfers, since a single point estimate
  overstates how confident a model this size can be), a "why this score"
  breakdown, and the nearest historical comparables.
- **`/browse.html`** — every scored transfer (~8,300), filterable by
  position and destination league, searchable by player/club name, sortable
  by score/date/fee/age, paginated. Click any row to open that transfer's
  full card (score + breakdown) in a modal.
- **`/loans.html`** — every scored loan spell (~4,300), same browse/filter/
  search/click-to-view-card experience as `/browse.html`, but scored on the
  loan-specific formula above (no fee/resale rows in the breakdown, and
  duration shown in months rather than years).
- **`/compare.html`** — set up two hypothetical transfers side by side
  (same player to two different clubs, or two different players entirely)
  and see both predictions, ranges, and top factors together with the
  point gap between them.
- **`/surprises.html`** — every scored permanent transfer with a held-out
  prediction (~93% of them - see `compute_prediction_surprises.py`), ranked
  by how far the real outcome diverged from what a model that never saw
  that transfer's own result would have guessed from pre-transfer data
  alone: the biggest overachievers and the biggest busts. Same filter/
  search/click-to-view-card experience as Browse, plus the model's own
  number shown alongside the real one.
- **`/clubs.html`** — every club that's bought or sold at least one scored
  permanent transfer, ranked as a recruiter: incoming transfers and their
  average score, total spent, buy-develop-resell profit on the subset it
  later sold on again, and how players it let go performed at their *next*
  club. A club needs at least 5 transfers (3 for resale-profit ranking)
  before a rate-based ranking includes it. Click a club for its full report
  card (best/worst signing, best/worst flip, best/worst departure) and a
  link to its complete transfer history on Browse.

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
- **The Biggest Surprises page's `predicted_score` isn't the deployed
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
  Barcelona"/"Barcelona", "Fenerbahçe"/"Fenerbahce" - ~75+ clusters,
  hand-checked one by one before shipping, across both transfers and loans);
  and via `CLUB_NICKNAME_GROUPS`, a curated list of nickname/official-name
  pairs that share no token at all ("Man City"/"Manchester City",
  "PSG"/"Paris Saint-Germain", "Bor. Dortmund"/"Borussia
  Dortmund"/"Dortmund", "Sporting"/"Sporting CP", and ~25 more) - each entry
  individually verified against the real data (matching
  `domestic_competition_id` on both sides, a non-contradictory
  `transfer_date` range) rather than assumed from football knowledge alone,
  which is exactly what caught that "Sporting" alone is safe to fold into
  Sporting CP specifically (every row carries league `PO1`, never Sporting
  Gijón's `ES1` or Royal Charleroi's `BE1`) rather than the genuinely
  ambiguous case it looks like at a glance - and, in the other direction,
  that a mechanical strip of the generic word "club" would have wrongly
  merged "Racing" with the unrelated Argentine "Racing Club" (`ARG1`),
  force-split instead (`CLUB_NAME_FORCE_SPLIT`). Applied once at startup to
  `to_club_name`/`from_club_name` in both `transfers_df` and `loans_df`
  (and to `comparables["meta"]`, the nearest-neighbors index behind
  Predict's "similar historical transfers"), so every page - Browse, Loans,
  Biggest Surprises, Predict, Compare, and Club Report Cards - shows the
  same name for the same club. Neither list is exhaustive across all ~700+
  club names here - a club whose nickname/official-name split was never
  spot-checked still shows up as two or more separate identities everywhere
  it appears, each missing part of the real history.
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
