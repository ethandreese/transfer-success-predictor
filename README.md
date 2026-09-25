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

The raw Transfermarkt CSVs (~730MB) are **not** committed. They're
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
updated, or after changing the scoring formula). Not needed just to run
the app:

```bash
./.venv/bin/python -c "import kagglehub; kagglehub.dataset_download('davidcariboo/player-scores')"
./.venv/bin/python scripts/fetch_transfer_types.py           # optional but recommended - see below
./.venv/bin/python scripts/fetch_fotmob_stats.py              # optional but recommended - see below
./.venv/bin/python scripts/fetch_pretransfer_fotmob_stats.py  # optional but recommended - see below
./.venv/bin/python scripts/build_dataset.py
./.venv/bin/python scripts/fetch_fotmob_stats.py              # run again - see below
./.venv/bin/python scripts/fetch_pretransfer_fotmob_stats.py  # run again - see below
./.venv/bin/python scripts/build_dataset.py                   # run again to merge in the FotMob data just fetched for this batch's own new transfers
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/fetch_current_fotmob_stats.py      # optional but recommended - see below
./.venv/bin/python scripts/build_lookups.py                   # run again to merge in the current-FotMob snapshot just fetched
./.venv/bin/python scripts/train_model.py
./.venv/bin/python scripts/compute_prediction_surprises.py
```

`fetch_transfer_types.py`, `fetch_fotmob_stats.py`,
`fetch_pretransfer_fotmob_stats.py`, and `fetch_current_fotmob_stats.py`
are all optional and all resumable (each caches to its own
`data/raw/*.csv` or `data/raw/fotmob_season_cache/`, safe to interrupt
and rerun). Skipping any of them just degrades gracefully:
`build_dataset.py` still runs without loan detection
(`fetch_transfer_types.py`, ~23k requests - every zero-fee transfer,
including real loans, is then scored as a permanent transfer) or the
four FotMob score components (`fetch_fotmob_stats.py`, ~14 seasons × 23
leagues - each missing component's weight is dropped and renormalized
per transfer); `train_model.py`/`build_lookups.py` still run without the
pre-transfer FotMob composites or live-search autofill
(`fetch_pretransfer_fotmob_stats.py`/`fetch_current_fotmob_stats.py`,
falling back to the population median/`None`). Both pretransfer scripts
reuse `fetch_fotmob_stats.py`'s cache/matching directly, so they run in
seconds once that cache is warm.

`fetch_fotmob_stats.py`/`fetch_pretransfer_fotmob_stats.py` and
`build_dataset.py` are circularly dependent - the fetches need
`build_dataset.py`'s own output to know which transfers to fetch stats
for, so run once, a transfer that's brand-new this run gets no fetch
attempt and is built with imputed FotMob features instead. Running both
a second time (as above) closes that loop in one pass: pass 1 makes the
new transfer visible, pass 2 fetches its real stats and merges them in -
the same reason `build_lookups.py` runs twice, below.

`compute_prediction_surprises.py` is the one exception - **not
optional**, since `app/main.py` loads its output
(`data/prediction_surprises.csv`) unconditionally at startup for the
Model vs Reality page. It reuses `train_model.py`'s exact feature
engineering with 5-fold cross-validation instead of one temporal split,
so every transfer's prediction comes from a fit that never saw its own
outcome - unlike `model.joblib` itself, which is refit on the *full*
dataset for accurate live serving (and is therefore partly circular on
historical transfers, since it saw the answer). Runs in seconds; doesn't
touch `model.joblib`.

### Automated monthly refresh

[`.github/workflows/monthly-data-refresh.yml`](.github/workflows/monthly-data-refresh.yml)
runs the exact sequence above on a schedule (06:00 UTC on the 1st of
each month), then commits + pushes `data/` and `app/model/` if anything
changed and `pytest` passes first. A push to `main` triggers Render's
deploy (`autoDeployTrigger: commit` in `render.yaml`), so a clean run
reaches production with no extra step.

Each fetch script's cache is committed and id-keyed, so a routine month
only fetches the delta since last run, not the ~23k-request full rebuild
described above - that only happens once, from an empty cache. A failed
`pytest` stops the job before anything's committed; GitHub's default
failed-scheduled-run email is the only alerting here.

Needs two repository secrets (**Settings → Secrets and variables →
Actions**): `KAGGLE_USERNAME`/`KAGGLE_KEY`, from a Kaggle account's
**Settings → API → Create New Token**. Test on demand from the
**Actions** tab → "Monthly data refresh" → **Run workflow**, optionally
with a small `transfer_types_limit` for a quick run.

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
they're still there), not just year one. That way a slow starter who
adapted late and a hot starter who faded aren't both mis-scored by a
fixed first-year window.

Built from ten sub-metrics, each converted to a percentile rank (so no
single stat's raw scale dominates), then blended with **weights that vary
by position and sub-position** (`data/score_weights.json`):

- **Performance level & change**: goal contributions/90 at the new club,
  adjusted for how hard it is to score in that specific league, then
  ranked within the player's position. "Change" compares against a
  regression-based *expectation* given their pre-transfer level, not a
  raw before/after difference, since otherwise an already-elite player
  would have nowhere to go but "down" on paper even while staying elite.
  Folded together with the FotMob "attacking" bucket below when both
  exist, so the two don't double-count the same signal.
- **Market value growth**: an 80/20 blend of growth *relative to the
  player's own starting value* and the *absolute euro gain*, each itself
  weighted 80/20 toward the tenure's peak value over its end-of-tenure
  value (a long career's market value naturally declines with age by the
  time a player leaves, which shouldn't read as failure).
- **Playing time**: 60% percent of the team's actual games played
  (catches injuries/rotation a raw count hides), 40% raw appearance
  count, with a forgiveness curve for a normal amount of missed games
  before ranking (even clearly-elite players miss a median of ~12 games
  a season). Weighted highest for goalkeepers, where winning the single
  starting job is an unusually clean signal.
- **Value for money**: fee vs. a position-weighted blend of every
  on-pitch signal, judged against the player's own market value at
  signing rather than the whole dataset's fee distribution. A premium up
  to 1.3x pre-transfer value counts as normal and gets zero penalty; only
  real overpays are ranked, against each other. Its weight decays with
  tenure length: a long, clearly successful career is no longer judged
  much on whether the initial fee looked reasonable.
- **Resale profit**: did the buying club later resell the player for a
  profit? Only counted when a real subsequent sale exists (~25% of
  transfers); weight decays with tenure length, the same idea as value
  for money.
- **Four FotMob-derived components** (rating, attacking, defensive,
  possession): kept separate rather than blended into one number, so an
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
at 30% instead of the usual 20% (checked directly: `rating` correlates
with the final score far more than raw defensive-action counting does
for those roles specifically, since elite defenders often need to make
fewer visible defensive actions, not more).

**Loans are detected and scored separately**, not folded into the
permanent-transfer formula. The packaged dataset has no loan/permanent
flag at all: its ETL parses "loan transfer" and "€0" down to the same
flat fee. So loans are identified by re-fetching each transfer's real
fee text from Transfermarkt's own transfer-history API
(`scripts/fetch_transfer_types.py`) and scored on
`data/loan_score_weights.json` instead. That formula drops the
value-for-money and resale components (most loans carry no real fee and
don't end in a sale) and weights playing time much more heavily, since
whether the loan delivered game time is usually the central question
it's judged on. A loan spell with zero appearances is kept, not filtered
out, since that's a real (bad) outcome the Loans tab exists to surface.

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
output, and origin/destination club & league strength), with nothing
about what happened after the move, and training rows weighted by
recency (an exponential decay, 3-year half-life, favoring newer transfers
- see `RECENCY_HALF_LIFE_YEARS` in `scripts/train_model.py`). Evaluated on
a temporal holdout (trained on transfers before mid-2023, tested on
transfers since): **MAE ≈ 12.16 points** on the 0–100 scale, **R² ≈
0.217**, vs. ≈14.37 MAE for always predicting the average. That's a
modest but real signal: predicting a player's *entire future tenure* from
pre-transfer stats alone is genuinely hard, since multi-year outcomes
depend heavily on injuries, tactics, and squad fit no pre-transfer number
can see. (The model was originally a tuned `GradientBoostingRegressor`; a
later investigation found a plain `Ridge` beat it and every other
alternative tried. See [Project history](#project-history).)

**Explainability.** For a known historical transfer, the app shows the
real 5-10 component breakdown with concrete numbers behind each one
("0.92 goal contributions/90 at Barcelona, ranked vs. other attackers").
For a hypothetical prediction, it shows each feature's contribution via a
leave-one-out swap against a "typical transfer" reference. That
reference itself is contextual, not one flat number for the whole
dataset: position-conditional medians for stats that vary by role
(position/sub-position are swapped together, never independently, since
a sub-position can't exist under a different broad position), a
paid-vs-free-specific fee reference (over half of all transfers are
free, which would otherwise drag the "typical fee" to €0), a fee
expectation regression against the player's own market value, and short
explanatory notes wherever a bare "X vs. typical Y" swap would otherwise
read as an unsupported claim - e.g. a bigger destination club's swing
cites its own directly-checked link to post-move rating rather than
asserting "bigger is just better," and a position/league/foot gap states
the real historical difference without inventing a cause that isn't
actually confirmed.

## Pages

The nav groups every page under one of three dropdowns (**Predict**,
**Browse**, **Insights**) plus **Home** and **About** on their own. Nine
pages was too many flat links, so `/` itself became a real landing page
rather than doubling as the predict form (see Project history).

- **`/`**: the home page, with a short pitch, the curated showcase cards,
  and links to every other page grouped the same way as the nav. Each
  historical showcase card links to that player's Player Timeline (their
  whole scored career, not just this one move); each hypothetical card
  links to Predict, pre-filled with that exact player/club/fee so it
  lands on a live, editable version of the same prediction instead of a
  blank form.
- **`/predict.html`**: predict a hypothetical transfer. Search a real
  player, pick a destination club, and get a predicted score, a likely
  range (from the 5 most similar real transfers, since a single point
  estimate overstates how confident a model this size can be), a "why
  this score" breakdown, and the nearest historical comparables. A
  "Where this lands" chart plots the prediction against the sitewide
  fee-vs-score and age-vs-score trend lines, so the number isn't shown
  in isolation from how similar real transfers actually went. The fee
  field live-re-predicts as you type a new value, so trying a different
  fee doesn't need a re-click each time (see Known limitations for a
  real but easily misread pattern in how fee affects the score). A
  "What would move this most" section shows which
  player-improvable stats (recent scoring rate, the four FotMob
  composites) would raise the score most, and by how much, offering a
  scouting-style answer to what the player should get better at
  (distinct from the fee/club-value framing "why this score" uses).
- **`/compare.html`**: set up 2-4 hypothetical transfers side by side
  (same player to different clubs, or entirely different players) and
  see every prediction, range, and top factors together, with the
  highest score highlighted and a plain-English verdict ("X scores
  highest at Y, Z points ahead of the next best"). Starts with two
  columns; "+ Add another option" reveals a 3rd and 4th. A finished
  comparison syncs to the URL, so it's bookmarkable and shareable. A
  "Compare by factor" table below the score cards shows every scenario's
  full set of factors side by side, most-differentiating first, with the
  cell each factor favors most highlighted, showing not just which
  option wins but which specific factors carry it.
- **`/browse.html`**: every scored transfer (~8,300), filterable by
  position, destination league, fee range, and age range, searchable by
  player/club name, sortable by score/date/fee/age, and paginated. Click
  any row to open that transfer's full breakdown in a modal. A
  results-summary line above the table ("8,358 transfers, avg score
  49.1, €46.97b total spent") reflects the entire filtered set, not just
  the current page. Every filter/search/sort selection syncs to the URL,
  and an "Export these results as CSV" link always exports every
  matching row, not just the current page.
- **`/loans.html`**: every scored loan spell (~4,300), with the same
  browse, filter, search, and export experience as `/browse.html`, but
  filterable by loan duration instead of fee (since most loans carry no
  real fee) and scored on the loan-specific formula above. A loan whose
  player later signed permanently for the same club gets a "✓ Permanent"
  badge, linking to that conversion's date and score.
- **`/surprises.html`** ("Model vs Reality"): a predicted-vs-actual
  scatter chart with a dashed y=x reference line and dots colored by
  which side of it a transfer landed on, followed by every scored
  permanent transfer with a held-out prediction (~93% of them), ranked
  by how far the real outcome diverged from what a model that never saw
  that transfer's own result would have guessed from pre-transfer data
  alone. Shows the biggest overachievers, the biggest busts, or (via the
  Show filter) the calls it got closest to right, with the same filter,
  search, and click-to-view-card experience as Browse. A second
  collapsible chart, "Model accuracy over time," plots the held-out
  model's mean absolute error by transfer year, showing whether the
  model's calibration is actually improving as more training data
  accumulates.
- **`/clubs.html`**: every club that's bought or sold at least one
  scored permanent transfer, ranked as a recruiter by incoming transfers
  and their average score, total spent, buy-develop-resell profit on
  players it later resold, and how players it let go performed at their
  next club (a club needs at least 5 transfers, 3 for resale-profit
  ranking, before a rate-based ranking includes it). Click a club for
  its full report card: best/worst signing, best/worst flip, best/worst
  departure (each clickable into that transfer's own breakdown), a
  spend-vs-incoming-quality-by-year chart, and a recruiting-by-position
  breakdown, plus a link to its complete transfer history on Browse. A
  collapsible "Compare two clubs head-to-head" section puts any two
  clubs' stats, charts, and breakdowns side by side, with the
  unambiguous "higher is better" rows highlighting whichever club wins;
  the comparison is bookmarkable and shareable. A "Most improved
  recruiters first" sort ranks clubs by second-half-vs-first-half
  average incoming score, split by transfer count rather than a fixed
  year window so every club gets a fair comparison regardless of how
  many years its history spans.
- **`/player.html`**: search a player to see their whole scored career
  as a timeline, with one point per permanent transfer or loan,
  positioned by its real date and colored by score. A loan renders as a
  hollow ring instead of a solid dot, visually distinct without a second
  legend. Click any point (or its table row) for that stop's full
  breakdown card. Search covers every player with a scored transfer or
  loan, a larger set than the Predict page's autocomplete can see (see
  Known limitations). A "Compare with" search overlays a second player's
  career on the same chart, rendered as squares instead of circles, with
  a merged, date-sorted table below. "Similar career shape" suggests up
  to 5 other players whose career trajectory is closest to the one just
  searched; clicking a suggestion loads it straight into the comparison.
  A small star marks each shown career's single highest-scored stop.
- **`/leagues.html`**: every league with at least 15 scored permanent
  transfers, ranked by transfer volume, average score/fee, or the change
  between its earliest and most recent 3 complete years (the current,
  in-progress year is excluded from every average). Click a league for a
  chart indexing average fee and average success score to the same
  early-period baseline, plus a plain-language verdict answering "are
  fees inflating faster than performance?" directly. Also shows a
  cross-league flow breakdown (its top trading partners by transfer
  count, incoming and outgoing, excluding transfers within the league
  itself) and a position-mix breakdown of that league's incoming
  transfers.
- **`/analytics.html`**: five hand-drawn SVG charts over every scored
  permanent transfer. Fee vs. success score and age vs. success score
  are scatter plots with a binned trend line overlaid. Height vs.
  success score has a separate trend line per position, so a reader can
  see whether height matters more for some positions (e.g. goalkeepers)
  rather than one blended line; a "Split by position" dropdown isolates
  any one position's dots and trend line. There's also fee vs. market
  value (log-log scatter with a y=x reference line) and transfer count
  vs. average fee by year (both indexed to the first year = 100, so two
  differently-scaled series can share one axis). Every scatter chart is
  click/tap-interactive: a point opens that transfer's full breakdown
  card, and a trend-line or yearly point shows a tooltip with its exact
  numbers. A sixth section, "What actually predicts success," ranks the
  deployed model's own learned feature weights, the real answer to what
  the model keys off.
- **`/about.html`**: what the site does and how the numbers are
  computed, in plain language.

## Known limitations

- **FotMob coverage is real but partial.** ~67% of scored permanent
  transfers (~39% of loans) get at least one of the four FotMob
  components; coverage per bucket runs 54-67% even among those. Only the
  23 leagues in `LEAGUE_MAP` are covered at all, each from whatever
  season FotMob's own history starts for that league (2016/17 for the
  big five, as late as 2021/22 for Greece), and Ukraine's Premier League
  has no defensive/possession/rating data on FotMob at all. A transfer
  missing a bucket just has that component's weight dropped and the rest
  renormalized. It's never guessed at.
- **FotMob/Transfermarkt player and club matching is name-based** (the
  two sites share no id), which is inherently approximate. Extensively
  debugged for identity-collision bugs (see [Project history](#project-history)),
  but the least-scrutinized of the 23 leagues likely still has a
  somewhat weaker match rate than the handful that were checked by hand.
- **Only transfers with ≥10 appearances in both the year before and the
  whole tenure after are scored** (~8,358 of ~112k permanent-transfer
  candidates), which skews the dataset toward established first-team
  players over fringe moves. Loans use a looser bar: only the pre-loan
  side needs ≥10 appearances, since a loan with zero game time afterward
  is a real outcome worth showing, not missing data.
- **The packaged `transfers.csv` has real gaps** (a handful of famous
  moves, like Eden Hazard's 2019 Real Madrid transfer, were entirely
  missing). `data/manual_transfers.csv` is a hand-verified backfill,
  built by cross-checking `appearances.csv`'s club-history against
  `transfers.csv` and re-fetching real gaps from Transfermarkt's own
  API. See [Project history](#project-history) for the full trail.
- **Loan detection depends on a one-time scrape** of Transfermarkt's
  live, unofficial transfer-history API. About 98% of candidates match a
  fetched record; the rest default to "unknown" and are scored as a
  permanent transfer. A loan that converts to a permanent deal at the
  same club, with no separate recorded event for the conversion, still
  scores as one continuous loan rather than splitting at the conversion
  point.
- **Sub-position weighting uses a player's single most-common fielded
  position across their entire career**, not the specific window of any
  one transfer. A player who genuinely changed roles mid-career has
  every transfer weighted by whichever role dominates their overall
  history.
- **Predicting a new hypothetical transfer is meaningfully less
  reliable** than the historical scores shown for known transfers, since
  the model only ever sees pre-transfer information by construction.
- **A higher fee nudges the predicted score up slightly**, which can
  read as "the model rewards overpaying" - it doesn't, straightforwardly.
  The pattern comes from `fee_to_value_ratio` (fee relative to the
  player's own market value), and a per-value-tier regression on the
  real training data shows it's only genuinely established for
  expensive, high-profile signings: for a typical-value transfer, the
  95% confidence interval on the effect crosses zero. An explicit
  interaction term to correct this directly was tested and made holdout
  accuracy slightly worse on every one of 5 temporal splits, so it
  wasn't shipped. Previously called out on the Predict page itself with
  a caveat under the fee field; removed in favor of documenting it here
  instead (see Project history).
- **"Why this score"'s leave-one-out swap (`explain_prediction`) can
  still show a contribution size that doesn't match a feature's raw
  historical average gap, for reasons that aren't a simple reference-
  mismatch bug.** After fixing `position`/`sub_position`'s reference
  (see Project history), destination league in particular can still
  show a swing (e.g. +7.6 for a Laliga-vs-Premier-League swap on one
  real prediction) much larger than the ~0.1-point gap between those
  leagues' historical average success_score. Checked directly whether
  this traces to `fee_to_value_ratio` (a theory `league_context_note`'s
  own explanatory text used to offer, since removed - see Project
  history) by swapping it jointly with league instead of independently -
  the swing barely moved (7.6 -> 7.8), ruling that out as the driver.
  This is a known, inherent limitation of a single-feature leave-one-out
  swap against a linear model with correlated inputs (a real SHAP-style
  explanation would marginalize more carefully) rather than a specific
  bug with an identified fix - unlike position/sub_position, no equally
  clean joint-swap correction was found for this one.
- **The Model vs Reality page's `predicted_score` isn't the deployed
  model.** It comes from 5-fold cross-validation (see
  `compute_prediction_surprises.py`), so every transfer's number is
  honest in the sense that matters for ranking "how surprising was this"
  (no prediction ever saw its own outcome). Each fold is a slightly
  different fit than `model.joblib`, which is refit on the *full*
  dataset for serving live predictions, so don't expect it to match a
  live `/api/predict` call for the same inputs exactly.
- **Every club name on the site is identified purely by its name
  string**, since `transfers_processed.csv`/`loans_processed.csv` have
  no club_id and never join `clubs_lookup.csv` cleanly (its naming, e.g.
  "Manchester City", doesn't always match the raw Transfermarkt names,
  e.g. "Man City"). `build_club_name_aliases` merges same-club spellings,
  both mechanically (generic legal-entity markers or accent differences,
  e.g. "FC Barcelona"/"Barcelona") and via a curated
  `CLUB_NICKNAME_GROUPS` list of nickname/official-name pairs that share
  no token at all (e.g. "PSG"/"Paris Saint-Germain", "Man
  City"/"Manchester City"). Each entry is individually verified against
  the real data rather than assumed from football knowledge alone, which
  also means genuinely ambiguous or contested cases (e.g. two unrelated
  clubs both called "Racing", or Ukraine's disputed "Metalist Kharkiv")
  are handled deliberately rather than merged automatically. Applied
  once at startup, so every page shows the same name for the same club,
  but the lists aren't exhaustive across all ~700+ club names: a
  spelling pair that was never spot-checked can still show up as two or
  more separate identities. See [Project history](#project-history) for
  the full investigation.
- **`players_lookup.csv` covers far fewer players than have real scored
  history.** It's filtered to a market-value threshold for the Predict
  form's autofill, which excludes ~59% of every player with a scored
  transfer or loan, mostly retired or lower-value players.
  `/player.html`'s search works around this by searching the
  transfer/loan data directly instead, but a career page for one of
  those players still can't show a position or current club in its
  header, since both come only from `players_lookup.csv`.
- **League Trends buckets by calendar year, not by season**, and only
  ever compares the destination league's own earliest and most recent
  3-year windows against each other. It doesn't control for anything
  else that changed across a league's whole football economy over a
  decade (transfer windows shifting, financial fair play rules, a
  different mix of buying clubs), so "fees grew faster than success" is
  a real, checked pattern in this data, not a claim about *why*. Only 14
  of the dataset's 23 leagues have enough transfers (≥15) to appear at
  all, and only those 14 happen to also have enough year-span (≥6
  complete years) for a trend. That's true for every league today, but
  the code still handles a future league that clears the first bar
  without the second by showing its basic stats with no chart, rather
  than disappearing it outright.
- **League-adjustment feeds the historical label, not the predict
  model's own features.** It was tested directly as an additional model
  input (league baselines, a league-adjusted performance number), but it
  didn't beat the simpler feature set on the holdout. It's kept for
  explanation display only (`data/league_baselines.csv`), not fed to the
  model itself.

## Project history

A condensed changelog of notable fixes and investigations, newest first
within each group. Full reasoning and numbers for anything here are in
the git history.

**Predict model & backend**
- Audited every user-facing description on the site for correctness;
  fixed `league_context_note`'s unsupported fee-premium claim (r=+0.21,
  p=0.47 across leagues, not significant) - rest checked out clean.
- Investigated a user report that a higher fee nudges the predicted
  score up (confirmed real - see Known limitations for the full finding);
  added an on-page caveat under Predict's fee slider, then removed both
  the slider and the caveat in favor of documenting it here instead.
- Fixed `position`/`sub_position` comparing every prediction against the
  wrong reference group in "why this score" (e.g. -13.7 for a real
  Left-Back vs. an actual ~1-point gap) - now swapped jointly against a
  position-conditional reference, matching the real gap (-0.2/-0.6).
- Added a visible baseline score and "N other factors combined" total to
  "why this score," since only the top 5 of ~20 factors were shown with
  no way to see them add up to the displayed score.
- Weighted predict-model training rows by recency (3-year half-life) -
  MAE 12.26→12.16, R² 0.211→0.217 across 5 temporal splits; also tested
  (and rejected) imputing missing fee-ratio rows and ElasticNet/Lasso.
- Fixed `/api/predict` 400ing whenever a composite FotMob feature had
  partial sub-stat data - the common case (~54-67% coverage), not an
  edge case.
- Switched the predict model from `GradientBoostingRegressor` to
  `Ridge`, which won on every one of 5 temporal splits (MAE
  12.40→12.26, R² 0.185→0.211); fixed thin-league overfitting via
  `OneHotEncoder(min_frequency=30)`.
- Investigated several unsuccessful model/feature ideas: interaction
  terms, a quadratic age term, polynomial features, robust-loss
  regression, nonlinear alternatives (KernelRidge, SVR), manager tenure,
  squad age/nationality mix, a newer Kaggle dataset version, Wikipedia
  pageviews, and injury history - none beat the plain model or were
  worth shipping.
- Found `fee_to_value_ratio` was silently computed from a column missing
  for 38.7% of transfers; fixing it recovered 41% of training data (MAE
  12.65→12.40, R² 0.159→0.185).
- Investigated a more historically-accurate per-transfer-date club-value
  feature - it measurably hurt the model despite being more accurate, so
  the simpler flat proxy was kept.
- Fixed the loan pipeline's early filter referencing the wrong
  (permanent-transfer) position-weight keys.
- Fixed an unbounded `limit` query param on player/club search, and
  deduplicated a club-value-proxy computation that had drifted into two
  copies.

**Data pipeline & scoring formula**
- Backfilled thousands of missing transfers (`data/manual_transfers.csv`)
  after discovering real gaps in the packaged `transfers.csv` (starting
  from Eden Hazard's missing 2019 Real Madrid move). Current counts:
  8,358 permanent transfers, 4,306 loans.
- Found and fixed several FotMob player/club identity-matching bugs
  (fuzzy-match collisions, ambiguous shared club-name words, an
  asymmetric similarity-ratio bug).
- Fixed FotMob season-stitching contamination at tenure boundaries (a
  mid-season arrival/departure could leak a different club's season
  into the aggregate).
- Fixed Norway/Sweden silently returning the wrong FotMob season
  (single-year vs. split-year label mismatch).
- Fixed a home/away misattribution bug in `appearances.csv` affecting
  some pre-transfer appearance counts.
- Fixed mangled league display names (title-cased URL slugs) across 18
  competitions.
- Corrected two inaccurate claims on `about.html` (a missing
  `resale_profit` mention; an overstated symmetric playing-time rule
  that doesn't apply to loans).

**Frontend**
- Made the homepage's showcase cards clickable: historical cards link to
  that player's Player Timeline, hypothetical cards link to Predict
  pre-filled with the same player/club/fee. Needed new URL-restore
  support on both pages (`?player_id=X` on Player Timeline,
  `?player_id=X&club_id=Y&fee=Z` on Predict, auto-running the
  prediction), the same idea Compare's shareable-link restore already
  used; refactored Predict's inline player/club select handlers into
  named functions so both the autocomplete and the URL-restore path
  share one code path instead of duplicating it.
- Abbreviated a few long league names in League Trends' cross-league
  flow list (full name still available on hover).
- Added eight features, one per page, from a second brainstorm pass:
  Predict's "What would move this most," Compare's "Compare by factor"
  table, Browse/Loans' results-summary line + sortable headers, Model vs
  Reality's accuracy-over-time chart, Club Report Cards' "Most improved
  recruiters" sort, League Trends' position-mix breakdown, Analytics'
  "What actually predicts success" feature-weight ranking, and Player
  Timelines' highest-scored-stop star.
- Added a round of single-page features: an Analytics height-vs-score
  chart split by position; Player Timelines' two-player overlay
  comparison and nearest-neighbor "similar career shape" suggestions; a
  League Trends cross-league flow breakdown; three Club Report Cards
  additions (head-to-head comparison, spend-vs-quality-by-year chart,
  recruiting-by-position breakdown); and a Model vs Reality
  predicted-vs-actual scatter chart colored by which side of y=x a
  transfer landed on.
- Added fee/age range filters and CSV export to Browse/Loans, plus a
  "✓ Permanent" badge for loans that later converted to a permanent
  transfer.
- Extended Compare from a fixed A/B pair to 2-4 scenarios with a ranked
  plain-English verdict, and made a finished comparison bookmarkable/
  shareable via URL sync.
- Added Predict's "Where this lands" section (prediction plotted against
  sitewide fee/age trend lines) and a live-updating fee slider.
- Synced Browse/Loans/Model vs Reality/Club Report Cards/League Trends'
  filter state to the URL.
- Removed Analytics' box plot and replaced hover-only chart tooltips
  with click/tap tooltips (hover never fires on touch devices).
- Fixed every hand-drawn chart rendering tiny on narrow viewports (a
  fixed pixel height fighting a responsive width); made Analytics'
  scatter charts clickable into a transfer's full breakdown.
- Added the Analytics page (5 hand-drawn SVG charts, no charting
  library).
- Renamed "Biggest Surprises" to "Model vs Reality," added a "most
  accurate" sort option.
- Ran two further club-name matching sweeps (string-similarity, then a
  broader shared-word pass) beyond the original alias list, adding
  ~110 more verified entries total and rejecting several look-alike
  traps along the way (e.g. "Barcelona"/"RCD Espanyol Barcelona," two
  different clubs).
- Made Club Report Cards' best/worst highlights clickable into their own
  transfer breakdown card.
- Restructured the nav into three dropdowns plus Home/About; split `/`
  into a real home page with Predict moved to its own URL.
- Added League Trends (`/leagues.html`, fee vs. success trend by league)
  and Player Timelines (`/player.html`, career-as-timeline chart).
- Added Club Report Cards (`/clubs.html`, clubs ranked as recruiters,
  including the original `build_club_name_aliases`/
  `CLUB_NICKNAME_GROUPS` alias-merging system); fixed resale profit
  being credited to the wrong (selling, not buying) club.
- Moved club-name canonicalization to run once at startup, so every
  page shows consistent names, not just Club Report Cards.
- Changed the canonical club-name pick from "most mentions" to "longest
  spelling" (e.g. "Tottenham Hotspur" over "Tottenham").
- Added billions formatting for large euro amounts (club spend/resale
  totals).
- Added the original "Biggest Surprises" page (predicted-vs-actual
  ranking).
- Added keyboard navigation and ARIA roles to the Predict/Compare
  autocomplete; blocked predicting a transfer to a player's own current
  club.
- Fixed the transfer-fee input not updating on currency change, and
  silently mis-scoring a foreign-currency-typed fee as EUR.
- Extended accessibility work to Browse/Loans (keyboard-openable rows,
  shared modal focus-trap).
- Fixed a CSS Grid bug where selecting a player/club visually stretched
  the sibling field beside it.
