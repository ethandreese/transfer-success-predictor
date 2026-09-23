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
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/fetch_current_fotmob_stats.py      # optional but recommended - see below
./.venv/bin/python scripts/build_lookups.py                   # run again to merge in the current-FotMob snapshot just fetched
./.venv/bin/python scripts/train_model.py
./.venv/bin/python scripts/compute_prediction_surprises.py
```

`fetch_transfer_types.py` re-fetches every candidate player's real transfer
history from transfermarkt's live API to tell loans apart from free
transfers (one request per player, ~23k, politely rate-limited). It takes
a few hours and is safe to interrupt and re-run, resuming from
`data/raw/transfer_types_cache.csv` rather than starting over. It's
optional: skip it and `build_dataset.py` still runs, just without loan
detection, so every zero-fee transfer (including loans) is treated as a
permanent transfer, and `data/loans_processed.csv` comes out empty.

`fetch_fotmob_stats.py` pulls FotMob's season stat leaderboards for the 23
leagues in `LEAGUE_MAP` and stitches each tenure (both permanent transfers
and loan spells) into `data/raw/fotmob_stats_cache.csv` in one pass. A few
thousand requests across ~14 seasons x 23 leagues, politely rate-limited,
resumable from `data/raw/fotmob_season_cache/` (gitignored) rather than
refetching every league-season from scratch. Also optional: skip it and
`build_dataset.py` still runs, just without the four FotMob components.
Each one's weight is then dropped and the other weights renormalized for
every transfer, the same as when resale data is unknown.

`fetch_pretransfer_fotmob_stats.py` and `fetch_current_fotmob_stats.py`
feed the *predict model* (not the historical score). Both reuse
`fetch_fotmob_stats.py`'s season-fetch/cache and matching functions
directly (imported, not duplicated), so if that cache is already warm
these run in seconds, not hours. Both are optional the same way: skip
either and `train_model.py`/`build_lookups.py` still run, just with every
`pre_fotmob_*` feature (or every searched player's `recent_fotmob_*`
autofill) falling back to the median/`None` a missing match already
degrades to.

`compute_prediction_surprises.py` is **not** optional like the four above:
`app/main.py` loads `data/prediction_surprises.csv` unconditionally at
startup as the Model vs Reality page's data source. It reuses
`train_model.py`'s exact feature-engineering functions and feature lists,
swapping 5-fold cross-validation in for that script's one temporal split,
so every transfer gets a `predicted_score` from a model that never saw
that transfer's own outcome during fitting. That's unlike `model.joblib`
itself, which is refit on the full dataset for serving accurate live
predictions, at the cost of its own predictions on historical transfers
being partly circular (it saw the answer). Runs in seconds; doesn't touch
`model.joblib`.

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
dataset: position-conditional medians for stats that vary by role, a
paid-vs-free-specific fee reference (over half of all transfers are
free, which would otherwise drag the "typical fee" to €0), a fee
expectation regression against the player's own market value, and short
explanatory notes wherever a bare "X vs. typical Y" swap would otherwise
read as a claim the data doesn't actually support (e.g. the Premier
League's lower average score is almost entirely a fee-premium effect,
not an on-pitch one, and the note says so explicitly).

## Pages

The nav groups every page under one of three dropdowns (**Predict**,
**Browse**, **Insights**) plus **Home** and **About** on their own. Nine
pages was too many flat links, so `/` itself became a real landing page
rather than doubling as the predict form (see Project history).

- **`/`**: the home page, with a short pitch, the curated showcase cards,
  and links to every other page grouped the same way as the nav.
- **`/predict.html`**: predict a hypothetical transfer. Search a real
  player, pick a destination club, and get a predicted score, a likely
  range (from the 5 most similar real transfers, since a single point
  estimate overstates how confident a model this size can be), a "why
  this score" breakdown, and the nearest historical comparables. A
  "Where this lands" chart plots the prediction against the sitewide
  fee-vs-score and age-vs-score trend lines, so the number isn't shown
  in isolation from how similar real transfers actually went. The fee
  field has a slider that live-updates the prediction as you drag it, so
  trying a range of fees doesn't need a re-click each time; age has no
  slider, since exploring a range of ages isn't as natural a question. A
  "What would move this most" section shows which player-improvable
  stats (recent scoring rate, the four FotMob composites) would raise
  the score most, and by how much, offering a scouting-style answer to
  what the player should get better at (distinct from the
  fee/club-value framing "why this score" uses).
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
- **"Why this score"'s leave-one-out swap (`explain_prediction`) can
  still show a contribution size that doesn't match a feature's raw
  historical average gap, for reasons that aren't a simple reference-
  mismatch bug.** After fixing `position`/`sub_position`'s reference
  (see Project history), destination league in particular can still
  show a swing (e.g. +7.6 for a Laliga-vs-Premier-League swap on one
  real prediction) much larger than the ~0.1-point gap between those
  leagues' historical average success_score. Checked directly whether
  this traces to `fee_to_value_ratio` (the theory `league_context_note`'s
  own explanatory text offers) by swapping it jointly with league
  instead of independently - the swing barely moved (7.6 -> 7.8), ruling
  that out as the driver. This is a known, inherent limitation of a
  single-feature leave-one-out swap against a linear model with
  correlated inputs (a real SHAP-style explanation would marginalize
  more carefully) rather than a specific bug with an identified fix -
  unlike position/sub_position, no equally clean joint-swap correction
  was found for this one.
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
- Fixed `sub_position` and `position` comparing every hypothetical
  prediction against the wrong reference group entirely in "why this
  score" (`explain_prediction`'s leave-one-out swap, `app/main.py`).
  `sub_position`'s reference was the dataset-wide modal sub-position
  across ALL positions (`Centre-Forward`), regardless of the player's
  own broad position; `position`'s own swap changed only `position`
  while leaving `sub_position` at its real value. Both produced
  internally-contradictory synthetic rows the model still happily
  scored - e.g. a real Left-Back's stats compared under a
  `Centre-Forward` role label, or a real Goalkeeper's stats compared
  under a `Defender` position with `sub_position` still `Goalkeeper`.
  Caught from a real user report (Lewis Hall, a Left-Back, showing -13.7
  for "Specific role" on a hypothetical Manchester City move) and
  confirmed on a second real example (a real Goalkeeper showing -6.5 for
  "Position" alone) - both swings were an order of magnitude larger than
  the real historical average gap between the two groups being compared
  (13.7 vs. an actual 1.1-point Left-Back/Centre-Forward gap; 6.5 vs. an
  actual 1.2-point Goalkeeper/Defender gap), which is what flagged them
  as wrong rather than merely surprising. Fixed by giving `sub_position`
  a position-conditional reference (`sub_position_reference_by_position`
  in `train_model.py`, the modal sub-position *within* the player's own
  broad position) and swapping `position`+`sub_position` jointly and
  consistently - the same joint-swap pattern `PLAYING_TIME_FEATURES`
  already used for exactly this "structurally-dependent features can't
  be swapped independently" reason. Both examples above dropped to -0.2
  and -0.6 respectively, in line with the real historical gaps. Had to
  guard one edge case: when the reference position already equals a
  player's own real position (true for most Defenders, the flat modal
  position), `sub_position` must stay untouched too, or the fix would
  silently double-count with `sub_position`'s own separate swap. Audited
  every other candidate correlated-feature pair the same way
  (independent swap vs. joint swap, compare the delta): destination
  league/fee-to-value ratio, club-quality ratio/club values, and
  goals/goal-contributions per 90 all showed no comparable issue (see
  Known limitations for one residual, unresolved observation from that
  audit) - position/sub_position was uniquely bad because it's a true
  nested categorical hierarchy (a sub-position cannot exist under a
  different broad position at all), not just a soft numeric correlation.
- Added a visible "starting point" (baseline score - the model's own
  prediction for an entirely typical transfer, not a flat 50) and an "N
  other factors combined" total to `explain_prediction`'s response and
  the Predict page. Previously only the top 5 of ~20 computed factors
  were ever shown, with no way to see the other 15 or the implicit
  non-50 starting point, so the displayed numbers could never be summed
  to reconcile to the shown score - caught from the same user report
  above (Lewis Hall's score of 64 didn't match 50 plus the 5 shown
  factors).
- Weighted training rows by recency (`compute_recency_weight` /
  `RECENCY_HALF_LIFE_YEARS` in `scripts/train_model.py`, an exponential
  decay with a 3-year half-life, applied to both the deployed model's fit
  and `compute_prediction_surprises.py`'s 5-fold CV), after a position-
  bias investigation into the historical score's own formula prompted a
  broader look at the predict model for other checkable improvements.
  Tested three ideas empirically rather than by inspection alone:
  imputing (instead of dropping) the ~550 training rows missing
  `fee_to_value_ratio`/`height_in_cm`/club-value data made no real
  difference (MAE +0.016, R² -0.002) despite recovering ~480 rows, so was
  not kept; ElasticNet/Lasso lost to plain Ridge on every one of 25
  alpha/l1_ratio combinations tried, confirming the feature set's signal
  really is close to linear; recency weighting won, checked across the
  same 5 temporal splits the original Ridge-vs-GBR decision used - MAE
  improved ~0.1 and R² ~0.005-0.011 on 4 of 5 splits, and was never worse
  (an ~0.001 R² wash on the 5th). Moved the temporal holdout from MAE
  12.26/R² 0.211 to MAE 12.16/R² 0.217.
- Fixed `/api/predict` (and everything built on it, including Compare
  and the examples cards) 400ing whenever a composite FotMob feature
  (e.g. "defensive") had real data overall but one specific raw sub-stat
  was individually missing (e.g. an attacker with real
  tackle/interception/recovery numbers but no recorded clearances at
  all, since attackers rarely attempt any). `explain_prediction`'s
  bulleted-breakdown builder tried to number-format that one `None`
  straight into the string and crashed; `describe_fotmob_component()`
  (the historical-score version of the same breakdown) already handled
  this correctly by only listing sub-stats that are actually present, so
  the pre-transfer version now does the same. Not a rare edge case:
  partial FotMob bucket coverage (~54-67%, per Known limitations) is the
  common case, not the exception. Caught testing 3-way Compare by hand.
- Switched the predict model from `GradientBoostingRegressor` to a plain
  `Ridge` regression after benchmarking it against every tree-based
  alternative tried (HistGradientBoosting, RandomForest, ExtraTrees, and
  GBR variants). Ridge won on every one of 5 different temporal splits:
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
  38.7% of transfers, silently excluding them from model training.
  Fixing it recovered 41% of usable training data (MAE 12.65→12.40, R²
  0.159→0.185).
- Investigated a more historically-accurate, per-transfer-date club-value
  feature (reconstructed from contemporaneous roster valuations); it
  measurably hurt the model despite being more accurate, so the simpler
  flat proxy was kept.
- Fixed the loan pipeline's early filter referencing the wrong
  (permanent-transfer) position-weight keys. Harmless today only by
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
- Fixed FotMob season-stitching contamination at both ends of a tenure:
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
- Abbreviated a handful of long league names ("Bundesliga (Germany)" ->
  "Bundesliga (Ger.)", "Scottish Premiership" -> "Scottish Prem.") in
  League Trends' cross-league flow list. Checked directly that these
  were the two actually wrapping onto a second line in the list's fixed
  7.5rem label column (`LEAGUE_SHORT_NAMES` in `leagues.js`); the full
  name is still available via a `title` hover tooltip on the abbreviated
  label.
- Added eight features from a second brainstorm pass, one per page
  (Predict, Compare, Browse/Loans, Model vs Reality, Club Report Cards,
  League Trends, Analytics, Player Timelines):
  - Predict: "What would move this most" (`sensitivity_analysis` in
    `app/main.py`): swaps a handful of genuinely player-improvable
    stats to a better value and reports the gain, the inverse question
    from the existing "why this score" (typical-value comparison).
  - Compare: a "Compare by factor" table showing every scenario's full
    factor set side by side (`/api/predict` gained an optional `top_k`
    param, defaulting to 5 for Predict's own display; `compare()` passes
    25 so no two scenarios' explanations can fail to cover the same
    factors), ranked by cross-scenario spread, winning cell highlighted.
  - Browse/Loans: a results-summary line reading `/api/transfers`'/
    `/api/loans`' new `summary` field (avg score + total spent/avg
    duration, computed over the full filtered set server-side before
    pagination) and click-to-sort table headers, driving the same
    sort-select value the dropdown already did so the two can never
    drift apart. Added the previously-missing reverse-direction sort
    options (`transfer_fee:asc`, `age_at_transfer:desc`, `tenure_days:*`)
    so every sortable column has both directions available either way.
  - Model vs Reality: a second collapsible chart, "Model accuracy over
    time" (mean absolute surprise per year, from `/api/surprises/scatter`'s
    new `accuracy_by_year`). Required porting the click/tap chart-
    tooltip pattern (`showChartTooltip`/`#chart-tooltip`) into
    `surprises.js`, which hadn't needed it before this chart.
  - Club Report Cards: a "Most improved recruiters first" sort
    (`score_improvement`, the second-half-vs-first-half average incoming
    score, split by transfer count rather than a fixed year window so
    it works regardless of how many years a club's own history spans),
    surfaced as a "Recruiting trend" line in the club's own report card.
  - League Trends: a position-mix breakdown of each league's incoming
    transfers, reusing the same `.breakdown-row` bar markup as the
    cross-league flow lists right above it.
  - Analytics: "What actually predicts success", the *deployed model's*
    own learned feature weights (`FEATURE_IMPORTANCE`, computed once at
    import from `pipeline.named_steps["model"].coef_`, not a fresh
    analysis), reusing Predict/Compare's `.explain-row` bar markup.
  - Player Timelines: a star marking each shown career's single highest-
    scored stop, shown for both careers at once in a comparison.
- Added a new Analytics chart: height vs. success score, split into four
  per-position trend lines via `height_trend_by_position`, rather than
  one sitewide line that would hide whether the relationship actually
  differs by role. `trendLinePath()` in `analytics.js` gained an
  overridable `color` param so the four lines could each get their own
  hue without duplicating the whole function; four new `--pos-*` CSS
  variables back those colors, deliberately not reusing
  `--accent`/`--accent-mid`/`--accent-bad` since the scatter dots
  underneath already use that triad for score: a position line in one
  of those same colors would read as a score claim instead of a position
  label. A "Split by position" dropdown re-renders the chart scoped to
  just one position's dots and trend line at a time (or all four
  overlaid, the default), toggling the corresponding legend entries.
  `buildHeightScoreChart()` itself doesn't know or care whether it's
  been handed all four positions' rows/trends or one, so no separate
  code path was needed for the filtered view. Also tried a foot vs.
  success score chart (a plain bar breakdown, right/left/both); dropped
  it after building it, since the real spread between the three was
  under a point, not worth a permanent chart. Skipped the third
  Analytics idea from the original brainstorm (a finer sub-position
  breakdown) per explicit direction.
- Added a two-player overlay comparison and "similar career shape"
  suggestions to Player Timelines. A new career-shape feature vector
  (n_stops, first/last/avg score, score_range, score_trend, span_years;
  see `build_player_shape_vectors`) is precomputed once at startup for
  every player with scored history and z-score normalized, so
  `nearest_similar_careers` can rank all ~12,700 players by normalized
  Euclidean distance to a query player in one vectorized pass rather
  than per-pair dynamic time warping. It's a deliberately lightweight
  heuristic ("did these two careers start, end, and swing a similar
  way"), not a claimed exact trajectory match. `/api/players/{id}/career`
  now carries its own `similar_careers` (top 5, empty for a one-stop
  career, since there's no real "shape" to match) alongside the existing
  timeline data, so the suggestions need no second request. On the
  frontend, clicking a suggestion (or manually searching a second
  player) overlays their career onto the same chart on real calendar
  dates. Player B's stops render as squares rather than circles so the
  two are visually distinguishable without needing a second dot color,
  since color is already reserved sitewide for score. The stops table
  gains a Player column and merges both careers' rows by date;
  searching a brand-new primary player clears any active comparison,
  since the old comparison partner is no longer relevant to a different
  career.
- Added a cross-league flow breakdown to League Trends: each league's
  modal now shows its top trading partners by transfer count (incoming
  grouped by origin league, outgoing grouped by destination league),
  excluding any transfer that stayed within the league itself, since
  otherwise every league's own top "buys from" entry would trivially be
  itself. Computed for all ~20-25 qualifying leagues at startup
  (`league_flow()`/`build_league_trends`) rather than on demand, unlike
  the per-club detail added to Club Report Cards below: few enough
  leagues that precomputing every one is cheap, and
  `/api/leagues/trends` already ships every row unpaginated. Reuses the
  same `.breakdown-row`/`.breakdown-bar-fill` bar markup as Club Report
  Cards' position breakdown, scaled to each list's own top count rather
  than a 0-100 score.
- Added three Club Report Cards features together: a head-to-head club
  comparison (two autocomplete pickers over a new lightweight
  `/api/clubs/report-card-search`, which searches club_report_cards_df's
  own club-name universe rather than clubs_lookup.csv's; see
  build_club_report_cards for why those two never get joined), a
  spend-vs-incoming-quality-by-year chart per club (indexed to that
  club's own first year of data = 100, adapting Analytics'
  `buildVolumeChart`/League Trends' `buildTrendSVG` pattern), and a
  recruiting-by-position breakdown (reusing the existing
  `.breakdown-row`/`.breakdown-bar-fill` score-breakdown markup for a
  free, consistent-looking bar chart). Both new per-club datasets are
  computed fresh per request via a new `/api/clubs/report-card?name=X`
  endpoint (exact-match lookup, 404 if not found) rather than added to
  every row of the paginated leaderboard, since only one club is ever
  viewed in this much detail at a time. The existing single-club modal
  now fetches this endpoint too and renders the same two sections into a
  placeholder a moment after the rest of the card, which already renders
  instantly from the leaderboard row in hand. The comparison table only
  highlights a "winner" on stats with an unambiguous direction (avg
  incoming score, avg resale profit). Total spent, transfer counts, and
  departure score are shown neutrally, since a departing player's
  next-club average isn't obviously good or bad for the club that let
  them go. Comparison state (`club_a`/`club_b`) was folded into the
  page's existing `syncURL()` alongside the table's own filters, rather
  than each writing its own half of the query string and clobbering the
  other's, since `writeURLParams` replaces the whole query string on
  every call. The head-to-head section is collapsible too, same
  `.collapse-toggle` pattern as Model vs Reality's chart below.
- Added a predicted-vs-actual scatter chart to Model vs Reality
  (`/api/surprises/scatter`, the full ~7,800-point set, not scoped to
  the table's own filters, same reasoning as `/api/analytics`), reusing
  Analytics' chart-drawing conventions (`linearScale`, `scoreGridlines`,
  click/tap-to-open-detail-card via event delegation) but colored by
  `deltaColor` (green above the y=x line, red below) instead of the
  usual absolute-score `scoreColor`, since the question this chart
  answers is which side of the model's guess a transfer landed on, not
  how good the outcome was in isolation. Made the chart's card
  collapsible (a `<button>` wrapping the `<h2>`, toggling
  `aria-expanded` and the body's `[hidden]`), since it's the one
  card-length chart on a page that's otherwise a table, so a reader who
  just wants the table shouldn't have to scroll past ~7,800 plotted
  points to reach it.
- Added fee/age range filters and a "Export these results as CSV" link
  (`/api/transfers/export`, `/api/loans/export`, uncapped, ignores
  pagination) to Browse and Loans, plus a "✓ Permanent" badge on any
  loan whose player later signed permanently for the same club, in both
  the table and the detail card. The conversion check
  (`find_loan_conversion()`) matches a loan row against a later transfer
  row on player id + exact from/to club, taking the earliest such match;
  verified against a real example (Timur Suleymanov's 2023 Pari NN →
  Loko Moscow loan converting to a permanent transfer in mid-2024). All
  three filters, the export link, and the new range inputs stay in sync
  with each other and the URL through the same `currentFilterParams()`/
  `syncURL()` pattern the existing search/position/league filters
  already used. The fee inputs' label and displayed value didn't update
  on a currency change at first, the same bug Predict's fee field
  already had and fixed (see `updateFeeCurrencyDisplay` below). Ported
  the same fix to Browse, keeping `state.minFeeEurM`/`maxFeeEurM` as the
  real currency-independent bounds and only converting for display.
- Extended Compare from a fixed A/B pair to 2-4 scenarios: "+ Add
  another option" reveals a 3rd/4th column (each removable), the
  results grid and "Top factors" breakdowns are now built dynamically
  per result instead of two hardcoded columns, the highest score gets a
  highlighted border, and the old single pairwise "delta" sentence
  became a `verdictSentence()` that ranks all of them and names the
  winner plus its gap to the runner-up. `/api/compare` changed shape to
  match (a `scenarios: [{request, label}, ...]` list, 2-4 of them,
  instead of hardcoded `a`/`b`/`label_a`/`label_b`), a breaking change
  with no other consumer to worry about. The URL-sync/restore feature
  below generalized the same way: `syncURL()` writes every scenario that
  was actually in the last comparison (dropping a stale 3rd/4th
  scenario's params if the next comparison only has two), and
  `restoreFromURL()` reveals however many c/d columns a link specifies
  before resolving them.
- Made a finished Compare comparison bookmarkable and shareable: both
  scenarios' player/club/fee/age sync to the URL after a successful
  compare, and opening that link resolves everything and re-runs the
  comparison automatically. Needed a new `/api/players/{player_id}`
  endpoint (players_lookup.csv has no by-id lookup otherwise, only
  `/api/players/search`), registered *after* the existing
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
  already exists. Deliberately fee-only, not age too, since a player's
  age at a hypothetical transfer isn't something to explore a range of.
  Caught while testing it by hand: since age has no live re-predict, the
  age marker chart was reading the age field live at render time, so
  editing age without re-submitting and then triggering any re-render
  (e.g. a currency change) plotted the new, unsubmitted age against the
  *old* score. Fixed by snapshotting the exact age/fee a prediction was
  computed from (`state.lastPredictInputs`) instead of re-reading the
  form.
- Synced Browse/Loans/Model vs Reality/Club Report Cards/League Trends'
  search/filter/sort/page state to the URL (via `history.replaceState`),
  so a filtered view is finally bookmarkable and shareable instead of
  always resetting on reload. See `readURLParams()`/`writeURLParams()`
  in `settings.js`.
- Removed the Analytics page's success-score-by-position box plot (down
  to four charts) and replaced every chart's hover-only `<title>`
  tooltip (the trend-line markers, the yearly points on the
  market-over-time chart) with a real click/tap-triggered tooltip
  (`#chart-tooltip`, positioned in JS near the click point via
  `showChartTooltip()` in `analytics.js`). `<title>` alone never fires
  on a touch device (there's no hover state to trigger it) and has a
  real delay even on desktop; `<title>` is kept alongside the new
  `data-tooltip` attribute as a free bonus for a patient mouse user, but
  it's no longer the thing relied on.
- Fixed every hand-drawn chart on the site (Player Timelines, League
  Trends, Analytics) rendering tiny and stranded in a large dead gap
  below a narrow viewport. Each `<svg>` set `width="100%"` but a fixed
  pixel `height`, which only matches its `viewBox`'s aspect ratio at
  exactly the viewBox's own width; narrower than that, the browser
  scales the chart down to fit the width while the element's box keeps
  the old fixed height. `.timeline-chart-wrap svg { width: 100%; height:
  auto; }` in `style.css` fixes all of them from one place, no per-chart
  JS changes needed. Also made the Analytics page's three scatter charts
  (fee vs. score, age vs. score, fee vs. market value) clickable. Each
  point now opens its full `/api/transfers/detail` card, the same as
  every other list page, via one delegated click listener per chart
  rather than one per point; a larger invisible circle on top of each
  small visible dot gives touch a real target to hit. And restyled the
  fee/age trend line from a flat `var(--text)` line (black in light
  theme, competing with the scoreColor'd dots for attention) to a dashed
  light-blue line (`--trend-line`) with small hoverable markers at each
  underlying bucket, reading as its own series rather than a slash
  across the chart.
- Added an Analytics page (`/analytics.html` + `/api/analytics`): five
  hand-drawn SVG charts (fee vs. score, age vs. score, score by position,
  fee vs. market value, market volume over time) in the same no-library
  inline-SVG style as Player Timelines' career chart and League Trends'
  index chart, rather than pulling in a charting dependency for one page.
  Caught along the way: `DataFrame.where(cond, None)` on a float64 column
  casts `None` straight back to `NaN` to preserve the column's dtype
  instead of promoting it to `object`, silently reintroducing the exact
  raw-NaN-leak bug a `/api/leagues/trends` test already guards against on
  a different endpoint. Fixed by converting each scatter column
  explicitly (`None if pd.isna(v) else float(v)`, per value) instead of
  trusting a whole-DataFrame `.where()`, and added the equivalent test for
  this endpoint.
- Renamed the "Biggest Surprises" page to "Model vs Reality" and added a
  "Most accurate predictions first" option to its Show filter (sorting by
  `abs_surprise_delta`, the new backend field is `|surprise_delta|`,
  computed once alongside the existing merge in `app/main.py`), since the
  old name undersold a page that can now also surface the model's closest
  calls, not just its biggest misses.
- Ran a third club-name sweep, this time matching on string similarity
  (`difflib`) instead of shared whole words, to catch the abbreviation/typo
  pairs the second sweep's method structurally couldn't ("Y.
  Malatyaspor"/"Yeni Malatyaspor" share no whole word; neither do "Aalesund"/
  "Aalesunds FK" once the trailing "s" is accounted for). Added 16 more
  verified `CLUB_NICKNAME_GROUPS` entries, one genuine data-entry
  inconsistency ("FC Helsingör"/"FC Helsingør", two different Nordic
  letters for the same Danish club), and rejected "Atalanta"/"Atlanta"
  (Italy's Atalanta BC vs. MLS's Atlanta United, different leagues),
  "Metalurg D."/"Metalurg Z." (two different Ukrainian clubs sharing a
  single-top-flight country, so the usual league check can't tell them
  apart), and "Al-Wehda"/"Al-Wahda" (likely different Saudi/UAE clubs).
  Found a second Metalist-shaped case too: Belgian "Beerschot AC" only
  appears in 2013, the year the club went bankrupt. Flagged to the user,
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
  its own transfer history. One cluster, Ukrainian "Metalist Kharkiv" (which
  has a real, still-contested ownership dispute after its 2016 bankruptcy),
  was flagged to the user rather than guessed; only the confirmed-safe half
  of that cluster was merged.
- Made Club Report Cards' six highlights (best/worst signing, best/worst
  flip, best/worst departure) clickable. Each one now drills into that
  specific transfer's own full breakdown card in the same modal, with a
  "← Back" link to return to the club view. Required adding `player_id` to
  `build_club_report_cards`' highlight dicts in `app/main.py`, since they
  previously only carried enough to describe themselves, not enough to
  look themselves up via `/api/transfers/detail`.
- Restructured the nav into three dropdowns (Predict, Browse, Insights)
  plus Home and About, and split `/` into a real home page (the curated
  showcase cards, moved here, plus a linked directory of every page) with
  the predict form moved to its own `/predict.html`, since nine flat nav
  links had become too many for one row. `settings.js` gained the shared
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
  couple of points), a genuine "spending is outpacing performance" pattern
  in this data, not an assumption going in.
- Added Player Timelines (`/player.html` + `/api/players/{id}/career` +
  `/api/players/career-search`): search any player to see their whole
  scored career as one chronological, real-date-scaled chart: permanent
  transfers as solid dots, loans as hollow rings, both colored by score.
  Added a dedicated search endpoint rather than reusing the Predict form's
  player autocomplete, since that one is filtered to `players_lookup.csv`'s
  market-value threshold and would have silently hidden ~59% of players
  with real scored history (checked directly), mostly retired or
  lower-value players, exactly the kind of long, uneven career this page
  exists to show.
- Added Club Report Cards (`/clubs.html` + `/api/clubs/leaderboard`): every
  club ranked as a recruiter (incoming transfers, spend, buy-develop-resell
  profit, how departing players did elsewhere). Found and fixed a real bug
  along the way: an early version attributed a club's *buyer's* eventual
  resale profit to whichever club had sold it that player originally (e.g.
  crediting Real Madrid's sale of Ronaldo to Juventus with Juventus's own
  later resale of him). Resale profit belongs to the buyer
  (`to_club_name`), not the seller, since `next_transfer_fee` is what a
  third club later paid *that buyer*. Also added `build_club_name_aliases`
  to merge ~75 same-club name variants that differ only by a generic
  legal-entity marker or accent encoding ("FC Barcelona"/"Barcelona"), then
  extended it with `CLUB_NICKNAME_GROUPS`, ~28 hand-verified
  nickname/official-name pairs sharing no token at all ("Man
  City"/"Manchester City", "PSG"/"Paris Saint-Germain", "Bor.
  Dortmund"/"Borussia Dortmund"/"Dortmund"), each checked against the real
  data (matching league both sides, non-contradictory dates) rather than
  assumed, which is what confirmed "Sporting" is safe to fold into Sporting
  CP specifically rather than the ambiguous case ("Sporting" could also
  mean Sporting Gijón or Royal Charleroi) it looks like on name alone.
  `transfers_processed.csv` has no club_id, so without either fix, several
  major clubs' report cards were silently missing part of their real
  history.
- Moved club-name canonicalization to run once at startup on
  `transfers_df`/`loans_df`/`comparables["meta"]` directly (previously only
  applied inside Club Report Cards' own aggregation). Browse, Loans,
  Biggest Surprises, and Predict/Compare's comparables now all show the
  same name for the same club too, not just `/clubs.html`. Also caught and
  force-split a false merge the mechanical pass would otherwise have made
  once loans data joined the count: "Racing" and the unrelated Argentine
  "Racing Club" share the generic word "club" but not a league.
- Changed the canonical-name pick from "whichever spelling has the most
  transfer mentions" to "whichever spelling is longest" ("Tottenham
  Hotspur" over "Tottenham", "Arsenal FC" over "Arsenal"), since a reader
  unfamiliar with a shorthand is more likely to recognize the fuller name.
- Added billions formatting (`eur_m()`/`formatMoney()`, e.g. `"€2049m"` ->
  `"€2.05b"`). No individual transfer fee reaches a billion, but a club's
  aggregate spend or resale-profit total on `/clubs.html` regularly does.
- Added a "Biggest Surprises" page (`/surprises.html` + `/api/surprises`):
  every transfer ranked by how far its real outcome diverged from a 5-fold
  cross-validated, held-out model prediction, showing the biggest
  overachievers and busts a pre-transfer-only model didn't see coming.
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
