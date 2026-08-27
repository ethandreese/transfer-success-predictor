# Transfer Success Predictor ⚽

Predicts how a football (soccer) transfer is likely to go, trained on real
historical transfer data rather than hand-picked examples like Haaland→City
or Dembélé→Barça.

## How it works

**Data.** [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)
(mirrored on Kaggle as [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores)),
~50k players, ~175k transfers, ~1.9M appearances, ~656k market valuations.

**Success score (the training label, 0–100).** For each transfer, computed
from data that only exists *after* the move — and measured over the
player's **entire tenure** at the new club (from the transfer until their
next departure, or "now" if they're still there), not just year one. A
fixed first-year window either unfairly penalizes a slow starter who took
time to adapt, or misses someone who started hot and faded once the
honeymoon period ended.

Six sub-metrics, each converted to a percentile rank across the dataset
(so no single stat's raw scale dominates), then blended with **weights that
vary by position** (see `data/score_weights.json`):

- **performance level** — goal contributions per 90 minutes at the new
  club *relative to the actual league average for that position* (see
  below), ranked *within the player's position group* (comparing a
  striker's output to the whole dataset, mostly defenders and keepers,
  made every decent attacker look elite and barely separated "good" from
  "Haaland")
- **performance change** — improved or declined vs. their league-relative
  level before the move, also position-ranked
- **market value growth** — blends two signals: 60% growth to the
  *peak* value reached at any point during the tenure, 40% growth to the
  value near the end of it. End-value alone unfairly reads a long,
  valuable career as a decline, since even the best players' market value
  falls with age by the time they eventually leave — Heung-min Son joined
  Tottenham valued at ~€16m, peaked at €90m mid-tenure, and was worth only
  ~€20m a decade later when he left. Peak alone would ignore a real
  late-tenure collapse (injury, loss of form), so end-value is still kept
  as a smaller factor.
- **playing time** — blends two signals: 60% percent of the *team's actual
  games* played during the tenure (from `games.csv`/`club_games.csv` — the
  club's full match schedule across all competitions, not just games the
  player featured in), 40% raw appearance count. The percentage catches
  injuries/rotation that a raw count hides — Dembélé made 185 appearances
  for Barcelona over 6 years (a big number on its own), but that's only
  57% of the 327 games Barcelona actually played in that span, versus
  Haaland at 83% for Man City. Raw count is kept alongside it so a long,
  genuinely sustained career at the club still counts for something beyond
  the percentage alone.
- **value for money** — performance level vs. what was paid *relative
  to the player's market value at the time* (comparing fees to the whole
  dataset's fee distribution made any nine-figure move look "expensive"
  even when it was a bargain for that specific player)
- **resale profit** (weight varies by tenure length, only when known —
  see below) — did the buying club later resell the player for more than
  they paid? A real, distinct signal from sporting performance: a
  decent-but-unspectacular player who's later flipped for a profit is a
  good outcome for the club even if he was never a star there.

Weights: attackers lean heavily on performance (28%/14%) since goal
contributions are a real, differentiating signal for them (only 5% have
zero goal contributions in a given window). That signal gets progressively
less reliable for midfielders (11% zero) and defenders (21% zero), and is
essentially meaningless for goalkeepers (nearly all have exactly 0 goals +
assists both before and after a move — no dataset column captures clean
sheets, saves, or defensive actions). So performance weight shrinks from
42% combined (attackers) to 0% (goalkeepers), shifted into value growth,
playing time, and value for money instead — signals that stay meaningful
regardless of position.

**Resale profit is only counted when known**, which is deliberately rare:
only ~17% of transfers have a genuine subsequent sale for a recorded fee.
The raw data doesn't distinguish loans from permanent transfers, and most
"next transfer, fee €0" cases are loans rather than real free exits, so
those are treated as *unknown* rather than guessed at as a loss either way
— a still-at-the-club or loaned-out player isn't penalized for something
that hasn't happened yet. When it's unknown, resale profit's weight is
dropped and the other five weights are renormalized to still sum to 1,
rather than filling in a fabricated "neutral" score for data that doesn't
exist. When it *is* known, it's a real signal: Moisés Caicedo joined
Brighton for free and was later sold to Chelsea for €116m — a textbook
example of a transfer that looks fine on the pitch but was primarily a
business win.

**Resale profit's weight scales with tenure length**, from ~20% for a
transfer of a few months down to ~2% for a decade-long career (see
`resale_weight_curve` in `data/score_weights.json`: `min + (max - min) *
exp(-tenure_years / decay_years)`, landing around the old flat 8% at the
curve's ~2-year reference point). A quick flip weights the eventual sale
price heavily, since making money on the resale is often close to the
point of a short-term deal. A long career barely moves regardless of how
the sale eventually went, because the club already extracted years of
on-pitch value from the player having played there — Heung-min Son was
bought for ~€30m and eventually sold for less, a resale loss on paper,
but after a decade of service that loss counts for only ~2% of his score
(75.6 overall). The "why this score" breakdown states this explicitly whenever
it applies, e.g. *"counts for only 3% of the score here — after a
6.0-year tenure the club already got most of its value..."*

**League-adjusted performance.** Goal contributions are judged against how
hard it actually is to score in that specific league, not the whole
dataset. For each (league, position) pair we compute the average goal
contributions/90 across *all* appearances in that league (not just our
~7,300 filtered transfers — this uses the full ~1.9M-appearance dataset, so
even leagues with few transfers in our sample get a stable baseline). A
player's raw output is then expressed as a multiple of that baseline (e.g.
"2.2x the league average") before being percentile-ranked. Concretely:
Bundesliga attackers average 0.54 goal contributions/90 across the whole
dataset, Premier League attackers average 0.48 — moving from one to the
other is a real step up in difficulty, and the score now reflects that.
This is what fixed Haaland's Dortmund→Man City score initially looking too
low: his raw output dipped slightly (1.39→1.07 per 90), but relative to
each league's baseline he stayed just as dominant (2.6x→2.2x), which the
formula now credits instead of penalizing.

**Performance change vs. expectation, not raw delta.** Even with league
adjustment, comparing a player's league-relative level *before* the move to
*after* the move (a raw difference) still unfairly penalizes players who
were already near the top: someone at 2.6x their league's average has far
more room to fall than to rise, so almost any real outcome short of getting
*even better* reads as "decline" — even though everyone that dominant tends
to pull back toward the pack somewhat (regression to the mean), and that
pullback isn't really a sign the move went badly. So instead of a raw
difference, "performance change" fits a simple regression of post-transfer
level on pre-transfer level (per position, across the dataset) and measures
the *residual* — did the player end up better or worse than what's
statistically typical for someone who started at their level? A player who
pulls back by exactly the expected amount scores neutrally; someone who
beats that expectation (Haaland: expected ~1.3x, actually stayed at 2.2x)
scores well; someone who falls far short of even the regressed expectation
(Sancho at Man Utd: expected ~1.2x, actually fell to 0.7x) still scores
badly. This is what pushed Haaland's "performance change" from the 30th
percentile to the 96th without touching genuine busts like Sancho.

**Model.** A gradient-boosted regressor trained on pre-transfer-only
features (age, position, physical attributes, fee, market value, prior-year
performance, and origin/destination club & league strength) — nothing about
what happened after the move. Evaluated on a temporal holdout (trained on
transfers before mid-2023, tested on transfers since): **MAE ≈ 16 points**
on the 0–100 scale, R² ≈ 0.10, vs. ≈17 MAE for always predicting the
average. That's a modest but real signal, and honestly weaker than scoring
a fixed first year would give — predicting a player's *entire future stint*
at a new club from pre-transfer stats alone is genuinely hard, since
multi-year outcomes depend heavily on injuries, tactics, and squad fit that
no pre-transfer number can see. The app surfaces this error rate and a
per-prediction "why this score" breakdown rather than presenting the number
as gospel.

**Explainability.** For known historical transfers, the app shows the
actual 5-component breakdown above, each with the concrete underlying
numbers (e.g. "0.92 goal contributions/90 at Barcelona, ranked vs. other
attackers", "€33m → €60m market value"), not just a bare score. For a new
hypothetical prediction (where there's no real post-transfer data yet), it
shows each feature's contribution by comparing the prediction to what a
"typical transfer" would score with that one feature swapped to its
dataset median or mode — e.g. "35.1 yrs vs. a typical transfer's 25.4 yrs,
lowering the score by 11.1 pts" — a simple, transparent stand-in for a
proper SHAP explanation, with league codes and fees resolved to readable
names/€m rather than raw feature values.

## Pages

- **`/`** — predict a hypothetical transfer: search a real player, pick a
  destination club, see a predicted score, a likely range (min/max among
  the 5 most similar real transfers, since a single point estimate
  overstates how confident a R²≈0.10 model can be), a "why this score"
  breakdown, and the nearest historical comparables.
- **`/browse.html`** — every scored transfer (~7,300), filterable by
  position and destination league, searchable by player/club name, sortable
  by score/date/fee/age, paginated. Click any row to open that transfer's
  full card (score + breakdown) in a modal.
- **`/compare.html`** — set up two hypothetical transfers side by side
  (same player to two different clubs, or two different players entirely)
  and see both predictions, ranges, and top factors together with the
  point gap between them.

## Known limitations

- No column in the source data captures defense-specific output (tackles,
  clean sheets, saves) — value growth, playing time, and value for money
  have to carry defenders' and goalkeepers' scores almost entirely, so a
  quietly excellent defensive performance that the market didn't
  re-value accordingly may be underrated.
- Only transfers with ≥10 appearances in both the year before and the whole
  tenure after are included (~7,300 of ~159k), which skews the training
  data toward established first-team players rather than fringe/loan moves.
- "Playing time" still keeps a 40% raw-count component alongside the
  percent-of-games-played signal, so a longer tenure still has somewhat
  more room to accumulate a high score than a short, excellent one — a
  deliberate choice (sustained presence is itself part of "success"), but
  worth knowing. The "games the team played" denominator counts all
  competitions combined (league, domestic cup, continental) rather than
  just league games, on the view that squad rotation happens across all of
  them - but that also means a player who's rested for cup games (not
  unavailable, just rotated) looks identical to one who's actually
  injured; the data doesn't distinguish the two.
- Predicting a *new* hypothetical transfer is meaningfully less reliable
  than the historical scores shown for known transfers, since the model
  only sees pre-transfer information by construction.
- League-adjustment only feeds into the *label* (the historical success
  score), not the model's input features. This was tried deliberately:
  `data/league_baselines.csv` (the same per-league/position goal-
  contribution baselines the label uses, persisted as a small artifact so
  the app can share it) was added as extra model features - a
  league-adjusted pre-transfer performance number, plus the raw
  origin/destination league baselines - and tested against 5 feature
  combinations and 5 model configs on the temporal holdout. None beat the
  original, simpler feature set's R² (0.103); a couple made it slightly
  worse. The destination league category already captures most of that
  signal for leagues with enough training transfers, and ~4,400 training
  rows isn't enough to reliably learn the added continuous relationships
  on top of that. Kept the simpler feature set rather than adding
  complexity that measurably doesn't help - see `scripts/train_model.py`
  for the reasoning. `league_baselines.csv` is still put to use, just for
  explanation quality rather than accuracy: the live "why this score"
  panel shows a player's pre-transfer output as a multiple of their
  current league's average (e.g. "0.43 per 90 (2.1x their current
  league's average)"), consistent with how historical cards explain
  performance, even though the model itself learns from the raw number.
- League baselines are averaged across the whole 2013–2026 window rather
  than computed per-season, so a league that got notably more/less
  attacking over that time isn't captured precisely.

## Project layout

```
scripts/build_dataset.py   # raw Transfermarkt CSVs -> data/transfers_processed.csv
scripts/train_model.py     # trains the model + comparable-transfers index
scripts/build_lookups.py   # small player/club search tables for the web app
app/main.py                 # FastAPI backend (serves the API + the static frontend)
app/static/                 # vanilla HTML/CSS/JS frontend (index/browse/compare)
data/                        # committed: small derived CSVs only (~3.5MB total)
tests/                       # pytest suite - runs against committed artifacts only
```

The raw Transfermarkt CSVs (~730MB) are **not** committed — they're
downloaded locally via [kagglehub](https://github.com/Kaggle/kagglehub) and
only used to regenerate `data/*.csv`.

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# only needed if you want to regenerate data/*.csv from scratch:
./.venv/bin/python -c "import kagglehub; kagglehub.dataset_download('davidcariboo/player-scores')"
./.venv/bin/python scripts/build_dataset.py
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/train_model.py

./.venv/bin/uvicorn app.main:app --reload
```

Then open http://localhost:8000.

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
search and the compare/predict flows).
