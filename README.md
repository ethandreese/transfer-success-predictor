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

Five sub-metrics, each converted to a percentile rank across the dataset
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
- **market value growth** — from just before the transfer to near the
  end of the tenure
- **playing time** — total appearances made (established starter vs.
  bench/injury-plagued)
- **value for money** — performance level vs. what was paid *relative
  to the player's market value at the time* (comparing fees to the whole
  dataset's fee distribution made any nine-figure move look "expensive"
  even when it was a bargain for that specific player)

Weights: attackers lean heavily on performance (30%/15%) since goal
contributions are a real, differentiating signal for them (only 5% have
zero goal contributions in a given window). That signal gets progressively
less reliable for midfielders (12% zero) and defenders (21% zero), and is
essentially meaningless for goalkeepers (nearly all have exactly 0 goals +
assists both before and after a move — no dataset column captures clean
sheets, saves, or defensive actions). So performance weight shrinks from
45% combined (attackers) to 0% (goalkeepers), shifted into value growth,
playing time, and value for money instead — signals that stay meaningful
regardless of position.

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
  overstates how confident a R²≈0.07 model can be), a "why this score"
  breakdown, and the nearest historical comparables.
- **`/browse.html`** — every scored transfer (~7,300), filterable by
  position and destination league, searchable by player/club name, sortable
  by score/date/fee/age, paginated.
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
- "Playing time" is a raw appearance count, so a longer tenure has more
  chances to accumulate it than a short, excellent one — a deliberate
  choice (staying and playing for years is itself part of "success"), but
  worth knowing.
- Predicting a *new* hypothetical transfer is meaningfully less reliable
  than the historical scores shown for known transfers, since the model
  only sees pre-transfer information by construction.
- League-adjustment currently only feeds into the *historical* score (the
  training label). The predictive model's input features still use raw
  pre-transfer output rather than a league-adjusted version, so it can
  only learn league-difficulty effects indirectly (via the destination
  league as a category); giving it an explicit league-relative input
  feature is a natural next improvement.
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
