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
(so no single stat's raw scale dominates), then blended:

- **30% performance level** — goal contributions per 90 minutes at the new
  club, ranked *within the player's position group* (comparing a striker's
  output to the whole dataset, mostly defenders and keepers, made every
  decent attacker look elite and barely separated "good" from "Haaland")
- **15% performance change** — improved or declined vs. their level before
  the move, also position-ranked
- **20% market value growth** — from just before the transfer to near the
  end of the tenure
- **10% playing time** — total appearances made (established starter vs.
  bench/injury-plagued)
- **25% value for money** — performance level vs. what was paid *relative
  to the player's market value at the time* (comparing fees to the whole
  dataset's fee distribution made any nine-figure move look "expensive"
  even when it was a bargain for that specific player)

**Model.** A gradient-boosted regressor trained on pre-transfer-only
features (age, position, physical attributes, fee, market value, prior-year
performance, and origin/destination club & league strength) — nothing about
what happened after the move. Evaluated on a temporal holdout (trained on
transfers before mid-2023, tested on transfers since): **MAE ≈ 16 points**
on the 0–100 scale, R² ≈ 0.07, vs. ≈17 MAE for always predicting the
average. That's a modest but real signal, and honestly weaker than scoring
a fixed first year would give — predicting a player's *entire future stint*
at a new club from pre-transfer stats alone is genuinely hard, since
multi-year outcomes depend heavily on injuries, tactics, and squad fit that
no pre-transfer number can see. The app surfaces this error rate and a
per-prediction "why this score" breakdown rather than presenting the number
as gospel.

**Explainability.** For known historical transfers, the app shows the
actual 5-component breakdown above. For a new hypothetical prediction
(where there's no real post-transfer data yet), it instead shows each
feature's contribution by comparing the prediction to what a "typical
transfer" would score with that one feature swapped to its dataset median —
a simple, transparent stand-in for a proper SHAP explanation.

## Known limitations

- Goal contributions per 90 is a weak signal for goalkeepers and
  ball-playing defenders; the score leans more informative for attacking
  players.
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

## Project layout

```
scripts/build_dataset.py   # raw Transfermarkt CSVs -> data/transfers_processed.csv
scripts/train_model.py     # trains the model + comparable-transfers index
scripts/build_lookups.py   # small player/club search tables for the web app
app/main.py                 # FastAPI backend (serves the API + the static frontend)
app/static/                 # vanilla HTML/CSS/JS frontend
data/                        # committed: small derived CSVs only (~3.5MB total)
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
