# Transfer Success Predictor ⚽

Predicts how a football (soccer) transfer is likely to go, trained on real
historical transfer data rather than hand-picked examples like Haaland→City
or Dembélé→Barça.

## How it works

**Data.** [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)
(mirrored on Kaggle as [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores)),
~50k players, ~175k transfers, ~1.9M appearances, ~656k market valuations.

**Success score (the training label, 0–100).** For each transfer, computed
from data that only exists *after* the move:

- 50% — change in goal contributions per 90 minutes, comparing the player's
  final year at the old club to their first year at the new club
- 30% — market value growth from just before the transfer to ~1 year after
- 20% — appearances made in the first year at the new club (a proxy for
  becoming an established starter vs. a bench/injury-plagued outcome)

Each sub-metric is converted to a percentile rank across the dataset before
blending, so outliers (own goals in the data, a single crazy season) don't
dominate.

**Model.** A gradient-boosted regressor trained on pre-transfer-only
features (age, position, physical attributes, fee, market value, prior-year
performance, and origin/destination club & league strength) — nothing about
what happened after the move. Evaluated on a temporal holdout (trained on
transfers before mid-2023, tested on transfers since): **MAE ≈ 14 points**
on the 0–100 scale, vs. ≈17 for always predicting the average. That's a real
but modest signal — transfer outcomes are inherently noisy (injuries,
tactics, squad fit), and the app says so.

## Known limitations

- The success score only looks at the player's *first year* at the new
  club — it won't fully capture a story like Dembélé's, whose "bust"
  reputation was really about recurring injuries across several seasons.
- Goal contributions per 90 is a weak signal for goalkeepers and
  ball-playing defenders; the score leans more informative for attacking
  players.
- Only transfers with ≥10 appearances in both the year before and the year
  after are included (~6,700 of ~148k), which skews the training data
  toward established first-team players rather than fringe/loan moves.

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
