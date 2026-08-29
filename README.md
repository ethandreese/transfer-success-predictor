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

Ten sub-metrics, each converted to a percentile rank across the dataset
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
- **four FotMob-derived components** (each only when FotMob has that
  specific bucket's tenure stats — see below), kept as four separate
  scores rather than blended into one, so a player's actual profile
  survives into the score instead of getting smoothed away — an
  attack-minded fullback and a purely defensive one could land on the same
  *blended* number despite having very different games. Each is
  percentile-ranked *within position group*, like performance level:
  - **rating** — FotMob's own per-match rating, averaged across the
    tenure. The one component goalkeepers get a real, direct quality
    signal from — every other non-FotMob component either doesn't apply to
    them (performance level/change) or is a financial/availability proxy
    (value growth, playing time, value for money).
  - **attacking** — goals, expected goals (xG), expected assists (xA), and
    chances created per 90, all averaged together. Zero weight for
    goalkeepers. Distinct from performance level/change above: those
    measure *actual goal contributions already banked* (a results-based,
    lagging signal), this measures the *underlying attacking process* — a
    player generating high xG and chances created despite a quiet
    finishing spell shows up well here even if perf_level doesn't (yet)
    reflect it.
  - **defensive** — tackles, interceptions, clearances, and recoveries per
    90 for outfielders; saves per 90, save percentage, and goals conceded
    per 90 (inverted, since fewer is better) for goalkeepers — genuinely
    different stat pools, not a shared one, since neither set means
    anything for the other position.
  - **possession** — accurate passes per 90 and successful dribbles per
    90. Dribbles are grouped here rather than under attacking because
    they're fundamentally about ball-carrying and retention under
    pressure — a possession skill, even though a dribble can also lead
    directly to a chance. Matches how e.g. FBref categorizes take-ons
    under "Possession" rather than "Shooting" or "Passing".
- **resale profit** (weight varies by tenure length, only when known —
  see below) — did the buying club later resell the player for more than
  they paid? A real, distinct signal from sporting performance: a
  decent-but-unspectacular player who's later flipped for a profit is a
  good outcome for the club even if he was never a star there.

Weights: attackers lean heavily on performance (26%/13%) since goal
contributions are a real, differentiating signal for them (only 5% have
zero goal contributions in a given window). That signal gets progressively
less reliable for midfielders (11% zero) and defenders (21% zero), and is
essentially meaningless for goalkeepers (nearly all have exactly 0 goals +
assists both before and after a move — no *Transfermarkt* column captures
clean sheets, saves, or defensive actions). So performance weight shrinks
from 39% combined (attackers) to 0% (goalkeepers), shifted into value
growth, playing time, value for money, and — now that they exist — the
four FotMob components instead, whose combined weight grows from 10%
(attackers, a minor signal on top of real goal data, split 2%/5%/1%/2%
rating/attacking/defensive/possession) up to 35% (goalkeepers, split
7%/0%/24%/4% — defensive dominates since shot-stopping is essentially the
job, attacking is zeroed out entirely) as goal contributions become less
meaningful.

**Resale profit is only counted when known**, which is deliberately rare:
only ~31% of transfers have a genuine subsequent sale for a recorded fee
(up from ~17% once loans were pulled out of the transfer chain - see below;
previously a loan-out sitting between a permanent signing and its eventual
resale made `next_transfer_fee` land on the loan's own unrecorded fee
instead of skipping through to the real sale). A still-at-the-club player,
or one whose next move is a real free transfer or a loan (see "Loan spells
are scored separately" below — loans can no longer land here at all, now
that they're detected and pulled out of the chain before this is
computed), isn't penalized for something that hasn't happened yet; those
cases are treated as *unknown* rather than guessed at
as a loss either way. When it's unknown, resale profit's weight is
dropped and the other five weights are renormalized to still sum to 1,
rather than filling in a fabricated "neutral" score for data that doesn't
exist. When it *is* known, it's a real signal: Randal Kolo Muani joined
Frankfurt from Nantes for free and was sold on to PSG for €95m about 14
months later — a textbook example of a transfer that looks fine on the
pitch but was primarily a business win.

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

**Loans are detected and excluded from the transfer score, then scored
separately.** The packaged Transfermarkt dataset has no loan/permanent flag
at all — its upstream ETL parses every non-numeric fee string (including
"loan transfer", "End of loan", and "Loan fee: €X") down to a flat `0`,
identical to a genuine free transfer, so a loan spell was originally
getting judged with `success_score` as if a club had chosen to buy the
player outright. Fixed by re-fetching each player's real transfer history
from transfermarkt's own live `transferHistory` API
(`scripts/fetch_transfer_types.py`), which still carries that distinction
in its raw fee text, and using it to classify every transfer as
paid/free/loan/unknown (cached in `data/raw/transfer_types_cache.csv`).
Loan rows are dropped before `tenure_end`/`next_transfer_fee` are computed
in `load_transfers()`, which fixes two things at once: the loan spell no
longer scores as a permanent move, and a loan-out interruption in the
middle of a permanent tenure (join → loan elsewhere → return → eventual
sale) no longer cuts that tenure short or gets mistaken for the eventual
resale. Concretely: 55,433 of the ~159k candidate transfers (2013–2026)
are loans - more than a third - and removing them shrank the final scored
transfer set from 7,264 to 5,062, since a fair number of those old rows
were really a loan's tenure being measured, or a permanent tenure getting
cut short at a loan-out date that no longer applies.

**Loan spells are scored separately, on a different formula
(`data/loan_score_weights.json`), not folded into the same score.** A loan
isn't a permanent-transfer decision, so it's judged on different terms: no
*value for money* or *resale profit* component (most loans carry no real
fee, and a loan doesn't end in a sale of its own), and *playing time*
weighted much more heavily than in the permanent score — whether the loan
actually delivered game time is usually the central question it gets
judged on, independent of how well the player performed when they did
play. Unlike the permanent-transfer pipeline, a loan spell with zero
post-loan appearances is *kept*, not filtered out — a player who was sent
out and never played is a real (bad) outcome the Loans tab exists to
surface, not missing data. 3,199 loan spells (of ~28k candidates) are
scored this way - see `/loans.html`.

**The rating/attacking/defensive/possession components come from FotMob,
not Transfermarkt.** The packaged Transfermarkt dataset has no column at
all for defense-specific output — no tackles, clean sheets, saves, or any
other defensive stat — so `scripts/fetch_fotmob_stats.py` (optional, like
`fetch_transfer_types.py`) separately pulls FotMob's public season stat
leaderboards (rating, goals/xG/xA, tackles/interceptions/clearances/
recoveries, passes, saves, etc.) for the 23 leagues that appear as a
transfer destination, and stitches each transfer's *entire tenure* into
`data/raw/fotmob_stats_cache.csv`, the same tenure-window philosophy as the
rest of the score. Two problems had to be solved to make that stitching
correct, not just plausible:

- *Multi-season tenures.* FotMob only exposes whole-season leaderboards, so
  a multi-year tenure needs several seasons combined — rate stats
  (rating, per-90 numbers) are minutes-weighted averages across the
  seasons used, count stats are summed.
- *Same-league, mid-season moves.* FotMob attributes a player's entire
  season total to whichever club they're registered at when fetched, not
  split by stint — confirmed on real cases, e.g. Marc Guéhi's 19 Jan 2026
  Crystal Palace → Man City move showing a near-full season of minutes
  under Man City, most of which were actually played for Palace. This only
  contaminates a same-*league* move made mid-season (a cross-league
  arrival's "before" stats live in an entirely different league's
  leaderboard, so they can't leak in) — the contaminated join season is
  excluded from the stitched aggregate whenever it applies, verified
  correct on real cases like Virgil van Dijk's January 2018
  Southampton → Liverpool move (2017/18 correctly excluded, the other 9
  seasons stitched together).

Matching a FotMob player onto a Transfermarkt one is also harder than it
sounds — neither site exposes the other's player id, and club names often
share no text in common at all ("Man City"/"Manchester City", "PSG"/"Paris
Saint-Germain", "Copenhagen"/the Danish spelling "København", which was
initially a real bug: the accent-stripping step only handled letters that
decompose into base+diacritic, so `ø` was being silently dropped instead of
transliterated). A small number of these are handled with an explicit
alias table per league, found by inspecting real unmatched rows rather than
guessed at up front — building one for all 23 leagues wasn't attempted,
since inspecting our own data showed the same club can appear under several
different short names even within one league (Bundesliga transfers use
both "Dortmund" and "Bor. Dortmund" for the same club).

FotMob's coverage has two real ceilings, not effort problems: nothing
before a league-specific season (varies a lot — the Premier League and
Norway's Eliteserien have opposite ends of that range, 2016/2017 vs.
2013/2014), and Ukraine's Premier League specifically, where FotMob's own
league page exposes only 5 basic stat categories (goals, assists,
goals+assists, yellow/red cards) — nothing that overlaps what any of these
four components actually need, so it's a genuine, permanent 0% for that
one league. Overall, 82% of scored transfers end up with a usable score in
*at least one* of the four buckets — coverage varies by bucket (rating
70%, attacking 81%, defensive 72%, possession 72%; rating specifically
seems to need more minutes/matches than the others to qualify on FotMob's
side). Each bucket's weight is dropped independently for a row missing
it, and the other weights renormalized, the same pattern already used for
`resale_profit` - so a transfer can show, say, attacking and possession
but not rating and defensive, and the score still sums to the same 0-100
scale.

**League-adjusted performance.** Goal contributions are judged against how
hard it actually is to score in that specific league, not the whole
dataset. For each (league, position) pair we compute the average goal
contributions/90 across *all* appearances in that league (not just our
~5,000 filtered transfers — this uses the full ~1.9M-appearance dataset, so
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
transfers before mid-2023, tested on transfers since): **MAE ≈ 13.6 points**
on the 0–100 scale, R² ≈ 0.09, vs. ≈14.6 MAE for always predicting the
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

**The "typical" reference for fee-related features is conditional on
whether the transfer being explained is itself paid.** Over half of all
transfers are free (an out-of-contract move or an academy graduate signing
- a fundamentally different circumstance from an active paid deal), which
drags the *overall* median transfer fee to €0. Comparing a real €50m fee
against "a typical transfer's €0m" is misleading — it reads as if paying
anything at all is unusual, rather than telling you whether €50m is high
or low *among paid deals*. So log_transfer_fee and fee_to_value_ratio each
have a second reference value computed only from transfers with a real fee
(`reference_values_paid` in metadata.json, e.g. a typical paid fee is
~€6m, a typical paid fee-to-value ratio is ~1.0x) — used instead of the
overall reference whenever the transfer being explained has `transfer_fee
> 0`. A genuinely free transfer still compares against the overall
reference, which correctly reflects that being free is itself common. One
side effect worth knowing: since paid-vs-paid variation is naturally
smaller than paid-vs-free variation, transfer fee's contribution shrank
and often no longer makes the top-5 explanation — that's the fix working
as intended, not a regression.

**Every "typical" reference is now contextual, not a single flat number
for the whole dataset.** A flat average is a weak baseline for two
reasons, both reported by hands-on use of the app: (1) it ignores
position — an attacker's typical goal contributions (~0.47/90) look
nothing like the whole population's (~0.20/90, dragged down by defenders
and goalkeepers averaging near 0), so *every* attacker's performance
looked artificially inflated against it, and vice versa for defenders; (2)
for fee specifically, it ignores the player's own market value — a €100m
fee for a player already valued at €70m is a modest ~1.4x premium, in
line with what similarly-valued players go for, but comparing the raw
€100m against a flat "typical paid fee" (~€6m, dragged down by many
cheaper deals) made it look like a wild outlier regardless of context.
Two fixes, both in `scripts/train_model.py`:

- `reference_values_by_position` — the median of each position-sensitive
  feature (age, height, appearances, minutes, goal rates, market values)
  computed *within that position*, used instead of the flat median for
  `position_conditional_features`. The explanation now says "vs. a typical
  attacker's 0.47 per 90" instead of "vs. a typical transfer's 0.20 per
  90".
- `fee_regression` — a simple fit of `log(fee) ~ log(market value)` on
  paid transfers, so the fee reference for any given prediction is "what's
  typically paid for a player valued this highly" (e.g. ~€73m for a €70m
  valuation) rather than one number for everyone. This is the same
  regression-to-expectation pattern already used for the historical
  "performance change" component (`compute_expected_post_performance`),
  applied here to fee instead of performance.

Concretely, for a €100m fee on a €70m-valued player, "Transfer fee" used
to read "*vs. a typical transfer's €0m, raising the score by 7.5 pts*"
(comparing against mostly-free transfers) and then, after the first fee
fix, "*vs. a typical paid transfer's €6m, raising the score by 1.2 pts*"
(comparing against a flat paid average) — both frame paying anything
substantial as unusually large. It now reads "*vs. what's typically paid
for a similarly-valued player: €72.7m, **lowering** the score by 1.2
pts*" — correctly recognizing that €100m is a modest premium over a
€70m valuation, not an outlier, and that paying somewhat above market
rate for a player is if anything a mild risk factor rather than
inherently a sign of a big, ambitious move.

**"Compare to similar transfers" already exists as a separate mechanism**
(`find_comparables`, a nearest-neighbor lookup over the full feature
space) and powers both the "most similar historical transfers" list and
the predicted score's likely range — that part was already contextual by
design. The reference-value work above fixes the *per-feature* SHAP-style
breakdown specifically, which used flat dataset-wide statistics rather
than the nearest-neighbor mechanism (using neighbors chosen by a feature
to explain that same feature would be circular — the neighbors would
already be similar on it by construction, trivializing the comparison).

## Pages

- **`/`** — predict a hypothetical transfer: search a real player, pick a
  destination club, see a predicted score, a likely range (min/max among
  the 5 most similar real transfers, since a single point estimate
  overstates how confident a R²≈0.10 model can be), a "why this score"
  breakdown, and the nearest historical comparables.
- **`/browse.html`** — every scored transfer (~5,000), filterable by
  position and destination league, searchable by player/club name, sortable
  by score/date/fee/age, paginated. Click any row to open that transfer's
  full card (score + breakdown) in a modal.
- **`/loans.html`** — every scored loan spell (~3,200), same browse/filter/
  search/click-to-view-card experience as `/browse.html`, but scored on the
  loan-specific formula above (no fee/resale rows in the breakdown, and
  duration shown in months rather than years).
- **`/compare.html`** — set up two hypothetical transfers side by side
  (same player to two different clubs, or two different players entirely)
  and see both predictions, ranges, and top factors together with the
  point gap between them.

## Known limitations

- The base Transfermarkt dataset has no column for defense-specific output
  (tackles, clean sheets, saves) - now substantially addressed by the four
  FotMob-derived components (see above), but not fully: they only cover
  permanent transfers into the 23 leagues FotMob was matched against (loans
  still have no defensive signal at all - see `data/loan_score_weights.json`),
  only from whatever season FotMob's own coverage happens to start for that
  specific league, and not at all for Ukraine's Premier League. About 18%
  of scored permanent transfers still have no FotMob data in any of the
  four buckets and fall back to value growth, playing time, and value for
  money carrying the position almost entirely, as before - and even among
  covered transfers, individual buckets have uneven coverage (rating 70%,
  attacking 81%, defensive 72%, possession 72%), so it's common for a
  transfer to show some but not all four rows.
- FotMob player/club matching relies on name/club text matching (no shared
  id exists between the two sites), which is inherently approximate.
  Verified well for a few high-volume leagues by hand (English club
  abbreviations, then Ligue 1's "PSG"/"Stade Rennais" and Denmark's
  "Copenhagen"/"København", found by inspecting real unmatched rows and
  fixed with explicit aliases - see `scripts/fetch_fotmob_stats.py`), but
  the other 20 leagues rely on a generic matcher only, so their match rate
  is somewhat weaker and less scrutinized (e.g. Ligue 1 and Denmark were
  both under 80% before their specific fixes landed; some other
  unreviewed league likely has a similar gap sitting in it right now). A
  wrong match would show a real player's tenure with a different real
  player's stats rather than failing loudly, which is a meaningfully worse
  failure mode than simply missing data - the fuzzy-match fallback
  requires a destination-club match before accepting a non-exact name, as
  a partial guard against that.
- Only transfers with ≥10 appearances in both the year before and the whole
  tenure after are included (~5,000 of ~104k candidate permanent transfers,
  once loans are excluded), which skews the training data toward
  established first-team players rather than fringe moves. Loans are
  covered separately (see `/loans.html`, ~3,200 of ~28k candidate loan
  spells) with a looser bar — only the pre-loan side needs ≥10
  appearances, not the loan itself.
- Loan detection depends on a one-time batch fetch from transfermarkt's
  live, unofficial `transferHistory` API (`scripts/fetch_transfer_types.py`)
  — an undocumented endpoint, not a published third-party API, so it isn't
  polled live and could break if transfermarkt changes it. About 98% of
  candidate transfers match a fetched record by (player, date, clubs); the
  rest (an unfetched player, or a rare club id the live API doesn't
  resolve) default to "unknown" and are treated like any other permanent
  transfer rather than being dropped, so a small number of undetected loans
  may still be scored as permanent moves.
- A loan with an option/obligation to buy that converts to a permanent deal
  at the same club, without a separate recorded transfer event for the
  conversion, is still scored as one continuous loan spell running through
  to whatever transfer comes next - the permanent phase isn't split out and
  scored on the permanent formula instead. A loan that sends the player
  onward to a second loan club before they return, by contrast, is handled
  correctly - each leg gets its own row and its own bounded window.
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
scripts/fetch_transfer_types.py  # (optional) backfills data/raw/transfer_types_cache.csv from transfermarkt's live API
scripts/fetch_fotmob_stats.py    # (optional) backfills data/raw/fotmob_stats_cache.csv from FotMob's public stat leaderboards
scripts/build_dataset.py   # raw Transfermarkt CSVs -> data/transfers_processed.csv + data/loans_processed.csv
scripts/train_model.py     # trains the model + comparable-transfers index
scripts/build_lookups.py   # small player/club search tables for the web app
app/main.py                 # FastAPI backend (serves the API + the static frontend)
app/static/                 # vanilla HTML/CSS/JS frontend (index/browse/loans/compare)
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
./.venv/bin/python scripts/fetch_transfer_types.py  # optional but recommended - see below
./.venv/bin/python scripts/fetch_fotmob_stats.py    # optional but recommended - see below
./.venv/bin/python scripts/build_dataset.py
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/train_model.py
```

`fetch_transfer_types.py` re-fetches every candidate player's real transfer
history from transfermarkt's live API to tell loans apart from free
transfers (see "Loans are detected..." above) - one request per player
(~23k), politely rate-limited, so it takes a few hours and is safe to
interrupt and re-run (it resumes from `data/raw/transfer_types_cache.csv`
rather than starting over). It's optional: skip it and `build_dataset.py`
still runs, just without loan detection - every zero-fee transfer
(including loans) is treated as a permanent transfer, and
`data/loans_processed.csv` comes out empty.

`fetch_fotmob_stats.py` pulls FotMob's season stat leaderboards for the 23
leagues in `LEAGUE_MAP` and stitches each transfer's tenure into
`data/raw/fotmob_stats_cache.csv` (see "The rating/attacking/defensive/
possession components come from FotMob..." above) - a few thousand
requests across ~14 seasons x 23 leagues, politely rate-limited, resumable
from `data/raw/fotmob_season_cache/` (gitignored - regenerable, not meant
to be committed) rather than refetching every league-season from scratch.
Also optional: skip it and `build_dataset.py` still runs, just without the
four FotMob components - each one's weight is dropped and the other
weights renormalized for every transfer, the same as when resale data is
unknown.

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
