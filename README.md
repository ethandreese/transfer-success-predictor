# Transfer Success Predictor ⚽

[![Tests](https://github.com/ethandreese/transfer-success-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/ethandreese/transfer-success-predictor/actions/workflows/tests.yml)

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
  "Haaland"). Whenever FotMob has an attacking-bucket score for the same
  transfer (the common case — see **attacking** below), this doesn't count
  as an *eleventh* independent signal on top of it: the two measure the
  same underlying thing, so they're folded into one combined bucket
  (`fold_perf_level_into_attacking` in `scripts/build_dataset.py`) instead
  of double-counting goal-based output. Performance level only stands
  alone, at its own weight, when FotMob has nothing for that transfer —
  older transfers and leagues FotMob's coverage doesn't reach that far
  back (see **Known limitations**) still get judged on it either way.
- **performance change** — improved or declined vs. their league-relative
  level before the move, also position-ranked
- **market value growth** — blends two different views of growth, each
  independently percentile-ranked against the whole dataset: growth
  *relative to the player's own pre-transfer value* (a cheap breakout
  signing wins big here — €5m to €20m is 4×) and the *absolute euro gain*
  (a marathon-sized fee can still win here on a comparatively modest ratio
  — €75m to €110m is "only" 1.47× but a real €35m paper gain). Ratio alone
  systematically buried already-expensive transfers: Moisés Caicedo's
  peak value at Chelsea (€75m → €110m, nearly recouping his world-record
  €116m fee) scored a mediocre 65 on ratio alone, since the bigger the
  starting value the harder it is to move the ratio at all — a signing
  the club could clearly sell at close to no loss was scoring the same as
  a middling outcome. Blending in the absolute-gain view puts it at 81.
  Each of those two views is itself an 80/20 blend of growth to the
  *peak* value reached during the tenure and growth to the value near
  the end of it, leaning heavily toward peak — end-value alone unfairly
  reads a long, valuable career as a decline, since even the best
  players' market value falls with age by the time they eventually leave
  (Heung-min Son joined Tottenham valued at ~€16m, peaked at €90m
  mid-tenure, and was worth only ~€20m a decade later when he left — his
  value_growth score is 93, not a mediocre one, because it's judged
  mostly on the €90m peak he reached, not the €20m he'd fallen to by the
  time he left), while peak alone would ignore a real late-tenure
  collapse (injury, loss of form). The heavy lean toward peak (rather
  than an even split) is also because end-of-tenure value already has
  its own dedicated signal elsewhere in the score - `resale_profit`, what
  the club actually realized when they sold the player, when that's
  known - so leaning value_growth itself more toward peak avoids doubly
  punishing a decline off a peak that resale_profit already accounts for
  on its own terms.
- **playing time** — blends two signals: 60% percent of the *team's actual
  games* played during the tenure (from `games.csv`/`club_games.csv` — the
  club's full match schedule across all competitions, not just games the
  player featured in), 40% raw appearance count. The percentage catches
  injuries/rotation that a raw count hides — Dembélé made 185 appearances
  for Barcelona over 6 years (a big number on its own), but that's only
  57% of the 327 games Barcelona actually played in that span, versus
  Haaland at 83% for Man City. Raw count is kept alongside it so a long,
  genuinely sustained career at the club still counts for something beyond
  the percentage alone. Weighted meaningfully higher for goalkeepers
  (17%, vs. 10-11% for outfielders) — a club fields exactly *one*
  starting keeper, so whether a signing actually won that job is an
  unusually clean, binary signal, unlike an outfield rotation slot where
  a squad player can have real value without ever being first-choice.
- **value for money** — performance level vs. what was paid *relative
  to the player's market value at the time* (comparing fees to the whole
  dataset's fee distribution made any nine-figure move look "expensive"
  even when it was a bargain for that specific player). Weighted flat
  10% for every position (see **Weights** below). Paying up to 1.3x a
  player's pre-transfer value counts as a completely normal premium and
  gets zero fee penalty — the dataset's own median fee/value ratio is
  0.62x and the 75th percentile is only 1.17x, so a modest premium is
  common, not some rare extreme, and clubs routinely pay one for
  transfers that still work out fine. Only the ~19% of transfers that
  actually exceed 1.3x get penalized at all, and only relative to *each
  other* — not the whole dataset, which would otherwise still let a fee
  just barely over the line land in roughly the same bad percentile as a
  genuinely enormous overpay (see `compute_fee_penalty_pct` in
  `scripts/build_dataset.py` for why a plain scaled/clipped ratio fed
  into one global percentile rank doesn't actually achieve this: rank
  only depends on relative order, so scaling or clipping the ratio
  before ranking leaves an above-threshold transfer's rank essentially
  unchanged — splitting into two separate populations is what actually
  makes the difference). The "performance" side of this comparison isn't
  just performance level, either — it's a weighted blend of *every*
  on-pitch quality signal already computed for the transfer (performance
  level, attacking, defensive, possession, rating), weighted the same
  way the rest of the score already weights them for that specific
  position/sub-position (see `value_for_money_performance_proxy` in
  `scripts/build_dataset.py`). Performance-level-alone started out fine
  for attackers but was a real problem for defense-oriented roles: goal
  contributions are meaningless for goalkeepers (82% have exactly 0 in
  the tenure window, all tied at the same percentile regardless of how
  well they actually played) and barely correlated with actual defensive
  quality even when nonzero — checked directly, goal contributions vs.
  the defensive component correlate at just 0.01 for centre-backs, and
  are *negative* for full-backs and defensive midfielders (-0.14 to
  -0.18). Using it as "performance" for those rows wasn't measuring
  performance at all; it silently collapsed value for money into "was
  the fee reasonable" alone. Blending in attacking/defensive/possession/
  rating, each weighted by how much that role's own score already leans
  on it, fixed that without hardcoding a list of positions — a
  Centre-Back's "performance" here leans on the defensive component the
  way a winger's leans on attacking/performance level, automatically,
  because that's how their weight profiles already differ. A component
  missing for a given row (no FotMob data for that bucket) just drops
  out of the blend and the rest are renormalized, same pattern used
  everywhere else in the score.
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
    (value growth, playing time, value for money). Weighted flat 20% for
    every position (see **Weights** below) — it's the same metric
    regardless of role, unlike attacking/defensive/possession, which are
    literally different stats depending on position.
  - **attacking** — goals, expected goals (xG), expected assists (xA), and
    chances created per 90, all averaged together, *then averaged again
    with performance level* whenever both are known for the transfer (see
    **performance level** above) — goal contributions already banked and
    the underlying attacking process (xG, chances created) are related
    enough that keeping them fully independent double-counted the same
    signal. Zero weight for goalkeepers.
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

Weights: attackers lean heavily on performance (16%/8%) since goal
contributions are a real, differentiating signal for them (only 5% have
zero goal contributions in a given window). That signal gets progressively
less reliable for midfielders (11% zero) and defenders (21% zero), and is
essentially meaningless for goalkeepers (nearly all have exactly 0 goals +
assists both before and after a move — no *Transfermarkt* column captures
clean sheets, saves, or defensive actions). So performance weight shrinks
from 24% combined (attackers) to 0% (goalkeepers), shifted into
attacking/defensive/possession instead, whose combined weight grows from
18% (attackers, a minor signal on top of real goal data) up to 42%
(goalkeepers, almost entirely defensive since shot-stopping is
essentially the job and attacking is zeroed out entirely) as goal
contributions become less meaningful.

**value_growth (15%), value_for_money (10%), and rating (20%) are flat
across every position, unlike everything else above.** value_growth and
value_for_money are each computed as a single percentile rank across the
*whole* dataset with no position grouping at all (unlike perf_level,
defensive, etc., which are ranked within position) — there's no
football-role reason for a defender's market-value growth or fee
justification to be weighted differently from an attacker's, since
nothing about how either number is computed treats positions
differently. rating is flat for a different reason: it's the same metric
(FotMob's overall per-match quality score) for every player regardless
of role, unlike attacking/defensive/possession, which are literally
different underlying stats depending on position — and, being an outside
rating service's own judgment of the player's *overall* performance
rather than one specific facet of it, it captures a lot a stat line
alone can't (positioning, decision-making, composure). All three used to
vary by position (value_growth 15/15/18/15%, value_for_money
20/22/22/25%, rating 2/3/4/7% for Attack/Midfield/Defender/Goalkeeper) —
that spread was really just leftover perf_level weight parked somewhere
convenient as goal contributions became less meaningful for a position,
not a deliberate choice, so it's been moved into
attacking/defensive/possession instead. value_for_money went through a
second cut after that: even flat, 20% was still a lot of weight riding
on one fee-vs-performance metric, so it's now flat 10% instead, with the
other 10% also routed into attacking/defensive/possession,
proportionally per position, same mechanism as the original flattening.
rating went the opposite direction, twice — first up to a flat 10%, then
up again to a flat 20%, on the reasoning above that a holistic per-match
rating deserves more than a token weight. Both rating moves are the one
exception to the attacking/defensive/possession routing: they come
entirely out of perf_level/perf_delta/playing_time instead (scaled down
proportionally, preserving their relative ratio to each other),
specifically so rating's growing weight wouldn't dilute the role-specific
tuning attacking/defensive/possession carry — see **Weights also vary by
sub-position** below for that tuning. A side effect worth knowing: for
positions where that funding bloc was already small (Centre-Back and
full-back especially), perf_level/perf_delta/playing_time are now down
to single-digit percentages each - William Saliba's Centre-Back weights,
for example, spend just 1%/1%/3% on them combined.

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
scored this way - see `/loans.html`. perf_level/perf_delta/value_growth/
playing_time are still ranked within the loans population only, not mixed
with permanent-transfer norms - a loan's much shorter window makes raw
appearance counts and value growth genuinely incomparable in scale to a
permanent tenure's. The four FotMob components (rating/attacking/
defensive/possession - see below) are the exception: they're ranked
against transfers and loans *combined* (see `attach_fotmob_components` in
`scripts/build_dataset.py`) - about 44% of loans end up with a usable
score in at least one bucket (lower than permanent transfers' 82%, since a
loan spell is usually a shorter tenure for a less-established player, so
it clears FotMob's per-category minutes thresholds less often), but a
given rating/tackles-per-90/etc. now means the same percentile whether
it's shown on a loan card or a permanent-transfer one.

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

**Each raw FotMob stat is league-adjusted before it's ranked, the same
idea `perf_level` already applies to goal contributions.** Playing style
genuinely differs by league in ways that show up directly in these raw
counts, independent of quality: centre-backs in Bundesliga/Ligue 1/
Denmark average measurably more tackles+interceptions+clearances+
recoveries per 90 than centre-backs in Premier League/La Liga (a ~20-point
gap in average `defensive_pct` before this was added — a more
transition-heavy, higher-turnover style of play generates more defensive
actions per player, it doesn't mean the players are better). Two real Arsenal
centre-backs made this concrete: Gabriel and William Saliba both scored
in the 30s on `defensive_pct` despite being well-regarded starters,
because every FotMob-derived stat was being ranked against centre-backs
in every league combined, with no adjustment for how many raw defensive
actions a given league's style tends to produce.
`compute_fotmob_league_baselines` in `scripts/build_dataset.py` fixes
this — for each (league, position) and each raw stat, it computes a
minutes-weighted average across the FotMob sample itself (not the full
`appearances.csv` the goal-contribution baseline uses, since these stats
only exist for the ~5,700 transfers/loans FotMob was matched to — well-
covered leagues like the Premier League get a stable baseline from 100+
matched centre-backs alone, thin ones fall back to the position-wide
average). Each row's raw stat is then compared against that baseline as
a plain difference ("N more/fewer than the league average"), not a ratio
like `perf_level` uses — several of these stats are negative by
construction (`fotmob_goals_conceded_inv`) or already a percentage
(`fotmob__save_percentage`), where a ratio's sign and scale get
confusing, while an offset works the same way for all of them. After
this fix, average `defensive_pct` across the top 12 leagues by matched
centre-back count clusters tightly around 50 (49–57, one 61 outlier in a
thin 23-player sample) instead of spanning 39–60.

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

**These four components are ranked against transfers and loans combined,
unlike everything else in either formula.** Every other percentile here -
performance level, value growth, playing time, resale profit for
transfers; the loan equivalents - is deliberately kept within its own
population, because the underlying raw numbers genuinely aren't on the
same scale: a loan's few months can't rack up the same raw appearance
count or market-value swing as a multi-year permanent tenure, so mixing
them would bias both directions. rating/attacking/defensive/possession
don't have that problem - they're per-90 rates (or, for rating, a plain
average), already normalized for how long the tenure was, so a loan
spell and a permanent tenure with genuinely identical on-pitch output
land on the same percentile rather than two different ones just because
of which pool happened to rank them. Combining also gives thinner slices
(goalkeepers especially) a bigger, more stable reference population than
either pool alone. `attach_fotmob_components` in `scripts/build_dataset.py`
does this: it merges FotMob stats onto both transfers and loans, then
runs the percentile ranking once across the combined set before handing
each its own slice back.

**Weights also vary by sub-position within each broad position, not just
by the four broad positions above.** A Defensive Midfielder and an
Attacking Midfielder both get the broad "Midfield" weights above by
default, which weight attacking output far too heavily for a player
whose job is mostly disruption and progression, not goals — Moisés
Caicedo (Brighton → Chelsea, a genuine defensive-midfield profile: 74th
percentile on defending, 68th on possession, but only 27th on attacking
output) scored a 45.1 under the broad Midfield weights despite that
being a clearly strong defensive-midfield tenure. `data/score_weights.json`'s
`_sub_positions` (and `data/loan_score_weights.json`'s equivalent) give
a handful of sub-positions their own weight row instead — Defensive
Midfield, Attacking Midfield, Left/Right Midfield, Centre-Back, Left/
Right-Back, Left/Right Winger — each shifting weight out of
perf_level/perf_delta/attacking (the same "attacking output" bloc
`fold_perf_level_into_attacking` already folds into one number) and into
defensive/possession for a defense-oriented role, or the other way for
an attack-oriented one. With the Defensive Midfield weights, Caicedo's
score becomes 64.8. Central Midfield, Centre-Forward, Second Striker,
and Goalkeeper have no override — either that already IS the broad
default's implicit profile, the position has no sub-split at all
(Goalkeeper), or the sample is too thin in this dataset for a confident
opinion (Second Striker: ~300 players total, before any transfer/
appearance filters even apply) — those fall back to the broad position's
row unchanged. **Percentile ranking itself is untouched by any of
this** — a Centre-Back is still ranked against every Defender, not just
other Centre-Backs, so the comparison population stays large and stable;
only the weights applied to an already-computed percentile differ by
sub-position.

**The sub-position used for this is not the one in the packaged
`players.csv`.** That column is a single, undated label — whatever
Transfermarkt currently lists for a player, the same value regardless of
which transfer or era is being scored, so a player who changed roles
over their career (central midfield early on, pushed into a more
defensive role later) gets every older transfer judged against today's
label instead of the one they actually held at the time. `game_lineups.csv`
(3.18M rows, one per (game, player), covering both `starting_lineup` and
`substitutes` rows) has the sub-position a player was actually fielded
in for every specific match, so `load_actual_sub_positions` in
`scripts/build_dataset.py` takes each player's single most-common
fielded sub-position across their entire lineup history and uses that
instead — still a career-wide summary, not a per-tenure one (a per-tenure
version would need to filter to each transfer's specific window, which
gets thin fast for short tenures and loans), but a real, dated one
rather than a today-only snapshot. The two disagree more than you'd
expect: for players with a real sample (≥10 lineup appearances),
`players.csv`'s label matches the position they were actually fielded in
most often only 74.7% of the time, and about a quarter of players never
settle into one dominant role at all (under 70% of their own lineup
appearances at their single most-common position). A player with no
`game_lineups.csv` rows at all falls back to `players.csv`'s own label.

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
performance including FotMob rating/xG/xA/passing/defensive output, and
origin/destination club & league strength) — nothing about what happened
after the move. Evaluated on a temporal holdout (trained on
transfers before mid-2023, tested on transfers since): **MAE ≈ 12.34 points**
on the 0–100 scale, R² ≈ 0.175, vs. ≈14.2 MAE for always predicting the
average. That's a modest but real signal, and honestly weaker than scoring
a fixed first year would give — predicting a player's *entire future stint*
at a new club from pre-transfer stats alone is genuinely hard, since
multi-year outcomes depend heavily on injuries, tactics, and squad fit that
no pre-transfer number can see. The app surfaces this error rate and a
per-prediction "why this score" breakdown rather than presenting the number
as gospel.

**Height is fed in relative to its own position's average, not as a raw
number.** Raw `height_in_cm` tested as a bigger model input than the
`position` category itself (3.4% feature importance vs. 1.7% for all four
position dummies combined), despite being far less informative on its
own — it was mostly acting as a silent proxy for position/body-type (tall
→ centre-back/striker/keeper) rather than earning its weight for the roles
height genuinely matters for (aerial duels). Centering it on its own
position's average (`height_vs_position` in `scripts/train_model.py`,
e.g. "+8cm" for a striker taller than the typical striker) tested as a
small but consistent improvement over both the raw value and dropping
height outright — MAE 13.20 → 13.15, R² 0.086 → 0.089 on the same
temporal holdout. A second idea tested alongside it — feeding in the
average `success_score` of each transfer's k nearest historical
neighbors, on the theory that "similar transfers tend to succeed/fail
together" should be a real signal — did *not* hold up: results bounced
non-monotonically with k (R² swung from 0.074 to 0.093 across k=5/15/30)
and got worse, not better, once combined with the height fix, on a
dataset this size (~3,000 training rows) too thin for that kind of
neighbor-average feature to add signal rather than noise. Kept the
height fix, dropped the neighbor-feature idea.

**The model itself was under-regularized for a training set this small,
and fixing that was a bigger win than any single feature.** The original
config (`max_depth=3`, `learning_rate=0.05`, every row and every leaf
size allowed) let individual trees fit noise in a ~3,000-row training
set. A grid search against the same temporal holdout found a shallower,
more constrained config — `max_depth=2`, `learning_rate=0.1`,
`subsample=0.8` (each tree only sees a random 80% of rows - stochastic
gradient boosting), `min_samples_leaf=10` — a real, seed-stable
improvement (R² 0.086 → ~0.11 across 5 random seeds, checked
specifically to rule out a lucky single run), not a one-off. Also tried
and rejected: `HistGradientBoostingRegressor` and `RandomForestRegressor`
as drop-in replacements (neither beat a well-tuned
`GradientBoostingRegressor`), and a smoothed destination/origin-club
historical-success-rate feature (target encoding), which looked
promising in isolation but made things *worse* once combined with the
other changes - a median of 5 transfers per club in the training data is
too thin for even a shrinkage-smoothed per-club average to add real
signal.

**`sub_position` (e.g. Centre-Back vs. Winger, not just the 4 broad
positions) is fed to the model as an additional category, alongside
`position` rather than instead of it.** The score-formula side of the
app already distinguishes these for weighting (see "Weights also vary by
sub-position" above); the predict model didn't have access to that
distinction at all before. Small additional gain on top of the
regularization fix: MAE 13.00 → 12.93, R² 0.112 → 0.114. Autofilled from
the searched player's own data (`players_lookup.csv` already carries it)
- no new form field needed. A player missing it entirely (2 of ~8,600 in
the lookup table) falls back to their broad position instead, the same
"fall back to the broad category" pattern already used for sub-position
weighting in the score formula.

**The predict model's pre-transfer performance signal was, until this
point, limited to Transfermarkt goal contributions alone (`pre_goals_p90`/
`pre_ga_p90`) - the same rating/xG/xA/passing/defensive-actions data the
historical score's post-transfer components already draw from was never
available on the pre-transfer side, simply because the model predates the
FotMob pipeline.** Filling that gap was the single biggest accuracy
improvement found on the predict model: **MAE 12.93 → 12.34, R² 0.114 →
0.175** on the same temporal holdout - checked directly against pure
model-seed noise (10 seeds, R² 0.161-0.175) to confirm it's a real,
stable gain, not a lucky run. Every individual bucket (rating, attacking,
possession, defensive) beat the without-FotMob baseline on its own, and
combining all of them kept helping - unlike most feature ideas tried
elsewhere in this project, nothing here needed to be walked back.

- `scripts/fetch_pretransfer_fotmob_stats.py` stitches each historical
  transfer's *pre*-transfer year at the *old* club - the mirror image of
  `scripts/fetch_fotmob_stats.py`'s post-transfer tenure stitching, reusing
  its season-fetch/cache and club-matching machinery unchanged. No new
  scraping was needed: it reads the same cached season leaderboards
  already on disk, since 99.7% of origin leagues fall within the same
  23-league set already covered for destinations (confirmed before
  writing a line of this).
- `scripts/fetch_current_fotmob_stats.py` does the equivalent for *live*
  predictions - a snapshot of each `players_lookup.csv` player's last 365
  days at their *current* club, autofilled into the predict form the same
  way `recent_apps`/`recent_goals_p90`/etc. already are, no new form
  field needed. Unlike the historical case there's no mid-season-move
  contamination to guard against - FotMob already attributes a season to
  whichever club a player is *currently* registered at, which is exactly
  the club this script wants.
- The predict model consumes these as **raw per-90 numbers**, not the
  percentile-ranked, league-baseline-adjusted components the historical
  score computes (`compute_fotmob_component_pcts`) - a tree ensemble can
  learn its own splits/thresholds directly, so that machinery (built to
  combine components onto one comparable 0-100 scale) isn't needed here.
- ~35-45% of transfers have no pre-transfer FotMob match (an uncovered
  origin league, or a real coverage gap - same ceilings as the
  post-transfer side). Missing values are median-imputed (fit-on-train
  for the holdout eval, full-dataset for the deployed model, same
  discipline as `height_vs_position`) rather than dropping those rows,
  alongside a `has_pre_fotmob_data` flag so the model can learn to
  discount an imputed placeholder instead of trusting it as real form.
- Two related bugs turned up and were fixed while wiring this in: (1) a
  GK-only stat (e.g. saves) has no real median at all for outfield
  positions, and the leave-one-out explanation swap was feeding that
  `NaN` straight into the model, crashing the request - fixed by falling
  back to the flat (non-position-conditional) reference whenever the
  position-specific one is missing; (2) `reference_values_by_position`
  was originally computed *after* the median-imputation step above,
  silently diluting each position's "typical" value with a chunk of
  imputed population-median rows rather than reflecting only the real
  observed ones - fixed by computing it from a pre-imputation snapshot
  instead (`median()` skips real `NaN` on its own, once nothing has
  overwritten it yet).

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

**A bare categorical swap ("Destination league: La Liga vs. a typical
transfer's Premier League") reads as "moving to Spain is inherently
better", with no hint of why — fixed for the features where that's
actually misleading.** Checked directly: Premier League genuinely is the
lowest-scoring major league in the historical data (48.4 average
`success_score` vs. La Liga's 50.2, France's 51.9), but on-pitch
components (attacking/defensive/possession/rating) are comparable across
leagues - the gap is almost entirely `value_for_money`. Premier League
clubs have historically paid a real, large fee premium over market value
(mean fee/value 1.55x, 47% of paid deals exceeding the formula's 1.3x
overpay line) vs. La Liga's 1.01x/21% - `value_for_money` is designed to
penalize exactly that, deliberately, so this isn't a bug in the score
(a club chronically overpaying is worse value even when the player
performs fine) - it's the *explanation* that was misleading by omitting
why. The "destination/origin league" explanation now appends the real
historical average and, for the destination league specifically, the fee
premium that drives it: *"Laliga vs. a typical transfer's Premier League
— transfers to Laliga have historically averaged 50.3 vs. 48.4 for
Premier League, largely reflecting fee premiums paid there (1.01x market
value on average vs. 1.55x)"* (`league_context_note` in `app/main.py`,
baselines computed in `scripts/train_model.py` and cached in
`metadata.json`). A league with too few transfers to trust a stable
average (`MIN_LEAGUE_SAMPLE = 15`) is left out of the baseline entirely
rather than shown a noisy number - the explanation just falls back to
the plain swing-only version for those. `fee_to_value_ratio` and
`club_quality_ratio` got the same treatment on a smaller scale - a short
appended clause explaining what the ratio means and, for
`fee_to_value_ratio`, that the historical scoring only penalizes fees
above ~1.3x market value, not any premium at all. `height_vs_position`'s
explanation was also cleaned up - it used to read "+8cm vs. a typical
transfer's +0cm" (technically correct but redundant, since the reference
is 0 by construction), now reads "+8cm vs. the position average".

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
  FotMob-derived components for both permanent transfers and loans (see
  above), but not fully: they only cover the 23 leagues FotMob was matched
  against (loans into a handful of other leagues - Brazil, MLS, Saudi
  Arabia, Argentina, and a few smaller ones, ~3.5% of loans - aren't in
  that set at all), only from whatever season FotMob's own coverage
  happens to start for that specific league, and not at all for Ukraine's
  Premier League. About 18% of scored permanent transfers (56% of loans)
  still have no FotMob data in any of the four buckets and fall back to
  value growth, playing time, and value for money (plus perf_level/perf_delta
  for the permanent score) carrying the position almost entirely, as
  before - and even among covered transfers, individual buckets have
  uneven coverage (rating 70%, attacking 81%, defensive 72%, possession
  72% for permanent transfers), so it's common for a transfer or loan to
  show some but not all four rows.
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
- Sub-position weighting (see above) uses each player's single most-common
  fielded sub-position across their *entire* career in `game_lineups.csv`,
  not the specific window of any one transfer's tenure — a player who
  genuinely changed roles mid-career (moved from a wide role into central
  midfield, say) has every one of their transfers weighted by whichever
  role dominates their overall history, which may not be the role they
  actually played during an older or shorter tenure. A small number of
  players' most-common lineup entry is a generic legacy label
  ("Midfield", "Attack", "Defender", not the granular sub-position
  vocabulary) rather than a real sub-position — those fall back to the
  broad position's weights like any other unmapped value, same as a
  player with no `game_lineups.csv` rows at all.
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
scripts/fetch_transfer_types.py           # (optional) backfills data/raw/transfer_types_cache.csv from transfermarkt's live API
scripts/fetch_fotmob_stats.py             # (optional) backfills data/raw/fotmob_stats_cache.csv from FotMob's public stat leaderboards
scripts/fetch_pretransfer_fotmob_stats.py # (optional) backfills data/raw/pretransfer_fotmob_stats_cache.csv - the predict model's pre-transfer FotMob signal
scripts/fetch_current_fotmob_stats.py     # (optional) backfills data/raw/current_fotmob_stats_cache.csv - live predictions' "current form" FotMob signal
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
./.venv/bin/python scripts/fetch_transfer_types.py           # optional but recommended - see below
./.venv/bin/python scripts/fetch_fotmob_stats.py              # optional but recommended - see below
./.venv/bin/python scripts/fetch_pretransfer_fotmob_stats.py  # optional but recommended - see below
./.venv/bin/python scripts/build_dataset.py
./.venv/bin/python scripts/build_lookups.py
./.venv/bin/python scripts/fetch_current_fotmob_stats.py      # optional but recommended - see below
./.venv/bin/python scripts/build_lookups.py                   # run again to merge in the current-FotMob snapshot just fetched
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
leagues in `LEAGUE_MAP` and stitches each tenure - both permanent
transfers and loan spells, in one pass - into
`data/raw/fotmob_stats_cache.csv` (see "The rating/attacking/defensive/
possession components come from FotMob..." above) - a few thousand
requests across ~14 seasons x 23 leagues, politely rate-limited, resumable
from `data/raw/fotmob_season_cache/` (gitignored - regenerable, not meant
to be committed) rather than refetching every league-season from scratch.
Also optional: skip it and `build_dataset.py` still runs, just without the
four FotMob components - each one's weight is dropped and the other
weights renormalized for every transfer, the same as when resale data is
unknown.

`fetch_pretransfer_fotmob_stats.py` and `fetch_current_fotmob_stats.py`
feed the *predict model* (not the historical score) - see "The predict
model's pre-transfer performance signal..." above. Both reuse
`fetch_fotmob_stats.py`'s season-fetch/cache and matching functions
directly (imported, not duplicated), so if that cache is already warm
these run in seconds, not hours - no separate scrape needed. Both are
optional the same way: skip either and `train_model.py`/`build_lookups.py`
still run, just with every `pre_fotmob_*` feature (or every searched
player's `recent_fotmob_*` autofill) falling back to the median/`None` a
missing match already degrades to.

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
