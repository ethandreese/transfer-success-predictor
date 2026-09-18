/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/**
 * Map a searched player's recent_fotmob_* fields (players_lookup.csv, via
 * /api/players/search) onto PredictRequest's pre_fotmob_* fields. null
 * (no recent FotMob match for that player/stat - an uncovered league, or
 * a real coverage gap) passes straight through as JSON null, which
 * build_feature_row on the backend fills in with the trained median.
 */
function pretransferFotmobFeatures(player) {
  return {
    pre_fotmob_rating: player.recent_fotmob_rating,
    pre_fotmob_expected_goals_per_90: player.recent_fotmob_expected_goals_per_90,
    pre_fotmob_expected_assists_per_90: player.recent_fotmob_expected_assists_per_90,
    pre_fotmob_chances_created_p90: player.recent_fotmob_chances_created_p90,
    pre_fotmob_accurate_pass: player.recent_fotmob_accurate_pass,
    pre_fotmob_won_contest: player.recent_fotmob_won_contest,
    pre_fotmob_total_tackle: player.recent_fotmob_total_tackle,
    pre_fotmob_interception: player.recent_fotmob_interception,
    pre_fotmob_effective_clearance: player.recent_fotmob_effective_clearance,
    pre_fotmob_ball_recovery: player.recent_fotmob_ball_recovery,
    pre_fotmob_saves: player.recent_fotmob_saves,
    pre_fotmob__save_percentage: player.recent_fotmob__save_percentage,
    pre_fotmob_goals_conceded: player.recent_fotmob_goals_conceded,
  };
}

// Up to 4 scenario columns - "a"/"b" are always present, "c"/"d" start
// hidden and are revealed by "+ Add another option" (see wireAddRemove()
// near the end of this file).
const SCENARIO_KEYS = ["a", "b", "c", "d"];

/** The scenario keys whose column is currently visible, in a-b-c-d order - the single source of truth for "how many-way is this comparison right now" (no separate count/list kept in state to drift out of sync with the DOM). */
function activeScenarioKeys() {
  return SCENARIO_KEYS.filter(k => !document.querySelector(`.compare-col[data-scenario="${k}"]`).hidden);
}

// Per-scenario selection state: the chosen player, that player's *current*
// club (fetched separately, for origin-league/value context), and the
// chosen destination club. Populated for all 4 keys up front (see
// setupScenario calls near the end of this file) regardless of which
// columns are currently visible - activeScenarioKeys() above is what
// actually decides which of these are used in a comparison.
const scenarios = {};

/**
 * Wire up one compare-column's player/club autocompletes and keep
 * `scenarios[key]` in sync with the selections, enabling the Compare
 * button once both columns have a player and a club chosen.
 */
function setupScenario(key) {
  const col = document.querySelector(`.compare-col[data-scenario="${key}"]`);
  // feeEurMillions is the fee input's "real" value in EUR millions - the
  // unit the backend always expects - kept separate from whatever the
  // input currently *displays* (€/$/£) so switching currency redisplays
  // the same real fee rather than reinterpreting the same digits as a
  // different amount (see app.js's identical state.feeEurMillions).
  const state = { player: null, playerClub: null, club: null, feeEurMillions: 50 };
  scenarios[key] = state;

  const playerInput = col.querySelector(".player-search");
  const playerList = col.querySelector(".player-list");
  const clubInput = col.querySelector(".club-search");
  const clubList = col.querySelector(".club-list");

  /** Enable the Compare button once every *visible* scenario column has both a player and a club selected - a hidden (not-yet-added, or removed) column's state is irrelevant. */
  function updateCompareButton() {
    const ready = activeScenarioKeys().every(k => scenarios[k].player && scenarios[k].club);
    document.getElementById("compare-btn").disabled = !ready;
  }

  /**
   * Debounced search-as-you-type dropdown for one input, scoped to this
   * scenario's column (same pattern as app.js's setupAutocomplete,
   * duplicated here since each compare column needs its own independent
   * instance). `filterResults` (optional) can drop items from the raw API
   * response before they're shown.
   *
   * Keyboard-navigable (ArrowUp/Down, Enter, Escape) with the standard
   * ARIA combobox pattern (role="combobox"/"listbox"/"option",
   * aria-activedescendant) - see app.js's setupAutocomplete for why this
   * matters (previously mouse/click only, so a keyboard-only or screen-
   * reader user couldn't select a player or club here at all). Uses
   * `list.id` (set in compare.html) to build unique option ids, since this
   * function is keyed by element reference, not a single page-wide id.
   */
  function wireAutocomplete(input, list, endpoint, renderLabel, onSelect, filterResults) {
    let debounceTimer = null;
    let items = [];
    let highlighted = -1;

    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.setAttribute("aria-controls", list.id);
    list.setAttribute("role", "listbox");

    function closeList() {
      list.classList.remove("open");
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      highlighted = -1;
    }

    function setHighlight(i) {
      highlighted = i;
      [...list.children].forEach((child, idx) => child.classList.toggle("highlighted", idx === i));
      if (i >= 0) {
        input.setAttribute("aria-activedescendant", `${list.id}-opt-${i}`);
        list.children[i].scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }

    function selectItem(i) {
      const item = items[i];
      if (!item) return;
      onSelect(item);
      input.value = renderLabel(item);
      closeList();
    }

    input.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      const q = input.value.trim();
      if (q.length < 2) {
        items = [];
        closeList();
        return;
      }
      debounceTimer = setTimeout(async () => {
        const res = await fetch(`${endpoint}?q=${encodeURIComponent(q)}`);
        const results = await res.json();
        items = filterResults ? filterResults(results) : results;
        if (!items.length) {
          closeList();
          return;
        }
        list.innerHTML = items.map((item, i) =>
          `<div id="${list.id}-opt-${i}" role="option" data-idx="${i}">${renderLabel(item)}</div>`
        ).join("");
        list.classList.add("open");
        input.setAttribute("aria-expanded", "true");
        setHighlight(-1);
        [...list.children].forEach((child, i) => {
          child.addEventListener("click", () => selectItem(i));
          child.addEventListener("mouseenter", () => setHighlight(i));
        });
      }, 200);
    });

    input.addEventListener("keydown", (e) => {
      if (!list.classList.contains("open")) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlight(Math.min(highlighted + 1, items.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlight(Math.max(highlighted - 1, 0));
      } else if (e.key === "Enter" && highlighted >= 0) {
        e.preventDefault();
        selectItem(highlighted);
      } else if (e.key === "Escape") {
        closeList();
      }
    });

    document.addEventListener("click", (e) => {
      if (e.target !== input) closeList();
    });
  }

  wireAutocomplete(
    playerInput, playerList, "/api/players/search",
    (p) => `${p.name} (${p.position}, ${p.current_club_name})`,
    async (p) => {
      state.player = p;
      col.querySelector(".player-chip").innerHTML =
        `<span class="selected-chip">${p.name} &middot; age ${Number(p.age_now).toFixed(1)} &middot; ${p.current_club_name}</span>`;
      col.querySelector(".age-input").value = Number(p.age_now).toFixed(1);
      if (p.current_club_id) {
        const res = await fetch(`/api/clubs/${p.current_club_id}`);
        state.playerClub = res.ok ? await res.json() : null;
      }
      updateCompareButton();
    },
  );

  // Excludes the selected player's own current club - a "transfer" to the
  // club a player is already at isn't a real scenario, and the model has
  // no way to flag that for you (see app.js's identical guard).
  wireAutocomplete(
    clubInput, clubList, "/api/clubs/search",
    (c) => c.name,
    (c) => {
      state.club = c;
      col.querySelector(".club-chip").innerHTML = `<span class="selected-chip">${c.name}</span>`;
      updateCompareButton();
    },
    (clubs) => state.player ? clubs.filter(c => c.club_id !== state.player.current_club_id) : clubs,
  );

  // Keep the fee field's label/displayed value in sync with the selected
  // currency - see the state.feeEurMillions comment above and app.js's
  // identical updateFeeCurrencyDisplay for the full reasoning.
  const feeLabel = col.querySelector('label[for="fee-input-' + key + '"]');
  const feeInput = col.querySelector(".fee-input");
  function updateFeeCurrencyDisplay() {
    feeLabel.textContent = `Transfer fee (${CURRENCY_SYMBOLS[settings.currency]}m)`;
    feeInput.value = Math.round(state.feeEurMillions * EXCHANGE_RATES[settings.currency] * 10) / 10;
  }
  updateFeeCurrencyDisplay();
  feeInput.addEventListener("input", () => {
    const typed = parseFloat(feeInput.value);
    state.feeEurMillions = Number.isNaN(typed) ? 0 : typed / EXCHANGE_RATES[settings.currency];
  });
  document.addEventListener("settingschange", updateFeeCurrencyDisplay);

  /**
   * Look up a player/club pair by id for a shared/bookmarked comparison
   * URL (see restoreFromURL() below), without touching any state or DOM -
   * deliberately split from applyScenario() below so restoreFromURL() can
   * resolve *both* columns first and only apply either one once it knows
   * both actually resolved. Applying this column's fields the moment they
   * arrive, one column at a time, would leave a stale/mistyped link's
   * failing column blank while the other one looks perfectly normal -
   * confusing on its own, and worse once a "why didn't Compare run"
   * question has no visible cause.
   */
  async function fetchScenario(playerId, clubId) {
    const [player, club] = await Promise.all([
      fetch(`/api/players/${playerId}`).then(r => r.ok ? r.json() : null),
      fetch(`/api/clubs/${clubId}`).then(r => r.ok ? r.json() : null),
    ]);
    return player && club ? { player, club } : null;
  }

  /** Apply an already-resolved {player, club} (see fetchScenario above) to this column's state and DOM - exactly what selecting them via the autocompletes would have done, plus the fee/age fields. */
  async function applyScenario({ player, club }, feeEurMillions, age) {
    state.player = player;
    playerInput.value = `${player.name} (${player.position}, ${player.current_club_name})`;
    col.querySelector(".player-chip").innerHTML =
      `<span class="selected-chip">${player.name} &middot; age ${Number(player.age_now).toFixed(1)} &middot; ${player.current_club_name}</span>`;
    if (player.current_club_id) {
      const res = await fetch(`/api/clubs/${player.current_club_id}`);
      state.playerClub = res.ok ? await res.json() : null;
    }

    state.club = club;
    clubInput.value = club.name;
    col.querySelector(".club-chip").innerHTML = `<span class="selected-chip">${club.name}</span>`;

    state.feeEurMillions = feeEurMillions;
    updateFeeCurrencyDisplay();
    col.querySelector(".age-input").value = age.toFixed(1);
    updateCompareButton();
  }

  return { fetchScenario, applyScenario };
}

/** Assemble a PredictRequest body for one scenario from its selected player/club and the fee/age fields in that column. */
function buildPayload(key) {
  const col = document.querySelector(`.compare-col[data-scenario="${key}"]`);
  const state = scenarios[key];
  const fromClub = state.playerClub || {
    domestic_competition_id: state.player.current_club_domestic_competition_id,
    club_value_proxy: 0,
  };
  return {
    age_at_transfer: parseFloat(col.querySelector(".age-input").value) || state.player.age_now,
    height_in_cm: state.player.height_in_cm,
    position: state.player.position,
    sub_position: state.player.sub_position || state.player.position,
    foot: state.player.foot || "unknown",
    pre_apps: state.player.recent_apps,
    pre_minutes: state.player.recent_minutes,
    pre_goals_p90: state.player.recent_goals_p90,
    pre_ga_p90: state.player.recent_ga_p90,
    pre_mins_per_app: state.player.recent_mins_per_app,
    transfer_fee: state.feeEurMillions * 1_000_000,
    value_before: state.player.market_value_in_eur,
    from_domestic_competition_id: fromClub.domestic_competition_id || "unknown",
    to_domestic_competition_id: state.club.domestic_competition_id || "unknown",
    from_total_market_value: fromClub.club_value_proxy || 1,
    to_total_market_value: state.club.club_value_proxy || 1,
    ...pretransferFotmobFeatures(state.player),
  };
}

/** One result's "Top factors" explanation list as HTML - same explain-row markup every result column uses, factored out since renderCompareResult() below now builds N of these instead of a fixed "a"/"b" pair. */
function renderExplanation(explanation) {
  return explanation.slice(0, 3).map(e => {
    const positive = e.contribution >= 0;
    const width = Math.min(Math.abs(e.contribution) * 4, 100);
    return `
      <div class="explain-row">
        <span class="tooltip-wrap explain-label">
          ${e.label}
          <span class="tooltip-box">
            ${convertMoneyInText(e.detail)}
            ${e.stats ? `<ul class="tooltip-stats">${e.stats.map(s => `<li>${convertMoneyInText(s)}</li>`).join("")}</ul>` : ""}
          </span>
        </span>
        <div class="explain-bar-track">
          <div class="explain-bar-fill ${positive ? "pos" : "neg"}" style="width:${width}%"></div>
        </div>
        <span class="explain-value">${positive ? "+" : ""}${e.contribution}</span>
      </div>
    `;
  }).join("");
}

/**
 * A plain-English verdict for a finished comparison's `results` (2-4 of
 * them, in scenario order - not necessarily score order). Names the top
 * scorer and, if it's not basically tied with the runner-up, how far
 * ahead it is - the same "toss-up" framing the original two-scenario-only
 * version used, generalized past a single pairwise delta (which doesn't
 * mean much once there are 3+ scores to place).
 */
function verdictSentence(results) {
  const ranked = [...results].sort((a, b) => b.success_score - a.success_score);
  const [top, second] = ranked;
  const gap = top.success_score - second.success_score;
  if (gap < 3) {
    return results.length === 2
      ? "These two scenarios score within a few points of each other, roughly a toss-up given the model's error margin."
      : `${top.label} scores highest at ${top.success_score}, but within a few points of ${second.label}, roughly a toss-up given the model's error margin.`;
  }
  return results.length === 2
    ? `${top.label} scores ${gap.toFixed(1)} points higher than ${second.label}.`
    : `${top.label} scores highest at ${top.success_score}, ${gap.toFixed(1)} points ahead of the next best (${second.label}).`;
}

/** Render a /api/compare response: every scenario's result card (score, range, top factors - highest score highlighted) plus a plain-English verdict. Factored out so a settings change (currency) can re-render the last comparison without re-comparing. */
function renderCompareResult(data) {
  document.getElementById("result").classList.add("open");
  const results = data.results;
  const topScore = Math.max(...results.map(r => r.success_score));
  document.getElementById("compare-results-grid").innerHTML = results.map(r => `
    <div class="compare-result-col ${r.success_score === topScore ? "is-winner" : ""}">
      <h2>${r.label}</h2>
      <div class="score-display">
        <div class="big" style="color:${scoreColor(r.success_score)}">${r.success_score}</div>
        <div class="score-label">/ 100</div>
      </div>
      <div class="score-range-note">Likely range: ${r.score_range[0]}–${r.score_range[1]}</div>
      <div class="comparables">
        <h3>Top factors</h3>
        <div>${renderExplanation(r.explanation)}</div>
      </div>
    </div>
  `).join("");
  renderFactorComparison(results);
  document.getElementById("compare-delta").textContent = verdictSentence(results);
}

/**
 * A per-factor comparison table across every active scenario - unlike
 * each scenario's own "Top factors" list (renderExplanation, top 3 by
 * that scenario's own ranking), this shows the *same* set of factors for
 * every scenario side by side, so a reader can see not just that one
 * option wins overall but which specific factors actually carry it.
 * Requires every scenario's explanation to cover the same factor set -
 * true here since /api/compare requests top_k=25 (comfortably more than
 * the ~17-20 factors that actually exist) for exactly this reason. Rows
 * are ranked by spread (max contribution minus min, across scenarios) -
 * the most differentiating factors first, not whatever order one
 * scenario's own ranking happened to produce. The winning cell per row
 * (highest contribution - most favorable for that scenario specifically)
 * reuses .club-compare-table/.is-winner-cell, the same restrained
 * accent-text treatment Club Report Cards' own head-to-head comparison
 * already uses for the same idea.
 */
function renderFactorComparison(results) {
  const byFeature = results.map(r => {
    const map = {};
    r.explanation.forEach(e => { map[e.feature] = e; });
    return map;
  });

  const rows = results[0].explanation
    .map(({ feature, label }) => {
      const cells = byFeature.map(m => m[feature]);
      const contributions = cells.map(c => c.contribution);
      return { label, cells, spread: Math.max(...contributions) - Math.min(...contributions) };
    })
    .sort((a, b) => b.spread - a.spread)
    .map(({ label, cells }) => {
      const maxContribution = Math.max(...cells.map(c => c.contribution));
      const tds = cells.map(c => {
        const positive = c.contribution >= 0;
        return `<td class="${c.contribution === maxContribution ? "is-winner-cell" : ""}" style="color:${positive ? "var(--accent)" : "var(--accent-bad)"}">${positive ? "+" : ""}${c.contribution}</td>`;
      }).join("");
      return `<tr><td>${label}</td>${tds}</tr>`;
    })
    .join("");

  document.getElementById("factor-compare-wrap").innerHTML = `
    <h2>Compare by factor</h2>
    <p class="surprises-intro">Every factor behind each option's score, most differentiating first. The highlighted cell is whichever option that specific factor favors most.</p>
    <table class="club-compare-table">
      <thead><tr><th></th>${results.map(r => `<th>${r.label}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

let lastCompareData = null;
// The scenario keys a comparison was actually run with, so syncURL() (and
// a later settingschange re-render) always reflect that run - not
// whatever happens to be visible/filled-in *now*, which can drift after
// the fact (e.g. the user adds a 3rd option but hasn't re-compared yet).
let lastCompareKeys = [];

/** Keep the address bar's query string in sync with every scenario that was actually in the last comparison (player/club/fee/age each) - see writeURLParams in settings.js. Rebuilding the whole query string from lastCompareKeys means a comparison that drops back from 3-way to 2-way also drops the stale 3rd scenario's params, not just adds/overwrites the active ones. */
function syncURL() {
  const params = {};
  lastCompareKeys.forEach(k => {
    params[`player_${k}`] = scenarios[k].player.player_id;
    params[`club_${k}`] = scenarios[k].club.club_id;
    params[`fee_${k}`] = scenarios[k].feeEurMillions;
    params[`age_${k}`] = document.querySelector(`.compare-col[data-scenario="${k}"] .age-input`).value;
  });
  writeURLParams(params);
}

/** Build every active scenario's payload, POST them together to /api/compare, and render the results plus a plain-English verdict. Used both by the "Compare" button and restoreFromURL()'s auto-run of a shared link. */
async function runCompare() {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = "";
  const keys = activeScenarioKeys();
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenarios: keys.map(k => ({
          request: buildPayload(k),
          label: scenarios[k].player.name + " → " + scenarios[k].club.name,
        })),
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Comparison failed");
    }
    lastCompareData = await res.json();
    lastCompareKeys = keys;
    renderCompareResult(lastCompareData);
    syncURL();
  } catch (e) {
    errorBox.textContent = e.message;
  }
}

document.getElementById("compare-btn").addEventListener("click", runCompare);

// A settings change (currency, ...) doesn't change the underlying data,
// just how it's displayed - re-render the last comparison from the cached
// response rather than re-comparing, if one is already showing.
document.addEventListener("settingschange", () => {
  if (lastCompareData) renderCompareResult(lastCompareData);
});

const scenarioControllers = Object.fromEntries(SCENARIO_KEYS.map(k => [k, setupScenario(k)]));

/**
 * Wire the "+ Add another option"/"×" controls that reveal/hide the c/d
 * columns. A revealed column starts completely blank (setupScenario's
 * initial state, never touched) - nothing to restore, since c/d are only
 * ever reached by a deliberate click here, never pre-filled. Removing a
 * column resets its scenario state and every visible field/chip too, so
 * a later "+ Add" doesn't resurrect stale data in a column that looks
 * freshly added.
 */
function wireAddRemove() {
  const addBtn = document.getElementById("add-option-btn");

  function updateAddButtonVisibility() {
    addBtn.hidden = activeScenarioKeys().length >= SCENARIO_KEYS.length;
  }

  addBtn.addEventListener("click", () => {
    const nextKey = SCENARIO_KEYS.find(k => document.querySelector(`.compare-col[data-scenario="${k}"]`).hidden);
    if (!nextKey) return;
    document.querySelector(`.compare-col[data-scenario="${nextKey}"]`).hidden = false;
    updateAddButtonVisibility();
    document.getElementById("compare-btn").disabled = true; // the new column has no player/club yet
  });

  ["c", "d"].forEach(key => {
    document.querySelector(`.remove-option-btn[data-scenario="${key}"]`).addEventListener("click", () => {
      const col = document.querySelector(`.compare-col[data-scenario="${key}"]`);
      col.hidden = true;
      scenarios[key] = { player: null, playerClub: null, club: null, feeEurMillions: 50 };
      col.querySelector(".player-search").value = "";
      col.querySelector(".player-chip").innerHTML = "";
      col.querySelector(".club-search").value = "";
      col.querySelector(".club-chip").innerHTML = "";
      col.querySelector(".age-input").value = "";
      updateAddButtonVisibility();
      document.getElementById("compare-btn").disabled = !activeScenarioKeys().every(k => scenarios[k].player && scenarios[k].club);
    });
  });

  updateAddButtonVisibility();
}

wireAddRemove();

/**
 * Restore every scenario a shared/bookmarked comparison URL specifies
 * (2-4 of them - player_a/club_a/fee_a/age_a required, player_c.../
 * player_d... each optional) and, once all resolve, run the comparison
 * automatically - a shared link's whole point is showing the comparison
 * immediately, not making the recipient re-pick every player/club and
 * click Compare themselves. A c/d column the URL specifies is revealed
 * before it's used, same as a manual "+ Add another option" click would.
 *
 * Resolves every column's player+club lookup first (fetchScenario, no DOM
 * writes) before applying any of them (applyScenario) - a stale/mistyped
 * link where even one id no longer resolves (e.g. after a data refresh)
 * must leave *every* column untouched, not populate the ones that
 * happened to succeed while another sits blank with no visible reason
 * Compare never ran. Reveals the needed c/d columns up front (so a
 * failed restore's rollback has something concrete to hide again) but
 * only actually shows player/club data in them once every lookup across
 * the whole comparison has succeeded.
 */
async function restoreFromURL() {
  const params = readURLParams();
  const keys = SCENARIO_KEYS.filter(k => params[`player_${k}`] && params[`club_${k}`] && params[`fee_${k}`] && params[`age_${k}`]);
  if (keys.length < 2) return;

  const revealedNow = keys.filter(k => (k === "c" || k === "d") && document.querySelector(`.compare-col[data-scenario="${k}"]`).hidden);
  revealedNow.forEach(k => { document.querySelector(`.compare-col[data-scenario="${k}"]`).hidden = false; });

  const fetched = await Promise.all(keys.map(k => scenarioControllers[k].fetchScenario(params[`player_${k}`], params[`club_${k}`])));
  if (fetched.some(f => !f)) {
    revealedNow.forEach(k => { document.querySelector(`.compare-col[data-scenario="${k}"]`).hidden = true; });
    return;
  }

  await Promise.all(keys.map((k, i) => scenarioControllers[k].applyScenario(fetched[i], parseFloat(params[`fee_${k}`]), parseFloat(params[`age_${k}`]))));
  document.getElementById("add-option-btn").hidden = activeScenarioKeys().length >= SCENARIO_KEYS.length;
  runCompare();
}

restoreFromURL();
