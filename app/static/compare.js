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

// Per-scenario ("a"/"b") selection state: the chosen player, that player's
// *current* club (fetched separately, for origin-league/value context),
// and the chosen destination club.
const scenarios = {};

/**
 * Wire up one compare-column's player/club autocompletes and keep
 * `scenarios[key]` in sync with the selections, enabling the Compare
 * button once both columns have a player and a club chosen.
 */
function setupScenario(key) {
  const col = document.querySelector(`.compare-col[data-scenario="${key}"]`);
  const state = { player: null, playerClub: null, club: null };
  scenarios[key] = state;

  const playerInput = col.querySelector(".player-search");
  const playerList = col.querySelector(".player-list");
  const clubInput = col.querySelector(".club-search");
  const clubList = col.querySelector(".club-list");

  /** Enable the Compare button once every scenario has both a player and a club selected. */
  function updateCompareButton() {
    const ready = Object.values(scenarios).every(s => s.player && s.club);
    document.getElementById("compare-btn").disabled = !ready;
  }

  /** Debounced search-as-you-type dropdown for one input, scoped to this scenario's column (same pattern as app.js's setupAutocomplete, duplicated here since each compare column needs its own independent instance). */
  function wireAutocomplete(input, list, endpoint, renderLabel, onSelect) {
    let debounceTimer = null;
    input.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      const q = input.value.trim();
      if (q.length < 2) {
        list.classList.remove("open");
        return;
      }
      debounceTimer = setTimeout(async () => {
        const res = await fetch(`${endpoint}?q=${encodeURIComponent(q)}`);
        const items = await res.json();
        if (!items.length) {
          list.classList.remove("open");
          return;
        }
        list.innerHTML = items.map((item, i) => `<div data-idx="${i}">${renderLabel(item)}</div>`).join("");
        list.classList.add("open");
        [...list.children].forEach((child, i) => {
          child.addEventListener("click", () => {
            onSelect(items[i]);
            list.classList.remove("open");
            input.value = renderLabel(items[i]);
          });
        });
      }, 200);
    });
    document.addEventListener("click", (e) => {
      if (e.target !== input) list.classList.remove("open");
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

  wireAutocomplete(
    clubInput, clubList, "/api/clubs/search",
    (c) => c.name,
    (c) => {
      state.club = c;
      col.querySelector(".club-chip").innerHTML = `<span class="selected-chip">${c.name}</span>`;
      updateCompareButton();
    },
  );
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
    transfer_fee: (parseFloat(col.querySelector(".fee-input").value) || 0) * 1_000_000,
    value_before: state.player.market_value_in_eur,
    from_domestic_competition_id: fromClub.domestic_competition_id || "unknown",
    to_domestic_competition_id: state.club.domestic_competition_id || "unknown",
    from_total_market_value: fromClub.club_value_proxy || 1,
    to_total_market_value: state.club.club_value_proxy || 1,
    ...pretransferFotmobFeatures(state.player),
  };
}

/** Render one scenario's predict() result (score, range, top-3 explanation) into the "a" or "b" result column, per `suffix`. */
function renderResult(suffix, result) {
  const scoreEl = document.getElementById(`score-value-${suffix}`);
  scoreEl.textContent = result.success_score;
  scoreEl.style.color = scoreColor(result.success_score);
  const [lo, hi] = result.score_range;
  document.getElementById(`score-range-note-${suffix}`).textContent = `Likely range: ${lo}–${hi}`;
  document.getElementById(`explanation-list-${suffix}`).innerHTML = result.explanation.slice(0, 3).map(e => {
    const positive = e.contribution >= 0;
    const width = Math.min(Math.abs(e.contribution) * 4, 100);
    return `
      <div class="explain-row">
        <span class="tooltip-wrap explain-label">
          ${e.label}
          <span class="tooltip-box">${convertMoneyInText(e.detail)}</span>
        </span>
        <div class="explain-bar-track">
          <div class="explain-bar-fill ${positive ? "pos" : "neg"}" style="width:${width}%"></div>
        </div>
        <span class="explain-value">${positive ? "+" : ""}${e.contribution}</span>
      </div>
    `;
  }).join("");
}

/** Render a /api/compare response: both scenario results plus the plain-English delta summary. Factored out so a settings change (currency) can re-render the last comparison without re-comparing. */
function renderCompareResult(data) {
  document.getElementById("result").classList.add("open");
  renderResult("a", data.a);
  renderResult("b", data.b);
  const delta = data.delta;
  const deltaEl = document.getElementById("compare-delta");
  if (Math.abs(delta) < 3) {
    deltaEl.textContent = "These two scenarios score within a few points of each other — roughly a toss-up given the model's error margin.";
  } else if (delta > 0) {
    deltaEl.textContent = `Option A scores ${delta.toFixed(1)} points higher than Option B.`;
  } else {
    deltaEl.textContent = `Option B scores ${Math.abs(delta).toFixed(1)} points higher than Option A.`;
  }
}

let lastCompareData = null;

// Build both scenarios' payloads, POST them together to /api/compare, and
// render both results plus a plain-English delta summary.
document.getElementById("compare-btn").addEventListener("click", async () => {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = "";
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        a: buildPayload("a"),
        b: buildPayload("b"),
        label_a: scenarios.a.player.name + " → " + scenarios.a.club.name,
        label_b: scenarios.b.player.name + " → " + scenarios.b.club.name,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Comparison failed");
    }
    lastCompareData = await res.json();
    renderCompareResult(lastCompareData);
  } catch (e) {
    errorBox.textContent = e.message;
  }
});

// A settings change (currency, ...) doesn't change the underlying data,
// just how it's displayed - re-render the last comparison from the cached
// response rather than re-comparing, if one is already showing.
document.addEventListener("settingschange", () => {
  if (lastCompareData) renderCompareResult(lastCompareData);
});

setupScenario("a");
setupScenario("b");
