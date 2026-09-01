const state = {
  player: null,
  playerClub: null,
  club: null,
};

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

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Render one transfer's score-component breakdown (label + bar + hover tooltip) as HTML, from the `breakdown` array the API returns. Descriptions run through convertMoneyInText since the backend always formats euro amounts in its prose. */
function renderBreakdown(breakdown) {
  return breakdown.map(b => `
    <div class="breakdown-row">
      <span class="tooltip-wrap breakdown-label">
        ${b.label}
        <span class="tooltip-box">
          ${convertMoneyInText(b.description)}
          ${b.stats ? `<ul class="tooltip-stats">${b.stats.map(s => `<li>${convertMoneyInText(s)}</li>`).join("")}</ul>` : ""}
        </span>
      </span>
      <div class="breakdown-bar-track">
        <div class="breakdown-bar-fill" style="width:${b.value}%; background:${scoreColor(b.value)}"></div>
      </div>
      <span class="breakdown-value">${b.value}</span>
    </div>
  `).join("");
}

/** Fetch the curated homepage cards from /api/examples and render them into #examples. */
async function loadExamples() {
  const el = document.getElementById("examples");
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    if (!data.length) {
      el.textContent = "No examples available.";
      return;
    }
    const years = (days) => (days / 365.25).toFixed(1);
    el.innerHTML = data.map(ex => `
      <div class="example-card">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(ex.success_score)}">${ex.success_score}</div>
        <div class="tenure-note">
          Scored over ${years(ex.tenure_days)} years at the club${ex.still_at_club ? " (still there)" : " (before leaving)"}
        </div>
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `).join("");
  } catch (e) {
    el.textContent = "Failed to load examples.";
  }
}

/**
 * Wire a text input to a debounced search-as-you-type dropdown: on input,
 * queries `endpoint?q=...`, renders each result via `renderLabel`, and
 * calls `onSelect(item)` when the user picks one. Closes the dropdown on
 * an empty/short query or a click outside the input.
 */
function setupAutocomplete({ inputId, listId, endpoint, onSelect, renderLabel }) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
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

/** Enable the "Predict success" button only once both a player and a destination club have been selected. */
function updatePredictButton() {
  document.getElementById("predict-btn").disabled = !(state.player && state.club);
}

// Player autocomplete: selecting a player fills in their current age/club
// and fetches their current club's details (for the "origin" side of the
// prediction payload).
setupAutocomplete({
  inputId: "player-search",
  listId: "player-list",
  endpoint: "/api/players/search",
  renderLabel: (p) => `${p.name} (${p.position}, ${p.current_club_name})`,
  onSelect: async (p) => {
    state.player = p;
    document.getElementById("player-chip").innerHTML =
      `<span class="selected-chip">${p.name} &middot; age ${Number(p.age_now).toFixed(1)} &middot; ${p.current_club_name}</span>`;
    document.getElementById("age-override").value = Number(p.age_now).toFixed(1);
    if (p.current_club_id) {
      const res = await fetch(`/api/clubs/${p.current_club_id}`);
      state.playerClub = res.ok ? await res.json() : null;
    }
    updatePredictButton();
  },
});

// Destination-club autocomplete: just records the selection, since the
// club's value proxy/league already come back in the search result.
setupAutocomplete({
  inputId: "club-search",
  listId: "club-list",
  endpoint: "/api/clubs/search",
  renderLabel: (c) => `${c.name}`,
  onSelect: (c) => {
    state.club = c;
    document.getElementById("club-chip").innerHTML =
      `<span class="selected-chip">${c.name}</span>`;
    updatePredictButton();
  },
});

/** Render a /api/predict response into the #result panel. Factored out from the click handler so a settings change (currency) can re-render the last result without re-predicting. */
function renderPredictResult(data) {
  document.getElementById("result").classList.add("open");
  const scoreEl = document.getElementById("score-value");
  scoreEl.textContent = data.success_score;
  scoreEl.style.color = scoreColor(data.success_score);
  const [lo, hi] = data.score_range;
  document.getElementById("score-range-note").textContent =
    `Likely range: ${lo}–${hi}, based on the most similar historical transfers`;
  document.getElementById("mae-value").textContent = data.model_test_mae;
  document.getElementById("r2-value").textContent = data.model_test_r2;
  document.getElementById("comparables-list").innerHTML = data.comparable_transfers.map(c => `
    <div class="comp-row">
      <span>${c.name} (${c.from_club} &rarr; ${c.to_club}, ${c.transfer_date.slice(0, 7)})</span>
      <span style="color:${scoreColor(c.success_score)}">${c.success_score}</span>
    </div>
  `).join("");
  document.getElementById("explanation-list").innerHTML = data.explanation.map(e => {
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

// Assemble a PredictRequest from the selected player/club plus the fee and
// (editable) age fields, POST it to /api/predict, and render the result.
// The fee input is always in EUR regardless of the currency setting (it
// feeds the model directly) - only the displayed output is converted.
document.getElementById("predict-btn").addEventListener("click", async () => {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = "";
  if (!state.player || !state.club) return;

  const fromClub = state.playerClub || {
    domestic_competition_id: state.player.current_club_domestic_competition_id,
    club_value_proxy: 0,
  };

  const payload = {
    age_at_transfer: parseFloat(document.getElementById("age-override").value) || state.player.age_now,
    height_in_cm: state.player.height_in_cm,
    position: state.player.position,
    sub_position: state.player.sub_position || state.player.position,
    foot: state.player.foot || "unknown",
    pre_apps: state.player.recent_apps,
    pre_minutes: state.player.recent_minutes,
    pre_goals_p90: state.player.recent_goals_p90,
    pre_ga_p90: state.player.recent_ga_p90,
    pre_mins_per_app: state.player.recent_mins_per_app,
    transfer_fee: (parseFloat(document.getElementById("fee").value) || 0) * 1_000_000,
    value_before: state.player.market_value_in_eur,
    from_domestic_competition_id: fromClub.domestic_competition_id || "unknown",
    to_domestic_competition_id: state.club.domestic_competition_id || "unknown",
    from_total_market_value: fromClub.club_value_proxy || 1,
    to_total_market_value: state.club.club_value_proxy || 1,
    ...pretransferFotmobFeatures(state.player),
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }
    state.lastPredictData = await res.json();
    renderPredictResult(state.lastPredictData);
  } catch (e) {
    errorBox.textContent = e.message;
  }
});

// A settings change (currency, ...) doesn't change the underlying data,
// just how it's displayed - reload the examples grid and, if a prediction
// is already showing, re-render it from the cached response rather than
// re-predicting.
document.addEventListener("settingschange", () => {
  loadExamples();
  if (state.lastPredictData) renderPredictResult(state.lastPredictData);
});

loadExamples();
