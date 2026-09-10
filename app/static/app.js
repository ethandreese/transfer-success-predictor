const state = {
  player: null,
  playerClub: null,
  club: null,
  // The fee input's "real" value in EUR millions - the unit the backend
  // always expects (see buildPayload-equivalent below) and the one every
  // other currency conversion on the site works from. Kept separate from
  // whatever the input currently *displays* (€/$/£, depending on the
  // currency setting) so switching currency redisplays the same real fee
  // in the new currency instead of reinterpreting the same digits as a
  // different amount - the same behavior every other money value on the
  // site already has via formatMoney, which this field previously didn't:
  // its label hardcoded "(€m)" and its typed number was always sent as
  // EUR regardless of the selected currency, silently wrong once the
  // currency setting was anything but EUR.
  feeEurMillions: 50,
};

/** Sync the fee field's label (currency symbol) and displayed value from `state.feeEurMillions` to the currently-selected currency - call after a currency change, or once at load if a non-EUR currency was already saved. */
function updateFeeCurrencyDisplay(labelEl, inputEl) {
  const symbol = CURRENCY_SYMBOLS[settings.currency];
  labelEl.textContent = `Transfer fee (${symbol}m)`;
  inputEl.value = Math.round(state.feeEurMillions * EXCHANGE_RATES[settings.currency] * 10) / 10;
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

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/**
 * Wire a text input to a debounced search-as-you-type dropdown: on input,
 * queries `endpoint?q=...`, renders each result via `renderLabel`, and
 * calls `onSelect(item)` when the user picks one. Closes the dropdown on
 * an empty/short query or a click outside the input. `filterResults`
 * (optional) can drop items from the raw API response before they're
 * shown - e.g. hiding a player's own current club from the destination
 * search.
 *
 * Keyboard-navigable (ArrowUp/Down to move the highlight, Enter to select,
 * Escape to close) and exposes the standard ARIA combobox pattern
 * (role="combobox" on the input, role="listbox"/"option" on the dropdown,
 * aria-activedescendant tracking the highlight) - previously mouse/click
 * only, which meant a keyboard-only or screen-reader user couldn't select
 * a player or club at all, i.e. couldn't use the predict form.
 */
function setupAutocomplete({ inputId, listId, endpoint, onSelect, renderLabel, filterResults }) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  let debounceTimer = null;
  let items = [];
  let highlighted = -1;

  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-expanded", "false");
  input.setAttribute("aria-controls", listId);
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
      input.setAttribute("aria-activedescendant", `${listId}-opt-${i}`);
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
        `<div id="${listId}-opt-${i}" role="option" data-idx="${i}">${renderLabel(item)}</div>`
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
// Excludes the selected player's own current club - a "transfer" to the
// club a player is already at isn't a real scenario, and the model has no
// way to flag that for you (it'll just score it like any other move).
setupAutocomplete({
  inputId: "club-search",
  listId: "club-list",
  endpoint: "/api/clubs/search",
  renderLabel: (c) => `${c.name}`,
  filterResults: (clubs) => state.player ? clubs.filter(c => c.club_id !== state.player.current_club_id) : clubs,
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

const feeLabel = document.querySelector('label[for="fee"]');
const feeInput = document.getElementById("fee");
updateFeeCurrencyDisplay(feeLabel, feeInput);
// Keep state.feeEurMillions (the real, currency-independent value) in
// sync with whatever the user types, converting from whichever currency
// is currently displayed - see updateFeeCurrencyDisplay for the other
// direction (a currency change redisplaying the same real fee).
feeInput.addEventListener("input", () => {
  const typed = parseFloat(feeInput.value);
  state.feeEurMillions = Number.isNaN(typed) ? 0 : typed / EXCHANGE_RATES[settings.currency];
});

// Assemble a PredictRequest from the selected player/club plus the fee and
// (editable) age fields, POST it to /api/predict, and render the result.
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
    transfer_fee: state.feeEurMillions * 1_000_000,
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
// just how it's displayed - redisplay the fee input in the new currency,
// and, if a prediction is already showing, re-render it from the cached
// response rather than re-predicting.
document.addEventListener("settingschange", () => {
  updateFeeCurrencyDisplay(feeLabel, feeInput);
  if (state.lastPredictData) renderPredictResult(state.lastPredictData);
});
