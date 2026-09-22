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

// The fee slider's range, in EUR millions - rescaled to the selected
// currency in updateFeeCurrencyDisplay below, same as the number input's
// displayed value, so the two controls' numbers always agree.
const FEE_SLIDER_MAX_EUR_M = 200;

/** Sync the fee field's label (currency symbol) and displayed value - both the number input and the slider - from `state.feeEurMillions` to the currently-selected currency. Call after a currency change, or once at load if a non-EUR currency was already saved. */
function updateFeeCurrencyDisplay(labelEl, inputEl, sliderEl) {
  const symbol = CURRENCY_SYMBOLS[settings.currency];
  const rate = EXCHANGE_RATES[settings.currency];
  labelEl.textContent = `Transfer fee (${symbol}m)`;
  const displayValue = Math.round(state.feeEurMillions * rate * 10) / 10;
  inputEl.value = displayValue;
  sliderEl.max = Math.round(FEE_SLIDER_MAX_EUR_M * rate);
  // A fee above the slider's max (a real Mbappé/Neymar-tier transfer) still
  // types fine into the number input - the slider itself just pins to its
  // own max rather than under/overflowing, same as a native range input
  // already does for a value outside [min, max].
  sliderEl.value = Math.min(displayValue, Number(sliderEl.max));
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

function linearScale(d0, d1, r0, r1) {
  return (v) => d1 === d0 ? (r0 + r1) / 2 : r0 + (v - d0) / (d1 - d0) * (r1 - r0);
}

function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  return (v) => l1 === l0 ? (r0 + r1) / 2 : r0 + (Math.log10(v) - l0) / (l1 - l0) * (r1 - r0);
}

/** Horizontal gridlines + left-edge labels at 0/25/50/75/100 - same convention as analytics.js's scoreGridlines(). */
function scoreGridlines(y, padL, padR, W) {
  return [0, 25, 50, 75, 100].map(score => `
    <line x1="${padL}" y1="${y(score).toFixed(1)}" x2="${W - padR}" y2="${y(score).toFixed(1)}" stroke="var(--border)" stroke-width="1" />
    <text x="${padL - 6}" y="${(y(score) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">${score}</text>
  `).join("");
}

/**
 * "Where this prediction lands": the sitewide binned-average trend line
 * (fee_trend/age_trend from /api/analytics/trends - see loadTrends()
 * below) plus one highlighted marker for this specific prediction's own
 * (fee or age, score) point, so a bare predicted number isn't shown
 * divorced from how every other real transfer at a similar fee/age
 * actually went. Same dashed-line styling as the Analytics page's trend
 * lines (var(--trend-line)), just without that page's underlying scatter
 * of individual transfers - this chart only ever draws one point.
 */
function buildTrendMarkerChart(trend, xValue, score, scaleType) {
  const W = 700, H = 170, padL = 40, padR = 16, padT = 14, padB = 24;
  const chartH = H - padT - padB;
  const y = (s) => padT + (100 - s) / 100 * chartH;

  const trendXs = trend.map(t => t.x);
  const minX = Math.min(...trendXs, xValue);
  const maxX = Math.max(...trendXs, xValue);
  const x = scaleType === "log" ? logScale(minX, maxX, padL, W - padR) : linearScale(minX, maxX, padL, W - padR);

  const trendLine = `<polyline points="${trend.map(t => `${x(t.x).toFixed(1)},${y(t.avg_score).toFixed(1)}`).join(" ")}" fill="none" stroke="var(--trend-line)" stroke-width="2.5" stroke-dasharray="7 4" stroke-linecap="round" opacity="0.9" />`;

  const markerColor = scoreColor(score);
  const markerX = x(xValue).toFixed(1), markerY = y(score).toFixed(1);
  const marker = `
    <line x1="${markerX}" y1="${padT}" x2="${markerX}" y2="${H - padB}" stroke="${markerColor}" stroke-width="1" stroke-dasharray="3 3" opacity="0.5" />
    <circle cx="${markerX}" cy="${markerY}" r="6" fill="${markerColor}" stroke="var(--panel)" stroke-width="2" />
  `;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="This prediction plotted against the sitewide trend">
    ${scoreGridlines(y, padL, padR, W)}
    ${trendLine}
    ${marker}
  </svg>`;
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
    // Clamped to the backend's accepted range (age_at_transfer: ge=15, le=42
    // in PredictRequest) - the field's own min/max attributes don't help
    // here since it isn't inside a <form>, so an out-of-range autofilled
    // age (e.g. a 42+ year-old player) would otherwise 422 the moment
    // Predict is clicked without editing it first.
    document.getElementById("age-override").value = Math.min(42, Math.max(15, Number(p.age_now))).toFixed(1);
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

let trendsPromise = null;
/** Fetch /api/analytics/trends once and cache it - the fee/age binned-average series behind the "Where this lands" marker charts don't change per-prediction, so there's no reason to re-fetch on every predict. */
function loadTrends() {
  if (!trendsPromise) trendsPromise = fetch("/api/analytics/trends").then(r => r.json());
  return trendsPromise;
}

/** The avg_score of whichever trend bucket's x is closest to `value` - a quick "roughly how the sitewide average compares" figure for a marker chart's caption. */
function nearestTrendAvg(trend, value) {
  return trend.reduce((best, t) => Math.abs(t.x - value) < Math.abs(best.x - value) ? t : best).avg_score;
}

/** "above"/"below"/"in line with" - a marker chart caption's comparison clause, with a small dead zone around 0 so a near-exact match doesn't read as a false "above" or "below". */
function landsClause(score, trendAvg) {
  const diff = score - trendAvg;
  if (Math.abs(diff) < 2) return "in line with";
  return diff > 0 ? "above" : "below";
}

/**
 * Render the "Where this lands" marker charts for the just-computed
 * prediction: this transfer's own (fee, score) and (age, score) plotted
 * against the sitewide binned-average trend from /api/analytics/trends
 * (fetched once and cached - see loadTrends()), so a bare predicted
 * number isn't shown divorced from how every other real transfer at a
 * similar fee/age actually went. `age`/`feeEur` come from
 * state.lastPredictInputs (the exact values that produced `score`), not
 * read live from the form - age has no live-repredict the way fee does
 * (see scheduleLiveRepredict), so if it were read live here, editing the
 * age field without re-clicking "Predict success" and then triggering
 * any re-render (e.g. a currency change) would plot the *new*, unsubmitted
 * age against the *old* score, a mismatched, misleading pair.
 */
async function renderTrendMarkers(score, age, feeEur) {
  const trends = await loadTrends();

  const feeChart = document.getElementById("fee-trend-chart");
  const feeDesc = document.getElementById("fee-trend-desc");
  if (feeEur > 0) {
    feeChart.innerHTML = buildTrendMarkerChart(trends.fee_trend, feeEur, score, "log");
    const trendAvg = nearestTrendAvg(trends.fee_trend, feeEur);
    feeDesc.textContent = `At ${formatMoney(feeEur)}, transfers around this fee average ${trendAvg}; this prediction (${score}) is ${landsClause(score, trendAvg)} that.`;
  } else {
    feeChart.innerHTML = "";
    feeDesc.textContent = "A free transfer has no fee to plot against the sitewide fee trend.";
  }

  document.getElementById("age-trend-chart").innerHTML = buildTrendMarkerChart(trends.age_trend, age, score, "linear");
  const ageTrendAvg = nearestTrendAvg(trends.age_trend, age);
  document.getElementById("age-trend-desc").textContent =
    `At age ${age.toFixed(1)}, transfers around this age average ${ageTrendAvg}; this prediction (${score}) is ${landsClause(score, ageTrendAvg)} that.`;
}

/** Render a /api/predict response into the #result panel. Factored out from the click handler so a settings change (currency) can re-render the last result without re-predicting. */
function renderPredictResult(data) {
  document.getElementById("result").classList.add("open");
  const placeholder = document.getElementById("result-placeholder");
  if (placeholder) placeholder.hidden = true;
  const scoreEl = document.getElementById("score-value");
  scoreEl.textContent = data.success_score;
  scoreEl.style.color = scoreColor(data.success_score);
  const [lo, hi] = data.score_range;
  document.getElementById("score-range-note").textContent =
    `Likely range: ${lo}–${hi}, based on the most similar historical transfers`;
  document.getElementById("mae-value").textContent = data.model_test_mae;
  document.getElementById("r2-value").textContent = data.model_test_r2;
  renderTrendMarkers(data.success_score, state.lastPredictInputs.age, state.lastPredictInputs.feeEur);
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
  renderSensitivity(data.sensitivity);
}

/** Render "What would move this most" (see /api/predict's sensitivity field) - same .explain-row/.explain-bar-fill markup as "Why this score" above, always the "pos" (green) bar since every entry here is by construction a genuine improvement. Hides the whole section when sensitivity is empty (nothing left with real data and room to improve) rather than showing an empty heading. */
function renderSensitivity(sensitivity) {
  const section = document.getElementById("sensitivity-section");
  section.hidden = sensitivity.length === 0;
  if (!sensitivity.length) return;
  document.getElementById("sensitivity-list").innerHTML = sensitivity.map(s => `
    <div class="explain-row">
      <span class="tooltip-wrap explain-label">
        ${s.label}
        <span class="tooltip-box">${s.detail}</span>
      </span>
      <div class="explain-bar-track">
        <div class="explain-bar-fill pos" style="width:${Math.min(s.gain * 15, 100)}%"></div>
      </div>
      <span class="explain-value">+${s.gain}</span>
    </div>
  `).join("");
}

const feeLabel = document.querySelector('label[for="fee"]');
const feeInput = document.getElementById("fee");
const feeSlider = document.getElementById("fee-slider");
updateFeeCurrencyDisplay(feeLabel, feeInput, feeSlider);

/**
 * Debounced live re-predict, fired whenever the fee changes (typed or
 * dragged) after a first real prediction already exists - lets dragging
 * the fee slider show the score update as you drag, instead of requiring
 * another click on "Predict success" for every fee tried. Never fires
 * before that first click: nothing meaningful to show yet, and silently
 * calling /api/predict for an unselected player/club would just error.
 * Age has no equivalent live control - the age field can still be edited
 * by hand, but a player's age at a hypothetical transfer isn't really
 * something to "explore a range of" the way a fee is.
 */
let liveRepredictTimer = null;
function scheduleLiveRepredict() {
  if (!state.lastPredictData) return;
  clearTimeout(liveRepredictTimer);
  liveRepredictTimer = setTimeout(runPrediction, 250);
}

// Keep state.feeEurMillions (the real, currency-independent value) in
// sync with whatever the user types or drags, converting from whichever
// currency is currently displayed - see updateFeeCurrencyDisplay for the
// other direction (a currency change redisplaying the same real fee).
// The number input and the slider mirror each other's value on every
// change, so typing an exact figure moves the slider's thumb too and
// vice versa.
feeInput.addEventListener("input", () => {
  const typed = parseFloat(feeInput.value);
  state.feeEurMillions = Number.isNaN(typed) ? 0 : typed / EXCHANGE_RATES[settings.currency];
  feeSlider.value = Math.min(typed || 0, Number(feeSlider.max));
  scheduleLiveRepredict();
});
feeSlider.addEventListener("input", () => {
  feeInput.value = feeSlider.value;
  const typed = parseFloat(feeSlider.value);
  state.feeEurMillions = Number.isNaN(typed) ? 0 : typed / EXCHANGE_RATES[settings.currency];
  scheduleLiveRepredict();
});

/** Assemble a PredictRequest from the selected player/club plus the fee and (editable) age fields, POST it to /api/predict, and render the result. Used both by the "Predict success" button and scheduleLiveRepredict()'s debounced fee-change re-predict. */
async function runPrediction() {
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
  // Snapshot exactly what this payload sends, for renderTrendMarkers() to
  // plot against - see its docstring for why this can't just read the
  // form fields live at render time.
  state.lastPredictInputs = { age: payload.age_at_transfer, feeEur: payload.transfer_fee };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(formatApiError(err.detail) || "Prediction failed");
    }
    state.lastPredictData = await res.json();
    renderPredictResult(state.lastPredictData);
  } catch (e) {
    errorBox.textContent = e.message;
  }
}

document.getElementById("predict-btn").addEventListener("click", runPrediction);

// A settings change (currency, ...) doesn't change the underlying data,
// just how it's displayed - redisplay the fee input in the new currency,
// and, if a prediction is already showing, re-render it from the cached
// response rather than re-predicting.
document.addEventListener("settingschange", () => {
  updateFeeCurrencyDisplay(feeLabel, feeInput, feeSlider);
  if (state.lastPredictData) renderPredictResult(state.lastPredictData);
});
