// minFeeEurM/maxFeeEurM are the fee-range inputs' "real" values in EUR
// millions - null when that bound is unset - the same currency-independent
// unit every other money value on the site works from (see formatMoney).
// Kept separate from whatever the inputs currently *display* (€/$/£,
// depending on the currency setting) so switching currency redisplays the
// same real bounds instead of reinterpreting the same typed digits as a
// different amount - see updateFeeCurrencyDisplay below.
const state = { offset: 0, total: 0, openCard: null, minFeeEurM: null, maxFeeEurM: null };

/** Sync the fee-range label's currency symbol and the min/max fee inputs' displayed values from state.minFeeEurM/maxFeeEurM to the currently-selected currency. Call after a currency change, or once at load if a non-EUR currency was already saved. */
function updateFeeCurrencyDisplay() {
  const symbol = CURRENCY_SYMBOLS[settings.currency];
  const rate = EXCHANGE_RATES[settings.currency];
  document.getElementById("fee-range-label").textContent = `Fee range (${symbol}m)`;
  document.getElementById("min-fee-input").value =
    state.minFeeEurM === null ? "" : Math.round(state.minFeeEurM * rate * 10) / 10;
  document.getElementById("max-fee-input").value =
    state.maxFeeEurM === null ? "" : Math.round(state.maxFeeEurM * rate * 10) / 10;
}
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Escape a value for safe interpolation inside an HTML attribute (e.g. a title="..." tooltip) - real club names can contain a literal " or & (e.g. Brighton & Hove Albion, or a club whose native name is quoted), which would otherwise break out of the attribute. */
function escapeAttr(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Read the fee/age range filter inputs into API-ready params, converting the fee bounds from their currency-independent EUR-millions state (see updateFeeCurrencyDisplay) to the raw euro amount the API expects - shared by currentFilterParams() and thus by the table fetch, URL sync, and export link alike, so all three always agree on the active range filters. */
function rangeFilterParams() {
  const params = {};
  if (state.minFeeEurM !== null) params.min_fee = Math.round(state.minFeeEurM * 1_000_000);
  if (state.maxFeeEurM !== null) params.max_fee = Math.round(state.maxFeeEurM * 1_000_000);
  const minAge = document.getElementById("min-age-input").value;
  const maxAge = document.getElementById("max-age-input").value;
  if (minAge) params.min_age = minAge;
  if (maxAge) params.max_age = maxAge;
  return params;
}

/** The current search/filter/sort/range selections as a plain object, with no pagination - shared by currentParams() (adds limit/offset for the table fetch) and updateExportLink() (which always covers every matching row, not just one page). */
function currentFilterParams() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  const params = { sort, order, ...rangeFilterParams() };
  const q = document.getElementById("search-input").value.trim();
  const position = document.getElementById("position-select").value;
  const league = document.getElementById("league-select").value;
  if (q) params.q = q;
  if (position) params.position = position;
  if (league) params.league = league;
  return params;
}

/** Build the /api/transfers query string from the current search box, filter dropdowns, sort selection, and pagination offset. Page size comes from the Settings panel (default 25). */
function currentParams() {
  return new URLSearchParams({
    ...currentFilterParams(),
    limit: settings.pageSize,
    offset: state.offset,
  });
}

/** Fetch the distinct positions/leagues from /api/filters and populate the two filter <select> dropdowns. */
async function loadFilters() {
  const res = await fetch("/api/filters");
  const data = await res.json();
  const posSelect = document.getElementById("position-select");
  data.positions.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p; opt.textContent = p;
    posSelect.appendChild(opt);
  });
  const leagueSelect = document.getElementById("league-select");
  data.leagues.forEach(l => {
    const opt = document.createElement("option");
    opt.value = l.id; opt.textContent = l.name;
    leagueSelect.appendChild(opt);
  });
}

/** Format a tenure in days as "X.Yy", flagging it "(current)" if the player is still at the club. */
function tenureDisplay(days, stillAtClub) {
  const years = (days / 365.25).toFixed(1);
  return `${years}y${stillAtClub ? " (current)" : ""}`;
}

/** Render "N transfers, avg score X, €Y total spent" above the table, from /api/transfers' summary (computed over the *full* filtered set server-side, not just the current page) - a read on the filtered view as a whole, not just whichever page happens to be showing. */
function renderResultsSummary(total, summary) {
  const el = document.getElementById("results-summary");
  if (!total) {
    el.textContent = "";
    return;
  }
  el.textContent = `${total.toLocaleString()} transfer${total === 1 ? "" : "s"}, avg score ${summary.avg_score}, ${formatMoney(summary.total_spent)} total spent`;
}

/**
 * Click-to-sort table headers, an alternative to the Sort by dropdown -
 * both drive the same sort-select value, so everything downstream
 * (currentParams/syncURL/the dropdown's own displayed selection) stays
 * in sync automatically no matter which control was actually used.
 * Clicking the already-active column's header toggles its order;
 * clicking a different one switches to it at a sensible default
 * direction. Every field/direction combination this can produce must
 * exist as a real <option> in the dropdown (see browse.html) - setting
 * sort-select.value to a string with no matching option would silently
 * fail to select anything and desync the header's own indicator from
 * what's actually being requested.
 */
const SORTABLE_COLUMN_DEFAULT_ORDER = {
  transfer_date: "desc", age_at_transfer: "asc", transfer_fee: "desc", tenure_days: "desc", success_score: "desc",
};

function wireSortableHeaders() {
  document.querySelectorAll(".sortable-th").forEach(th => {
    th.addEventListener("click", () => {
      const field = th.dataset.sort;
      const sortSelect = document.getElementById("sort-select");
      const [currentField, currentOrder] = sortSelect.value.split(":");
      const order = currentField === field
        ? (currentOrder === "desc" ? "asc" : "desc")
        : SORTABLE_COLUMN_DEFAULT_ORDER[field];
      sortSelect.value = `${field}:${order}`;
      resetAndLoad();
    });
  });
}

/** Add a small ▲/▼ to whichever column header matches the current sort-select value, and mark it .is-active-sort - called after every table load so the indicator tracks the dropdown too, not just header clicks. */
function updateSortIndicator() {
  const [field, order] = document.getElementById("sort-select").value.split(":");
  document.querySelectorAll(".sortable-th").forEach(th => {
    const isActive = th.dataset.sort === field;
    th.classList.toggle("is-active-sort", isActive);
    th.innerHTML = th.textContent.replace(/\s*[▲▼]$/, "") + (isActive ? ` <span class="sort-arrow">${order === "asc" ? "▲" : "▼"}</span>` : "");
  });
}

/** Fetch the current page of transfers (per currentParams()) and render the table body, pagination controls, and per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="9">Loading...</td></tr>`;
  const res = await fetch(`/api/transfers?${currentParams().toString()}`);
  const data = await res.json();
  state.total = data.total;
  renderResultsSummary(data.total, data.summary);
  updateSortIndicator();

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="9">No transfers match these filters.</td></tr>`;
  } else {
    tbody.innerHTML = data.results.map(r => `
      <tr data-player-id="${r.player_id}" data-transfer-date="${r.transfer_date}" tabindex="0" role="button" aria-label="View transfer details: ${r.name} to ${r.to_club}">
        <td title="${escapeAttr(r.name)}">${r.name}</td>
        <td>${r.position}</td>
        <td title="${escapeAttr(r.from_club + " → " + r.to_club)}">${r.from_club} &rarr; ${r.to_club}</td>
        <td title="${escapeAttr(r.to_league)}">${r.to_league}</td>
        <td>${r.transfer_date.slice(0, 7)}</td>
        <td>${r.age_at_transfer}</td>
        <td>${formatMoney(r.transfer_fee)}</td>
        <td>${tenureDisplay(r.tenure_days, r.still_at_club)}</td>
        <td style="color:${scoreColor(r.success_score)}; font-weight:700">${r.success_score}</td>
      </tr>
    `).join("");
    [...tbody.querySelectorAll("tr")].forEach(row => {
      row.addEventListener("click", () => {
        showCard(row.dataset.playerId, row.dataset.transferDate);
      });
      row.addEventListener("keydown", (e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        showCard(row.dataset.playerId, row.dataset.transferDate);
      });
    });
  }

  document.getElementById("prev-page").disabled = state.offset === 0;
  document.getElementById("next-page").disabled = state.offset + settings.pageSize >= state.total;
  renderPageInfo();
  syncURL();
}

/** Keep the address bar's query string in sync with the current search/filter/sort/range/page, so this view is bookmarkable and shareable - see writeURLParams in settings.js. Also refreshes the export link, since it's driven by the same filters and needs to change whenever they do. */
function syncURL() {
  writeURLParams({ ...currentFilterParams(), offset: state.offset || "" });
  updateExportLink();
}

/** Point the "Export as CSV" link at /api/transfers/export with the current search/filter/sort/range selections - export has no pagination, it always returns every matching row, so limit/offset are left out. */
function updateExportLink() {
  const params = new URLSearchParams(currentFilterParams());
  document.getElementById("export-link").href = `/api/transfers/export?${params.toString()}`;
}

/** Render "Page X of Y (Z transfers)", with X as a click-to-edit trigger for jumping to an arbitrary page. */
function renderPageInfo() {
  const page = Math.floor(state.offset / settings.pageSize) + 1;
  const pageCount = Math.max(1, Math.ceil(state.total / settings.pageSize));
  document.getElementById("page-info").innerHTML =
    `Page <span class="page-jump-trigger" id="page-jump-trigger" tabindex="0" role="button" aria-label="Jump to a specific page" title="Click to jump to a page">${page}</span> of ${pageCount} (${state.total} transfers)`;
}

/**
 * Swap the clickable page-number span for an inline number input, focused
 * and pre-selected so typing immediately replaces it. Enter or blur commits
 * the jump (reloading the table); Escape reverts without reloading. The
 * `committed` guard stops blur's commit() from double-firing after Enter or
 * Escape already handled it - both remove the input from the DOM, which
 * itself triggers a blur event.
 */
function startPageJumpEdit(trigger) {
  const pageCount = Math.max(1, Math.ceil(state.total / settings.pageSize));
  const input = document.createElement("input");
  input.type = "number";
  input.min = "1";
  input.max = String(pageCount);
  input.value = trigger.textContent;
  input.className = "page-jump-input";
  trigger.replaceWith(input);
  input.focus();
  input.select();

  let committed = false;
  const commit = () => {
    if (committed) return;
    committed = true;
    const page = parseInt(input.value, 10);
    if (Number.isFinite(page)) {
      state.offset = (Math.min(Math.max(page, 1), pageCount) - 1) * settings.pageSize;
    }
    loadTable();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    else if (e.key === "Escape") { committed = true; renderPageInfo(); }
  });
  input.addEventListener("blur", commit);
}

/** Jump back to page 1 and reload - called whenever a filter/search/sort control changes, so a new query starts from the top. */
function resetAndLoad() {
  state.offset = 0;
  loadTable();
}

/**
 * "Sort by club" only makes sense once a specific league is picked - sorted
 * across every league at once, clubs would just interleave alphabetically
 * with no useful grouping. Shows/hides that option based on whether
 * league-select currently has a value, and falls back to the default sort
 * if a league is cleared while club-sort is active.
 */
function updateClubSortAvailability() {
  const hasLeague = !!document.getElementById("league-select").value;
  const clubOption = document.getElementById("club-sort-option");
  clubOption.hidden = !hasLeague;
  const sortSelect = document.getElementById("sort-select");
  if (!hasLeague && sortSelect.value.startsWith("to_club_name")) {
    sortSelect.value = "success_score:desc";
  }
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

/** Open the modal and fetch+render the full transfer card for one clicked table row, via /api/transfers/detail. Remembers the open card (state.openCard) so a currency/theme change can refresh it in place. */
async function showCard(playerId, transferDate) {
  state.openCard = { playerId, transferDate };
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  cardModal.open();
  try {
    const res = await fetch(`/api/transfers/detail?player_id=${playerId}&transfer_date=${transferDate}`);
    if (!res.ok) throw new Error("Could not load this transfer.");
    const ex = await res.json();
    const years = (ex.tenure_days / 365.25).toFixed(1);
    content.innerHTML = `
      <div class="example-card" style="border:none; padding:1.25rem;">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(ex.success_score)}">${ex.success_score}</div>
        <div class="tenure-note">
          Scored over ${years} years at the club${ex.still_at_club ? " (still there)" : " (before leaving)"}
        </div>
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `;
  } catch (e) {
    content.innerHTML = `<div class="error-box">${e.message}</div>`;
  }
}

/** Close the transfer-card modal. */
function closeCard() {
  state.openCard = null;
  cardModal.close();
}

// Close the modal via the X button, a click on the dimmed backdrop (but not
// the card itself), or the Escape key.
document.getElementById("card-modal-close").addEventListener("click", closeCard);
document.getElementById("card-modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-modal-backdrop") closeCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCard();
});

document.getElementById("search-input").addEventListener("input", () => {
  clearTimeout(window.__searchDebounce);
  window.__searchDebounce = setTimeout(resetAndLoad, 300);
});
document.getElementById("position-select").addEventListener("change", resetAndLoad);
// Debounced like search-input - these are free-typed number fields, so
// reloading on every keystroke (e.g. between "1" and "15") would thrash.
// The fee inputs also update state.minFeeEurM/maxFeeEurM (converting from
// whichever currency is currently displayed) so a later currency change
// redisplays the same real bound instead of reinterpreting the same typed
// digits as a different amount - age has no such conversion, it's typed
// and sent as-is.
document.getElementById("min-fee-input").addEventListener("input", (e) => {
  const typed = parseFloat(e.target.value);
  state.minFeeEurM = Number.isNaN(typed) ? null : typed / EXCHANGE_RATES[settings.currency];
  clearTimeout(window.__rangeDebounce);
  window.__rangeDebounce = setTimeout(resetAndLoad, 300);
});
document.getElementById("max-fee-input").addEventListener("input", (e) => {
  const typed = parseFloat(e.target.value);
  state.maxFeeEurM = Number.isNaN(typed) ? null : typed / EXCHANGE_RATES[settings.currency];
  clearTimeout(window.__rangeDebounce);
  window.__rangeDebounce = setTimeout(resetAndLoad, 300);
});
["min-age-input", "max-age-input"].forEach(id => {
  document.getElementById(id).addEventListener("input", () => {
    clearTimeout(window.__rangeDebounce);
    window.__rangeDebounce = setTimeout(resetAndLoad, 300);
  });
});
document.getElementById("league-select").addEventListener("change", () => {
  updateClubSortAvailability();
  resetAndLoad();
});
document.getElementById("sort-select").addEventListener("change", resetAndLoad);
document.getElementById("prev-page").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - settings.pageSize);
  loadTable();
});
document.getElementById("next-page").addEventListener("click", () => {
  state.offset += settings.pageSize;
  loadTable();
});
document.getElementById("page-info").addEventListener("click", (e) => {
  if (e.target.id === "page-jump-trigger") startPageJumpEdit(e.target);
});
document.getElementById("page-info").addEventListener("keydown", (e) => {
  if (e.target.id === "page-jump-trigger" && (e.key === "Enter" || e.key === " ")) {
    e.preventDefault();
    startPageJumpEdit(e.target);
  }
});

// A settings change (currency, page size, ...) doesn't change the
// underlying data, just how it's displayed - reload the current page (from
// the top, since a page-size change shifts what "page 1" even means) and
// refresh the open card, if any, rather than requiring a manual refresh.
document.addEventListener("settingschange", () => {
  updateFeeCurrencyDisplay();
  resetAndLoad();
  if (state.openCard) showCard(state.openCard.playerId, state.openCard.transferDate);
});

// Restore search/sort/page straight from the URL (a bookmarked/shared
// link, or a club report card's "View every transfer involving X" link on
// /clubs.html, which only ever sets ?q=) so landing here already shows
// that view, not always the unfiltered default. position/league can't be
// applied until loadFilters() has populated their <option>s, so those two
// wait on it specifically - q/sort/offset don't depend on it and apply
// immediately so the very first loadTable() call already reflects them.
const urlParams = readURLParams();
if (urlParams.q) document.getElementById("search-input").value = urlParams.q;
if (urlParams.sort && urlParams.order) {
  const sortSelect = document.getElementById("sort-select");
  const sortValue = `${urlParams.sort}:${urlParams.order}`;
  if ([...sortSelect.options].some(o => o.value === sortValue)) sortSelect.value = sortValue;
}
state.offset = parseInt(urlParams.offset, 10) || 0;
if (urlParams.min_fee) state.minFeeEurM = parseFloat(urlParams.min_fee) / 1_000_000;
if (urlParams.max_fee) state.maxFeeEurM = parseFloat(urlParams.max_fee) / 1_000_000;
updateFeeCurrencyDisplay();
if (urlParams.min_age) document.getElementById("min-age-input").value = urlParams.min_age;
if (urlParams.max_age) document.getElementById("max-age-input").value = urlParams.max_age;

wireSortableHeaders();

if (urlParams.position || urlParams.league) {
  loadFilters().then(() => {
    if (urlParams.position) document.getElementById("position-select").value = urlParams.position;
    if (urlParams.league) document.getElementById("league-select").value = urlParams.league;
    updateClubSortAvailability();
    loadTable();
  });
} else {
  loadFilters();
  loadTable();
}
