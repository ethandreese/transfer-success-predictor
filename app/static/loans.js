const state = { offset: 0, total: 0, openCard: null };
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

/** Read the age/duration range filter inputs into API-ready params, converting the displayed duration in years to the raw day count the API expects - shared by currentFilterParams() and thus by the table fetch, URL sync, and export link alike, so all three always agree on the active range filters. */
function rangeFilterParams() {
  const params = {};
  const minAge = document.getElementById("min-age-input").value;
  const maxAge = document.getElementById("max-age-input").value;
  const minDuration = document.getElementById("min-duration-input").value;
  const maxDuration = document.getElementById("max-duration-input").value;
  if (minAge) params.min_age = minAge;
  if (maxAge) params.max_age = maxAge;
  if (minDuration) params.min_duration = Math.round(parseFloat(minDuration) * 365.25);
  if (maxDuration) params.max_duration = Math.round(parseFloat(maxDuration) * 365.25);
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

/** Build the /api/loans query string from the current search box, filter dropdowns, sort selection, and pagination offset. Page size comes from the Settings panel (default 25). */
function currentParams() {
  return new URLSearchParams({
    ...currentFilterParams(),
    limit: settings.pageSize,
    offset: state.offset,
  });
}

/** Fetch the distinct positions/leagues from /api/loans/filters and populate the two filter <select> dropdowns. */
async function loadFilters() {
  const res = await fetch("/api/loans/filters");
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

/** Format a loan's duration in days as "Xmo" (or "X.Yy" once it's a year or longer), flagging it "(ongoing)" if the loan hasn't ended yet. */
function durationDisplay(days, stillOnLoan) {
  const label = days >= 365 ? `${(days / 365.25).toFixed(1)}y` : `${Math.round(days / 30.44)}mo`;
  return `${label}${stillOnLoan ? " (ongoing)" : ""}`;
}

/** A small pill marking a loan whose player later signed permanently for the same club they were on loan at (see find_loan_conversion in main.py) - empty string if it never converted. */
function conversionBadge(convertedToPermanent, conversionDate, conversionScore) {
  if (!convertedToPermanent) return "";
  return `<span class="loan-conversion-badge" title="Signed permanently on ${conversionDate.slice(0, 7)}, scored ${conversionScore}">&#10003; Permanent</span>`;
}

/** Render "N loans, avg score X, avg duration Y days" above the table, from /api/loans' summary (computed over the *full* filtered set server-side, not just the current page). */
function renderResultsSummary(total, summary) {
  const el = document.getElementById("results-summary");
  if (!total) {
    el.textContent = "";
    return;
  }
  el.textContent = `${total.toLocaleString()} loan${total === 1 ? "" : "s"}, avg score ${summary.avg_score}, avg duration ${durationDisplay(summary.avg_duration_days, false)}`;
}

/**
 * Click-to-sort table headers, an alternative to the Sort by dropdown -
 * both drive the same sort-select value, so everything downstream stays
 * in sync no matter which control was used. Clicking the already-active
 * column toggles its order; clicking a different one switches to it at a
 * sensible default direction. Same pattern as browse.js's own version -
 * every field/direction combination this can produce must exist as a
 * real <option> in the dropdown (see loans.html).
 */
const SORTABLE_COLUMN_DEFAULT_ORDER = {
  transfer_date: "desc", age_at_transfer: "asc", tenure_days: "desc", loan_success_score: "desc",
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

/** Fetch the current page of loans (per currentParams()) and render the table body, pagination controls, and per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="9">Loading...</td></tr>`;
  const res = await fetch(`/api/loans?${currentParams().toString()}`);
  const data = await res.json();
  state.total = data.total;
  renderResultsSummary(data.total, data.summary);
  updateSortIndicator();

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="9">No loans match these filters.</td></tr>`;
  } else {
    tbody.innerHTML = data.results.map(r => `
      <tr data-player-id="${r.player_id}" data-transfer-date="${r.transfer_date}" tabindex="0" role="button" aria-label="View loan details: ${r.name} to ${r.to_club}">
        <td title="${escapeAttr(r.name)}">${r.name}</td>
        <td>${r.position}</td>
        <td title="${escapeAttr(r.from_club + " → " + r.to_club)}">${r.from_club} &rarr; ${r.to_club}</td>
        <td title="${escapeAttr(r.to_league)}">${r.to_league}</td>
        <td>${r.transfer_date.slice(0, 7)}</td>
        <td>${r.age_at_transfer}</td>
        <td>${durationDisplay(r.tenure_days, r.still_on_loan)}</td>
        <td style="color:${scoreColor(r.loan_success_score)}; font-weight:700">${r.loan_success_score}</td>
        <td>${conversionBadge(r.converted_to_permanent, r.conversion_transfer_date, r.conversion_success_score)}</td>
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

/** Point the "Export as CSV" link at /api/loans/export with the current search/filter/sort/range selections - export has no pagination, it always returns every matching row, so limit/offset are left out. */
function updateExportLink() {
  const params = new URLSearchParams(currentFilterParams());
  document.getElementById("export-link").href = `/api/loans/export?${params.toString()}`;
}

/** Render "Page X of Y (Z loans)", with X as a click-to-edit trigger for jumping to an arbitrary page. */
function renderPageInfo() {
  const page = Math.floor(state.offset / settings.pageSize) + 1;
  const pageCount = Math.max(1, Math.ceil(state.total / settings.pageSize));
  document.getElementById("page-info").innerHTML =
    `Page <span class="page-jump-trigger" id="page-jump-trigger" tabindex="0" role="button" aria-label="Jump to a specific page" title="Click to jump to a page">${page}</span> of ${pageCount} (${state.total} loans)`;
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
    sortSelect.value = "loan_success_score:desc";
  }
}

/** Render one loan's score-component breakdown (label + bar + hover tooltip) as HTML, from the `breakdown` array the API returns. Descriptions run through convertMoneyInText since the backend always formats euro amounts in its prose. */
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

/** Open the modal and fetch+render the full loan card for one clicked table row, via /api/loans/detail. Remembers the open card (state.openCard) so a currency/theme change can refresh it in place. */
async function showCard(playerId, transferDate) {
  state.openCard = { playerId, transferDate };
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  cardModal.open();
  try {
    const res = await fetch(`/api/loans/detail?player_id=${playerId}&transfer_date=${transferDate}`);
    if (!res.ok) throw new Error("Could not load this loan.");
    const ex = await res.json();
    const duration = durationDisplay(ex.tenure_days, ex.still_on_loan);
    content.innerHTML = `
      <div class="example-card" style="border:none; padding:1.25rem;">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(ex.loan_success_score)}">${ex.loan_success_score}</div>
        <div class="tenure-note">
          Scored over the loan spell (${duration})
        </div>
        ${ex.converted_to_permanent ? `
          <div class="tenure-note">
            ${conversionBadge(ex.converted_to_permanent, ex.conversion_transfer_date, ex.conversion_success_score)}
            signed permanently on ${ex.conversion_transfer_date.slice(0, 7)}, scored ${ex.conversion_success_score}
          </div>
        ` : ""}
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `;
  } catch (e) {
    content.innerHTML = `<div class="error-box">${e.message}</div>`;
  }
}

/** Close the loan-card modal. */
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
["min-age-input", "max-age-input", "min-duration-input", "max-duration-input"].forEach(id => {
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
  resetAndLoad();
  if (state.openCard) showCard(state.openCard.playerId, state.openCard.transferDate);
});

// Restore search/filter/sort/page straight from the URL (a bookmarked or
// shared link) so landing here already shows that view, not always the
// unfiltered default. position/league can't be applied until loadFilters()
// has populated their <option>s, so those two wait on it specifically -
// q/sort/offset don't depend on it and apply immediately so the very
// first loadTable() call already reflects them.
const urlParams = readURLParams();
if (urlParams.q) document.getElementById("search-input").value = urlParams.q;
if (urlParams.sort && urlParams.order) {
  const sortSelect = document.getElementById("sort-select");
  const sortValue = `${urlParams.sort}:${urlParams.order}`;
  if ([...sortSelect.options].some(o => o.value === sortValue)) sortSelect.value = sortValue;
}
state.offset = parseInt(urlParams.offset, 10) || 0;
if (urlParams.min_age) document.getElementById("min-age-input").value = urlParams.min_age;
if (urlParams.max_age) document.getElementById("max-age-input").value = urlParams.max_age;
if (urlParams.min_duration) document.getElementById("min-duration-input").value = (parseFloat(urlParams.min_duration) / 365.25).toFixed(1);
if (urlParams.max_duration) document.getElementById("max-duration-input").value = (parseFloat(urlParams.max_duration) / 365.25).toFixed(1);

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
