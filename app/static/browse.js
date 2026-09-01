const state = { offset: 0, total: 0, openCard: null };

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Build the /api/transfers query string from the current search box, filter dropdowns, sort selection, and pagination offset. Page size comes from the Settings panel (default 25). */
function currentParams() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  const params = new URLSearchParams({
    sort, order,
    limit: settings.pageSize,
    offset: state.offset,
  });
  const q = document.getElementById("search-input").value.trim();
  const position = document.getElementById("position-select").value;
  const league = document.getElementById("league-select").value;
  if (q) params.set("q", q);
  if (position) params.set("position", position);
  if (league) params.set("league", league);
  return params;
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

/** Fetch the current page of transfers (per currentParams()) and render the table body, pagination controls, and per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="9">Loading...</td></tr>`;
  const res = await fetch(`/api/transfers?${currentParams().toString()}`);
  const data = await res.json();
  state.total = data.total;

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="9">No transfers match these filters.</td></tr>`;
  } else {
    tbody.innerHTML = data.results.map(r => `
      <tr data-player-id="${r.player_id}" data-transfer-date="${r.transfer_date}">
        <td>${r.name}</td>
        <td>${r.position}</td>
        <td>${r.from_club} &rarr; ${r.to_club}</td>
        <td>${r.to_league}</td>
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
    });
  }

  document.getElementById("prev-page").disabled = state.offset === 0;
  document.getElementById("next-page").disabled = state.offset + settings.pageSize >= state.total;
  renderPageInfo();
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
  const backdrop = document.getElementById("card-modal-backdrop");
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  backdrop.classList.add("open");
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
  document.getElementById("card-modal-backdrop").classList.remove("open");
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

loadFilters();
loadTable();
