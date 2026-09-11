const state = { offset: 0, total: 0, rows: [], openClub: null, openTransfer: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page - resale/departure numbers are on the same 0-100 scale as success_score, so they reuse it too. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Green for a profitable total, red for a loss - total_resale_profit is a signed euro amount, not a 0-100 scale. */
function totalColor(amount) {
  return amount >= 0 ? "var(--accent)" : "var(--accent-bad)";
}

/** "+€45.0m" / "-€12.0m" - always signed so the direction reads at a glance. */
function signedMoney(amount) {
  return `${amount >= 0 ? "+" : "-"}${formatMoney(Math.abs(amount))}`;
}

/** Build the /api/clubs/leaderboard query string from the current search box, league filter, sort selection, and pagination offset. */
function currentParams() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  const params = new URLSearchParams({
    sort, order,
    limit: settings.pageSize,
    offset: state.offset,
  });
  const q = document.getElementById("search-input").value.trim();
  const league = document.getElementById("league-select").value;
  if (q) params.set("q", q);
  if (league) params.set("league", league);
  return params;
}

/** Fetch the distinct primary leagues from /api/clubs/leaderboard/filters and populate the league <select>. */
async function loadFilters() {
  const res = await fetch("/api/clubs/leaderboard/filters");
  const data = await res.json();
  const leagueSelect = document.getElementById("league-select");
  data.leagues.forEach(l => {
    const opt = document.createElement("option");
    opt.value = l.id; opt.textContent = l.name;
    leagueSelect.appendChild(opt);
  });
}

/** One <td>'s worth of a 0-100 average plus its sample size, or an em dash when there's no sample at all (e.g. a club that's only ever sold, never bought). */
function scoreCell(avg, count) {
  if (avg === null || !count) return "—";
  return `<span style="color:${scoreColor(avg)}; font-weight:700">${avg}</span> <span style="color:var(--muted)">(${count})</span>`;
}

/** Fetch the current page of clubs (per currentParams()) and render the table body, pagination controls, and per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="8">Loading...</td></tr>`;
  const res = await fetch(`/api/clubs/leaderboard?${currentParams().toString()}`);
  const data = await res.json();
  state.total = data.total;
  state.rows = data.results;

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="8">No clubs match these filters.</td></tr>`;
  } else {
    tbody.innerHTML = data.results.map((r, i) => `
      <tr data-index="${i}" tabindex="0" role="button" aria-label="View report card: ${r.club_name}">
        <td>${r.club_name}</td>
        <td>${r.league}</td>
        <td>${r.transfers_in}</td>
        <td>${scoreCell(r.avg_incoming_score, r.transfers_in)}</td>
        <td>${r.transfers_in ? formatMoney(r.total_spent) : "—"}</td>
        <td>${scoreCell(r.avg_resale_profit_pct, r.resales_count)}</td>
        <td>${r.transfers_out}</td>
        <td>${scoreCell(r.avg_departure_score, r.transfers_out)}</td>
      </tr>
    `).join("");
    [...tbody.querySelectorAll("tr")].forEach(row => {
      const open = () => showCard(state.rows[Number(row.dataset.index)]);
      row.addEventListener("click", open);
      row.addEventListener("keydown", (e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        open();
      });
    });
  }

  document.getElementById("prev-page").disabled = state.offset === 0;
  document.getElementById("next-page").disabled = state.offset + settings.pageSize >= state.total;
  renderPageInfo();
}

/** Render "Page X of Y (Z clubs)", with X as a click-to-edit trigger for jumping to an arbitrary page. */
function renderPageInfo() {
  const page = Math.floor(state.offset / settings.pageSize) + 1;
  const pageCount = Math.max(1, Math.ceil(state.total / settings.pageSize));
  document.getElementById("page-info").innerHTML =
    `Page <span class="page-jump-trigger" id="page-jump-trigger" tabindex="0" role="button" aria-label="Jump to a specific page" title="Click to jump to a page">${page}</span> of ${pageCount} (${state.total} clubs)`;
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
 * One highlight row (best/worst signing, flip, or departure) as a labeled
 * line, or nothing if this club has no data for it. A signing/departure
 * highlight carries success_score and gets a "scored X" clause; a flip
 * highlight (see `extra`) describes itself purely in fees bought/resold
 * for and has no success_score at all - appending an empty "scored"
 * clause for those would read as a sentence trailing off into nothing.
 *
 * Only the player's own name is the click target (a <button>, wired up
 * after insertion - see showCard), not the whole row: every highlight is
 * a real row of transfers_df now (see transfer_highlight/flip_highlight
 * in build_club_report_cards) and carries player_id for that button to
 * open via /api/transfers/detail, the same lookup Browse/Loans/Surprises/
 * Player Timelines already use - but the label/date/score around it are
 * plain descriptive text, not part of the link, so hovering them
 * shouldn't light up as if they were too.
 */
function highlightLine(label, h, extra) {
  if (!h) return "";
  const scoreClause = h.success_score !== undefined
    ? `, scored <strong style="color:${scoreColor(h.success_score)}">${h.success_score}</strong>`
    : "";
  return `
    <div class="highlight-row">
      <span class="highlight-label">${label}:</span>
      <button type="button" class="highlight-name" data-player-id="${h.player_id}" data-transfer-date="${h.transfer_date}" aria-label="View transfer details: ${h.name}">${h.name}</button>
      (${h.transfer_date.slice(0, 7)})${extra ? ` - ${extra(h)}` : ""}${scoreClause}
    </div>
  `;
}

/** Render one transfer's score-component breakdown (label + bar + hover tooltip), same shape as browse.js/loans.js/player.js. */
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

/**
 * Fetch and render one highlighted transfer's full breakdown card in place
 * of the club report card currently showing - a "← Back" link returns to
 * it (state.openClub is still intact, this never leaves the modal).
 */
async function showTransferDetail(playerId, transferDate) {
  state.openTransfer = { playerId, transferDate };
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  try {
    const res = await fetch(`/api/transfers/detail?player_id=${playerId}&transfer_date=${transferDate}`);
    if (!res.ok) throw new Error("Could not load this transfer.");
    const ex = await res.json();
    const years = (ex.tenure_days / 365.25).toFixed(1);
    content.innerHTML = `
      <div class="example-card" style="border:none; padding:1.25rem;">
        <a href="#" class="browse-link" id="back-to-club" style="margin:0 0 0.75rem;">&larr; Back to ${state.openClub.club_name} report card</a>
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(ex.success_score)}">${ex.success_score}</div>
        <div class="tenure-note">
          Scored over ${years} years at the club${ex.still_at_club ? " (still there)" : " (before leaving)"}
        </div>
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `;
    document.getElementById("back-to-club").addEventListener("click", (e) => {
      e.preventDefault();
      showCard(state.openClub);
    });
  } catch (e) {
    content.innerHTML = `<div class="error-box">${e.message}</div>`;
  }
}

/**
 * Build and open the report-card modal for one clicked club row - entirely
 * from data already returned by /api/clubs/leaderboard (best/worst
 * signing/flip/departure are precomputed server-side - see
 * build_club_report_cards in app/main.py), so no second request is needed.
 */
function showCard(club) {
  state.openClub = club;
  state.openTransfer = null;
  const content = document.getElementById("card-modal-content");
  const buyerSection = club.transfers_in ? `
    <h3>As a buyer</h3>
    <p>${club.transfers_in} incoming transfers, averaging <strong style="color:${scoreColor(club.avg_incoming_score)}">${club.avg_incoming_score}</strong> / 100, ${formatMoney(club.total_spent)} spent.</p>
    ${highlightLine("Best signing", club.best_signing)}
    ${highlightLine("Worst signing", club.worst_signing)}
  ` : `<h3>As a buyer</h3><p>No scored incoming transfers.</p>`;

  const resaleSection = club.resales_count ? `
    <h3>Buy, develop, resell</h3>
    <p>
      ${club.resales_count} of those signings were later resold, averaging
      <strong style="color:${scoreColor(club.avg_resale_profit_pct)}">${club.avg_resale_profit_pct}</strong> / 100 on the resale-profit percentile,
      a combined <strong style="color:${totalColor(club.total_resale_profit)}">${signedMoney(club.total_resale_profit)}</strong> across all of them.
    </p>
    ${highlightLine("Best flip", club.best_flip, h => `bought from ${h.bought_from} for ${formatMoney(h.fee_paid)}, resold for ${formatMoney(h.fee_received)}`)}
    ${highlightLine("Worst flip", club.worst_flip, h => `bought from ${h.bought_from} for ${formatMoney(h.fee_paid)}, resold for ${formatMoney(h.fee_received)}`)}
  ` : "";

  const sellerSection = club.transfers_out ? `
    <h3>Departures</h3>
    <p>${club.transfers_out} players left - the ones who left averaged <strong style="color:${scoreColor(club.avg_departure_score)}">${club.avg_departure_score}</strong> / 100 at their next stop.</p>
    ${highlightLine("Thrived elsewhere", club.best_departure)}
    ${highlightLine("Struggled elsewhere", club.worst_departure)}
  ` : `<h3>Departures</h3><p>No scored departures.</p>`;

  content.innerHTML = `
    <div class="example-card" style="border:none; padding:1.25rem;">
      <div class="name">${club.club_name}</div>
      <div class="route">${club.league}</div>
      ${buyerSection}
      ${resaleSection}
      ${sellerSection}
      <a class="browse-link" href="/browse.html?q=${encodeURIComponent(club.club_name)}">View every transfer involving ${club.club_name} on Browse &rarr;</a>
    </div>
  `;
  [...content.querySelectorAll(".highlight-name")].forEach(btn => {
    btn.addEventListener("click", () => showTransferDetail(btn.dataset.playerId, btn.dataset.transferDate));
  });
  cardModal.open();
}

/** Close the report-card modal. */
function closeCard() {
  state.openClub = null;
  state.openTransfer = null;
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
document.getElementById("league-select").addEventListener("change", resetAndLoad);
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
  if (state.openTransfer) {
    // Re-render in place without going through showCard (which would
    // clear state.openTransfer and drop the user back to the club view).
    showTransferDetail(state.openTransfer.playerId, state.openTransfer.transferDate);
  } else if (state.openClub) {
    showCard(state.openClub);
  }
});

loadFilters();
loadTable();
