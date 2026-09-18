const state = { offset: 0, total: 0, rows: [], openClub: null, openTransfer: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** The two sides of the head-to-head comparison (see wireClubAutocomplete/renderClubCompare below) - each is a full /api/clubs/report-card response (base stats + by_year + position_breakdown), or null before that side has a club selected. */
const clubCompare = { a: null, b: null };

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

/**
 * A club's total spend and average incoming score by year (see
 * /api/clubs/report-card's by_year), both indexed to their own first
 * year of data = 100 so a euro amount and a 0-100 score can share one
 * axis honestly - same trick as Analytics' market-over-time chart and
 * League Trends' own trend chart, indexed here against this one club's
 * own first year rather than a sitewide/leaguewide early-period average
 * (a per-club series is too short and idiosyncratic for that).
 */
function buildSpendQualityChart(byYear) {
  const W = 900, H = 240, padL = 40, padR = 16, padT = 16, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const years = byYear.map(d => d.year);
  const baseSpend = byYear[0].total_spent, baseScore = byYear[0].avg_score;
  const spendIdx = byYear.map(d => baseSpend > 0 ? (d.total_spent / baseSpend) * 100 : 100);
  const scoreIdx = byYear.map(d => baseScore > 0 ? (d.avg_score / baseScore) * 100 : 100);
  const allIdx = [...spendIdx, ...scoreIdx, 100];
  const maxIdx = Math.max(...allIdx) * 1.12;
  const minIdx = Math.min(0, Math.min(...allIdx) * 0.9);

  const x = (i) => years.length === 1 ? padL + chartW / 2 : padL + (i / (years.length - 1)) * chartW;
  const y = (v) => padT + (1 - (v - minIdx) / (maxIdx - minIdx)) * chartH;

  const baseline = `
    <line x1="${padL}" y1="${y(100).toFixed(1)}" x2="${W - padR}" y2="${y(100).toFixed(1)}" stroke="var(--border)" stroke-dasharray="4 3" stroke-width="1" />
    <text x="${padL}" y="${(y(100) - 5).toFixed(1)}" font-size="10" fill="var(--muted)">100 = ${years[0]}</text>
  `;

  function seriesPath(idxVals, rawVals, color, label, formatRaw) {
    const pts = idxVals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const dots = idxVals.map((v, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="4" fill="${color}"><title>${label} in ${years[i]}: ${formatRaw(rawVals[i])}</title></circle>`).join("");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" />${dots}`;
  }

  const yearLabels = years.map((yr, i) => `<text x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">${yr}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Spend and incoming quality index by year">
    ${baseline}
    ${seriesPath(spendIdx, byYear.map(d => d.total_spent), "var(--accent-mid)", "Spend", formatMoney)}
    ${seriesPath(scoreIdx, byYear.map(d => d.avg_score), "var(--accent)", "Avg incoming score", (v) => v.toFixed(1))}
    ${yearLabels}
  </svg>`;
}

/** A club's recruiting-by-position breakdown (see /api/clubs/report-card's position_breakdown) as labeled bar rows - the same .breakdown-row/.breakdown-bar-track/.breakdown-bar-fill markup the transfer-card score breakdown already uses elsewhere on the site, repurposed here for "avg incoming score per position" instead of "score component value." Positions below the backend's minimum sample are already excluded server-side. */
function renderPositionBreakdown(breakdown) {
  if (!breakdown.length) return `<p class="surprises-intro">Not enough incoming transfers in any one position yet.</p>`;
  return breakdown.map(p => `
    <div class="breakdown-row">
      <span class="breakdown-label">${p.position} <span style="color:var(--muted)">(${p.transfers})</span></span>
      <div class="breakdown-bar-track">
        <div class="breakdown-bar-fill" style="width:${p.avg_score}%; background:${scoreColor(p.avg_score)}"></div>
      </div>
      <span class="breakdown-value">${p.avg_score}</span>
    </div>
  `).join("");
}

/**
 * One row of the head-to-head comparison table. aRaw/bRaw drive the
 * winner highlight and are omitted entirely for stats with no clear
 * "higher is better" direction (total spent, transfer counts, departure
 * score - a club letting a lot of players go who then thrive elsewhere
 * isn't obviously good or bad) - only avg_incoming_score and
 * avg_resale_profit_pct get one, from renderClubCompare() below.
 */
function compareRow(label, aDisplay, bDisplay, aRaw, bRaw) {
  const aWins = aRaw != null && bRaw != null && aRaw > bRaw;
  const bWins = aRaw != null && bRaw != null && bRaw > aRaw;
  return `
    <tr>
      <td>${label}</td>
      <td class="${aWins ? "is-winner-cell" : ""}">${aDisplay}</td>
      <td class="${bWins ? "is-winner-cell" : ""}">${bDisplay}</td>
    </tr>
  `;
}

/** Render the head-to-head comparison (stat table + both clubs' spend-vs-quality charts + position breakdowns) once both sides have a club selected - clears the result area if either side is still unset. Also keeps the URL in sync, since club_a/club_b are part of this page's shareable state alongside the table's own filters (see syncURL). */
function renderClubCompare() {
  const container = document.getElementById("club-compare-result");
  syncURL();
  if (!clubCompare.a || !clubCompare.b) {
    container.innerHTML = "";
    return;
  }
  const [a, b] = [clubCompare.a, clubCompare.b];
  container.innerHTML = `
    <table class="club-compare-table">
      <thead><tr><th></th><th>${a.club_name}</th><th>${b.club_name}</th></tr></thead>
      <tbody>
        ${compareRow("Incoming transfers", a.transfers_in, b.transfers_in)}
        ${compareRow("Avg incoming score", scoreCell(a.avg_incoming_score, a.transfers_in), scoreCell(b.avg_incoming_score, b.transfers_in), a.avg_incoming_score, b.avg_incoming_score)}
        ${compareRow("Total spent", formatMoney(a.total_spent), formatMoney(b.total_spent))}
        ${compareRow("Avg resale profit", scoreCell(a.avg_resale_profit_pct, a.resales_count), scoreCell(b.avg_resale_profit_pct, b.resales_count), a.avg_resale_profit_pct, b.avg_resale_profit_pct)}
        ${compareRow("Departures", a.transfers_out, b.transfers_out)}
        ${compareRow("Avg departure score", scoreCell(a.avg_departure_score, a.transfers_out), scoreCell(b.avg_departure_score, b.transfers_out))}
      </tbody>
    </table>
    <div class="compare-grid">
      <div>
        <h3>${a.club_name}</h3>
        ${a.by_year.length ? `<div class="timeline-chart-wrap">${buildSpendQualityChart(a.by_year)}</div>` : `<p class="surprises-intro">Not enough year-by-year data yet.</p>`}
        ${renderPositionBreakdown(a.position_breakdown)}
        <a class="browse-link" href="#" data-open-club="a">View ${a.club_name}'s full report card &rarr;</a>
      </div>
      <div>
        <h3>${b.club_name}</h3>
        ${b.by_year.length ? `<div class="timeline-chart-wrap">${buildSpendQualityChart(b.by_year)}</div>` : `<p class="surprises-intro">Not enough year-by-year data yet.</p>`}
        ${renderPositionBreakdown(b.position_breakdown)}
        <a class="browse-link" href="#" data-open-club="b">View ${b.club_name}'s full report card &rarr;</a>
      </div>
    </div>
  `;
  container.querySelectorAll("[data-open-club]").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      showCard(clubCompare[link.dataset.openClub]);
    });
  });
}

/**
 * Debounced search-as-you-type dropdown for one input - same pattern as
 * compare.js's own wireAutocomplete (duplicated here per this session's
 * established per-page-JS-file convention, not shared as an import).
 * Keyboard-navigable (ArrowUp/Down, Enter, Escape) with the standard
 * ARIA combobox pattern, same reasoning as compare.js's version.
 */
function wireAutocomplete(input, list, endpoint, renderLabel, onSelect) {
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
      items = await res.json();
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

/** Wire one side ("a" or "b") of the head-to-head picker: search over /api/clubs/report-card-search (club_report_cards_df's own names, not clubs_lookup.csv - see that endpoint's docstring), then fetch the full report card for whichever club is selected. */
function wireClubAutocomplete(side) {
  const input = document.getElementById(`club-${side}-search`);
  const list = document.getElementById(`club-${side}-list`);
  wireAutocomplete(
    input, list, "/api/clubs/report-card-search",
    (c) => `${c.club_name} (${c.league})`,
    async (c) => {
      document.getElementById(`club-${side}-chip`).innerHTML = `<span class="selected-chip">${c.club_name}</span>`;
      const res = await fetch(`/api/clubs/report-card?name=${encodeURIComponent(c.club_name)}`);
      clubCompare[side] = res.ok ? await res.json() : null;
      renderClubCompare();
    },
  );
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
  syncURL();
}

/** Keep the address bar's query string in sync with the current search/filter/sort/page plus the head-to-head comparison's two selected clubs (if any), so this view - table and comparison alike - is bookmarkable and shareable. writeURLParams replaces the whole query string on every call, so anything not included here is dropped - this is the one place all of this page's shareable state comes together, rather than the table and the comparison each writing their own half and clobbering the other's. */
function syncURL() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  writeURLParams({
    q: document.getElementById("search-input").value.trim(),
    league: document.getElementById("league-select").value,
    sort, order,
    offset: state.offset || "",
    club_a: clubCompare.a ? clubCompare.a.club_name : "",
    club_b: clubCompare.b ? clubCompare.b.club_name : "",
  });
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
 * Build and open the report-card modal for one clicked club row - the
 * base sections (buyer/resale/departures prose and highlights) render
 * instantly from data already returned by /api/clubs/leaderboard
 * (precomputed server-side - see build_club_report_cards in
 * app/main.py), no request needed. The spend-vs-quality chart and
 * position breakdown are the exception - see loadClubDetailCharts below,
 * called at the end of this function - those two are deliberately not on
 * every leaderboard row (see /api/clubs/report-card's docstring), so
 * they render into a placeholder a moment after the rest of the card.
 */
function showCard(club) {
  state.openClub = club;
  state.openTransfer = null;
  const content = document.getElementById("card-modal-content");
  const trendClause = club.score_improvement != null
    ? ` Recruiting trend: <strong style="color:${totalColor(club.score_improvement)}">${club.score_improvement >= 0 ? "+" : ""}${club.score_improvement} pts</strong> (second half vs. first half of this club's incoming transfers, by date).`
    : "";
  const buyerSection = club.transfers_in ? `
    <h3>As a buyer</h3>
    <p>${club.transfers_in} incoming transfers, averaging <strong style="color:${scoreColor(club.avg_incoming_score)}">${club.avg_incoming_score}</strong> / 100, ${formatMoney(club.total_spent)} spent.${trendClause}</p>
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
      <div id="club-detail-charts"><p class="surprises-intro">Loading spend and position detail...</p></div>
      ${resaleSection}
      ${sellerSection}
      <a class="browse-link" href="/browse.html?q=${encodeURIComponent(club.club_name)}">View every transfer involving ${club.club_name} on Browse &rarr;</a>
    </div>
  `;
  [...content.querySelectorAll(".highlight-name")].forEach(btn => {
    btn.addEventListener("click", () => showTransferDetail(btn.dataset.playerId, btn.dataset.transferDate));
  });
  cardModal.open();
  loadClubDetailCharts(club.club_name);
}

/**
 * Fetch /api/clubs/report-card for the currently-open club and render its
 * spend-vs-quality chart + position breakdown into the #club-detail-charts
 * placeholder left by showCard() above. Re-checks state.openClub before
 * writing anything - if the user closed the modal or clicked a different
 * club while this was in flight, this response is stale and should never
 * overwrite whatever (or whoever) is showing now.
 */
async function loadClubDetailCharts(clubName) {
  try {
    const res = await fetch(`/api/clubs/report-card?name=${encodeURIComponent(clubName)}`);
    if (!state.openClub || state.openClub.club_name !== clubName) return;
    const target = document.getElementById("club-detail-charts");
    if (!target) return;
    if (!res.ok) throw new Error("Could not load this club's detail.");
    const data = await res.json();
    target.innerHTML = `
      ${data.by_year.length ? `
        <h3>Spend vs. incoming quality by year</h3>
        <div class="timeline-chart-wrap">${buildSpendQualityChart(data.by_year)}</div>
        <div class="timeline-legend">
          <span><span class="legend-dot" style="background:var(--accent-mid); border:none;"></span> Spend (indexed)</span>
          <span><span class="legend-dot" style="background:var(--accent); border:none;"></span> Avg incoming score (indexed)</span>
        </div>
      ` : ""}
      ${data.position_breakdown.length ? `<h3>Recruiting by position</h3>${renderPositionBreakdown(data.position_breakdown)}` : ""}
    `;
  } catch (e) {
    const target = document.getElementById("club-detail-charts");
    if (target && state.openClub && state.openClub.club_name === clubName) target.innerHTML = "";
  }
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
  // The comparison's spend chart formats money too - re-render so its
  // tooltips reflect the new currency (a no-op, safely, when neither side
  // has a club picked yet).
  renderClubCompare();
});

/** Toggle the head-to-head comparison card open/closed, same pattern as Model vs Reality's collapsible chart. */
document.getElementById("compare-toggle").addEventListener("click", (e) => {
  const expanded = e.currentTarget.getAttribute("aria-expanded") === "true";
  e.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("compare-body").hidden = expanded;
});

wireClubAutocomplete("a");
wireClubAutocomplete("b");

// Restore search/league/sort/page straight from the URL (a bookmarked or
// shared link) so landing here already shows that view, not always the
// unfiltered default. league can't be applied until loadFilters() has
// populated its <option>s, so it waits on that specifically - q/sort/
// offset don't depend on it and apply immediately so the very first
// loadTable() call already reflects them.
const urlParams = readURLParams();
if (urlParams.q) document.getElementById("search-input").value = urlParams.q;
if (urlParams.sort && urlParams.order) {
  const sortSelect = document.getElementById("sort-select");
  const sortValue = `${urlParams.sort}:${urlParams.order}`;
  if ([...sortSelect.options].some(o => o.value === sortValue)) sortSelect.value = sortValue;
}
state.offset = parseInt(urlParams.offset, 10) || 0;

if (urlParams.league) {
  loadFilters().then(() => {
    document.getElementById("league-select").value = urlParams.league;
    loadTable();
  });
} else {
  loadFilters();
  loadTable();
}

/** Look up one side of a shared comparison link by exact club name, without touching any state/DOM - split from applying it so restoreClubCompareFromURL() below can resolve both sides first (same reasoning as Compare's restoreFromURL: a stale/mistyped club_a shouldn't leave club_b looking normal while club_a silently fails). */
async function fetchClubForRestore(name) {
  const res = await fetch(`/api/clubs/report-card?name=${encodeURIComponent(name)}`);
  return res.ok ? await res.json() : null;
}

if (urlParams.club_a || urlParams.club_b) {
  Promise.all([
    urlParams.club_a ? fetchClubForRestore(urlParams.club_a) : null,
    urlParams.club_b ? fetchClubForRestore(urlParams.club_b) : null,
  ]).then(([a, b]) => {
    if (a) {
      clubCompare.a = a;
      document.getElementById("club-a-search").value = a.club_name;
      document.getElementById("club-a-chip").innerHTML = `<span class="selected-chip">${a.club_name}</span>`;
    }
    if (b) {
      clubCompare.b = b;
      document.getElementById("club-b-search").value = b.club_name;
      document.getElementById("club-b-chip").innerHTML = `<span class="selected-chip">${b.club_name}</span>`;
    }
    renderClubCompare();
  });
}
