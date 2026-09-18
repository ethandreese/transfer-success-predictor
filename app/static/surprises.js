const state = { offset: 0, total: 0, openCard: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Green for an overachiever (actual beat the model), red for a bust (actual fell short) - unlike scoreColor, this is signed around 0, not an absolute 0-100 scale. */
function deltaColor(delta) {
  return delta >= 0 ? "var(--accent)" : "var(--accent-bad)";
}

/** "+45.7" / "-57.7", always signed so the direction reads at a glance. */
function signed(n) {
  return `${n >= 0 ? "+" : ""}${n}`;
}

function linearScale(d0, d1, r0, r1) {
  return (v) => d1 === d0 ? (r0 + r1) / 2 : r0 + (v - d0) / (d1 - d0) * (r1 - r0);
}

/** Horizontal gridlines + left-edge labels at 0/25/50/75/100 - copied from analytics.js's chart helpers (same duplicated-per-page pattern every chart-drawing function on this site already follows). */
function scoreGridlines(y, padL, padR, W) {
  return [0, 25, 50, 75, 100].map(score => `
    <line x1="${padL}" y1="${y(score).toFixed(1)}" x2="${W - padR}" y2="${y(score).toFixed(1)}" stroke="var(--border)" stroke-width="1" />
    <text x="${padL - 6}" y="${(y(score) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">${score}</text>
  `).join("");
}

/** One clickable scatter point: a small visible dot plus a larger (invisible) circle purely to enlarge the tap/click target. Wrapped in a <g class="scatter-pt"> carrying the transfer's player_id/transfer_date/predicted_score/surprise_delta as data attributes, read by wireScatterClicks() via event delegation to open that transfer's full card (with the model-vs-reality banner) on click/tap. */
function scatterPoint(cx, cy, color, opacity, playerId, transferDate, predictedScore, delta, tooltip) {
  return `
    <g class="scatter-pt" data-player-id="${playerId}" data-transfer-date="${transferDate}" data-predicted="${predictedScore}" data-delta="${delta}" style="cursor:pointer">
      <circle cx="${cx}" cy="${cy}" r="6" fill="transparent" />
      <circle cx="${cx}" cy="${cy}" r="2.3" fill="${color}" opacity="${opacity}" />
      <title>${tooltip}</title>
    </g>
  `;
}

/** Turn /api/surprises/scatter's columnar payload ({name: [...], predicted_score: [...], ...}) into one row object per transfer - same convention Analytics uses for /api/analytics' scatter. */
function zipScatter(scatter) {
  const rows = [];
  for (let i = 0; i < scatter.name.length; i++) {
    rows.push({
      player_id: scatter.player_id[i],
      transfer_date: scatter.transfer_date[i],
      name: scatter.name[i],
      predicted_score: scatter.predicted_score[i],
      success_score: scatter.success_score[i],
      surprise_delta: scatter.surprise_delta[i],
    });
  }
  return rows;
}

/**
 * Every scored transfer with a held-out prediction, predicted score (x)
 * against actual score (y) - both already 0-100, so a plain linear scale
 * on both axes, with a dashed y=x reference line: above it, the model
 * undersold the transfer (actual beat predicted); below it, it oversold
 * it. Dots use deltaColor (green/red), not scoreColor - the color here
 * answers "which side of the diagonal", the same question the line
 * itself asks, not "how good was the actual score" the way every other
 * chart's dot color does.
 */
function buildPredictedActualChart(rows) {
  const W = 900, H = 340, padL = 50, padR = 16, padT = 14, padB = 30;
  const x = linearScale(0, 100, padL, W - padR);
  const y = linearScale(0, 100, H - padB, padT);

  const diagonal = `<line x1="${x(0).toFixed(1)}" y1="${y(0).toFixed(1)}" x2="${x(100).toFixed(1)}" y2="${y(100).toFixed(1)}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7" />`;

  const dots = rows.map(r => scatterPoint(
    x(r.predicted_score).toFixed(1), y(r.success_score).toFixed(1), deltaColor(r.surprise_delta), 0.4,
    r.player_id, r.transfer_date, r.predicted_score, r.surprise_delta,
    `${r.name}: predicted ${r.predicted_score}, actual ${r.success_score} (${signed(r.surprise_delta)}) - click for details`,
  )).join("");

  const ticks = [0, 25, 50, 75, 100];
  const xLabels = ticks.map(t => `<text x="${x(t).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${t}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Predicted vs actual success score scatter plot">
    ${scoreGridlines(y, padL, padR, W)}
    ${diagonal}
    ${dots}
    ${xLabels}
    <text x="${W - padR}" y="${H - padB - 6}" text-anchor="end" font-size="10" fill="var(--muted)">predicted score &rarr;</text>
  </svg>`;
}

/** Build the /api/surprises query string from the current search box, filter dropdowns, sort selection, and pagination offset. Page size comes from the Settings panel (default 25). */
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

/** Fetch the distinct positions/leagues from /api/filters and populate the two filter <select> dropdowns - the same universe /api/transfers uses, since surprises are a subset of that same data. */
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

/** Fetch the current page of surprises (per currentParams()) and render the table body, pagination controls, and per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="8">Loading...</td></tr>`;
  const res = await fetch(`/api/surprises?${currentParams().toString()}`);
  const data = await res.json();
  state.total = data.total;

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="8">No transfers match these filters.</td></tr>`;
  } else {
    tbody.innerHTML = data.results.map(r => `
      <tr data-player-id="${r.player_id}" data-transfer-date="${r.transfer_date}" data-predicted="${r.predicted_score}" data-delta="${r.surprise_delta}" tabindex="0" role="button" aria-label="View transfer details: ${r.name} to ${r.to_club}">
        <td>${r.name}</td>
        <td>${r.position}</td>
        <td>${r.from_club} &rarr; ${r.to_club}</td>
        <td>${r.to_league}</td>
        <td>${r.transfer_date.slice(0, 7)}</td>
        <td>${r.predicted_score}</td>
        <td style="color:${scoreColor(r.success_score)}; font-weight:700">${r.success_score}</td>
        <td style="color:${deltaColor(r.surprise_delta)}; font-weight:700">${signed(r.surprise_delta)}</td>
      </tr>
    `).join("");
    [...tbody.querySelectorAll("tr")].forEach(row => {
      const open = () => showCard(row.dataset.playerId, row.dataset.transferDate, row.dataset.predicted, row.dataset.delta);
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

/** Keep the address bar's query string in sync with the current search/filter/sort/page, so this view is bookmarkable and shareable - see writeURLParams in settings.js. */
function syncURL() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  writeURLParams({
    q: document.getElementById("search-input").value.trim(),
    position: document.getElementById("position-select").value,
    league: document.getElementById("league-select").value,
    sort, order,
    offset: state.offset || "",
  });
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

/**
 * Open the modal and fetch+render the full transfer card for one clicked
 * row, via /api/transfers/detail - same card as Browse, plus a banner
 * showing what the held-out model predicted vs. what actually happened
 * (predictedScore/delta come from the already-fetched list row, not a
 * second /api/surprises round-trip). Remembers the open card (state.openCard)
 * so a currency/theme change can refresh it in place.
 */
async function showCard(playerId, transferDate, predictedScore, delta) {
  state.openCard = { playerId, transferDate, predictedScore, delta };
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
        <div class="surprise-banner">
          A held-out model predicted <strong>${predictedScore}</strong> from pre-transfer data alone -
          the real outcome came in <strong style="color:${deltaColor(delta)}">${signed(delta)}</strong> points off that.
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

/**
 * Delegate clicks on the predicted-vs-actual chart's wrapper div to
 * whichever .scatter-pt group was hit, rather than attaching a listener
 * per point (there are ~7,800 of them) - same reasoning and pattern as
 * analytics.js's own wireScatterClicks.
 */
function wireScatterClicks(containerId) {
  document.getElementById(containerId).addEventListener("click", (e) => {
    const pt = e.target.closest(".scatter-pt");
    if (pt) showCard(pt.dataset.playerId, pt.dataset.transferDate, pt.dataset.predicted, pt.dataset.delta);
  });
}

/**
 * The model's typical miss (mean absolute surprise, in points) per year -
 * lower is better - from /api/surprises/scatter's accuracy_by_year. A
 * single plain line, not indexed to 100 the way the site's multi-series
 * charts are (League Trends, Analytics' market-over-time) - there's only
 * one series here, so raw point values are already directly comparable
 * year to year without needing a shared index.
 */
function buildAccuracyChart(accuracyByYear) {
  const W = 900, H = 220, padL = 34, padR = 16, padT = 16, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const years = accuracyByYear.map(d => d.year);
  const maes = accuracyByYear.map(d => d.mae);
  const minMae = Math.min(...maes) * 0.9, maxMae = Math.max(...maes) * 1.1;

  const x = (i) => years.length === 1 ? padL + chartW / 2 : padL + (i / (years.length - 1)) * chartW;
  const y = (v) => padT + (1 - (v - minMae) / (maxMae - minMae)) * chartH;

  const points = maes.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const dots = accuracyByYear.map((d, i) => {
    const cx = x(i).toFixed(1), cy = y(d.mae).toFixed(1);
    const label = `${d.year}: typical miss ${d.mae.toFixed(1)} pts (${d.n.toLocaleString()} predictions)`;
    return `
      <g style="cursor:pointer" data-tooltip="${label}">
        <circle cx="${cx}" cy="${cy}" r="9" fill="transparent" />
        <circle cx="${cx}" cy="${cy}" r="4" fill="var(--trend-line)" stroke="var(--panel)" stroke-width="1" />
        <title>${label}</title>
      </g>
    `;
  }).join("");

  const yTicks = [minMae, (minMae + maxMae) / 2, maxMae];
  const yLabels = yTicks.map(v => `<text x="${padL - 6}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">${v.toFixed(1)}</text>`).join("");
  const yearLabels = years.map((yr, i) => `<text x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">${yr}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Model prediction error by year">
    <polyline points="${points}" fill="none" stroke="var(--trend-line)" stroke-width="2" />
    ${dots}
    ${yLabels}
    ${yearLabels}
  </svg>`;
}

const chartTooltip = document.getElementById("chart-tooltip");

/** Show the shared chart-tooltip box near a click/tap position, clamped so it never runs off the right/bottom edge - the real interaction behind the accuracy chart's yearly points, same reasoning as analytics.js's identical helper: a native <title> hover tooltip never fires on a touch device at all. */
function showChartTooltip(text, clientX, clientY) {
  chartTooltip.textContent = text;
  chartTooltip.hidden = false;
  const pad = 14;
  const rect = chartTooltip.getBoundingClientRect();
  let left = clientX + pad, top = clientY + pad;
  if (left + rect.width > window.innerWidth) left = clientX - rect.width - pad;
  if (top + rect.height > window.innerHeight) top = clientY - rect.height - pad;
  chartTooltip.style.left = `${Math.max(4, left)}px`;
  chartTooltip.style.top = `${Math.max(4, top)}px`;
}

function hideChartTooltip() {
  chartTooltip.hidden = true;
}

document.addEventListener("click", (e) => {
  const marked = e.target.closest("[data-tooltip]");
  if (marked) showChartTooltip(marked.dataset.tooltip, e.clientX, e.clientY);
  else if (!e.target.closest("#chart-tooltip")) hideChartTooltip();
});

/** Fetch every scored transfer's held-out prediction (unfiltered, unpaginated - see /api/surprises/scatter) and render the predicted-vs-actual chart plus the model-accuracy-over-time chart - both come from the same response, so one fetch covers both cards. Independent of the table below: these charts show the model's overall calibration, not whatever filter/page the table currently has applied, so they're fetched once and never reloaded by a filter/sort/page change. */
async function loadScatter() {
  const res = await fetch("/api/surprises/scatter");
  const data = await res.json();
  const rows = zipScatter(data);
  document.getElementById("predicted-actual-chart").innerHTML = buildPredictedActualChart(rows);
  const overCount = rows.filter(r => r.surprise_delta >= 0).length;
  const pct = Math.round((overCount / rows.length) * 100);
  document.getElementById("predicted-actual-desc").textContent =
    `${rows.length.toLocaleString()} scored transfers with a held-out prediction. ${overCount.toLocaleString()} (${pct}%) landed on or above the model's guess, ${(rows.length - overCount).toLocaleString()} below it. Click/tap any point for that transfer's full breakdown.`;

  const accuracy = data.accuracy_by_year;
  document.getElementById("accuracy-chart").innerHTML = buildAccuracyChart(accuracy);
  const first = accuracy[0], last = accuracy[accuracy.length - 1];
  const trendWord = last.mae < first.mae ? "improved" : "worsened";
  document.getElementById("accuracy-desc").textContent =
    `The held-out model's typical miss by transfer year, ${first.year}–${last.year} (the current in-progress year excluded, same reasoning as every other by-year chart on the site). It's ${trendWord} from ${first.mae.toFixed(1)} to ${last.mae.toFixed(1)} points. Click/tap any point for that year's exact numbers.`;
}

/** Toggle the model-accuracy chart card open/closed, same pattern as the predicted-vs-actual card. */
document.getElementById("accuracy-toggle").addEventListener("click", (e) => {
  const expanded = e.currentTarget.getAttribute("aria-expanded") === "true";
  e.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("accuracy-body").hidden = expanded;
});

/** Toggle the predicted-vs-actual chart card open/closed - it's the one card-length chart on a page that's otherwise a table, so letting it collapse saves scrolling past ~7,800 plotted points to reach the filters. */
document.getElementById("predicted-actual-toggle").addEventListener("click", (e) => {
  const expanded = e.currentTarget.getAttribute("aria-expanded") === "true";
  e.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("predicted-actual-body").hidden = expanded;
});

// Close the modal via the X button, a click on the dimmed backdrop (but not
// the card itself), or the Escape key.
document.getElementById("card-modal-close").addEventListener("click", closeCard);
document.getElementById("card-modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-modal-backdrop") closeCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeCard(); hideChartTooltip(); }
});

document.getElementById("search-input").addEventListener("input", () => {
  clearTimeout(window.__searchDebounce);
  window.__searchDebounce = setTimeout(resetAndLoad, 300);
});
document.getElementById("position-select").addEventListener("change", resetAndLoad);
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
  if (state.openCard) {
    const { playerId, transferDate, predictedScore, delta } = state.openCard;
    showCard(playerId, transferDate, predictedScore, delta);
  }
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

if (urlParams.position || urlParams.league) {
  loadFilters().then(() => {
    if (urlParams.position) document.getElementById("position-select").value = urlParams.position;
    if (urlParams.league) document.getElementById("league-select").value = urlParams.league;
    loadTable();
  });
} else {
  loadFilters();
  loadTable();
}

wireScatterClicks("predicted-actual-chart");
loadScatter();
