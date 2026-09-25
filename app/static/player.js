const state = { openStop: null, careerA: null, careerB: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Escape a value for safe interpolation inside an HTML attribute (e.g. a title="..." tooltip) - real club/player names can contain a literal " or & (e.g. Brighton & Hove Albion, or a club whose native name is quoted), which would otherwise break out of the attribute. */
function escapeAttr(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * "3.1y" / "8mo". A permanent transfer always renders in years, matching
 * browse.js's tenureDisplay exactly (no duration threshold there - every
 * permanent stop is scored over at least the whole first season by
 * construction). A loan picks the unit by actual duration, matching
 * loans.js's durationDisplay (days >= 365 -> years, else months) - a
 * duration threshold matters there since most loans are under a year but
 * a real minority run multi-year.
 */
function tenureDisplay(days, type) {
  if (type !== "loan") return `${(days / 365.25).toFixed(1)}y`;
  return days >= 365 ? `${(days / 365.25).toFixed(1)}y` : `${Math.round(days / 30.44)}mo`;
}

/**
 * Wire a text input to a debounced search-as-you-type dropdown (same
 * pattern as app.js/compare.js's autocomplete, duplicated here since this
 * page only ever needs one instance). Keyboard-navigable
 * (ArrowUp/Down/Enter/Escape) with the standard ARIA combobox pattern.
 */
function setupAutocomplete({ inputId, listId, endpoint, onSelect, renderLabel }) {
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
      items = await res.json();
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

/** A small star above a career's single highest-scored stop - "where was the peak" at a glance on a long timeline. Shared by the solo and overlay chart builders below; `points` are the same screen coordinates each builder already computed for its dots, so this never recomputes the x/y scale itself. */
function peakMarker(stops, points) {
  let peakIndex = 0;
  for (let i = 1; i < stops.length; i++) {
    if (stops[i].score > stops[peakIndex].score) peakIndex = i;
  }
  const { x: cx, y: cy } = points[peakIndex];
  return `<text x="${cx}" y="${cy - 12}" text-anchor="middle" font-size="13" fill="var(--accent-mid)"><title>Career peak: ${stops[peakIndex].score}</title>&#9733;</text>`;
}

/**
 * Build the career timeline as an inline SVG: one point per stop,
 * positioned by its real transfer_date along the x-axis (not just evenly
 * spaced by index) so a long gap between moves or a flurry of loans in
 * one year both read honestly, connected by a plain line. A permanent
 * transfer is a solid dot; a loan is a hollow ring (same color scale,
 * different shape) so the two are visually distinct without a second
 * legend of colors. Score gridlines at 0/25/50/75/100 for scale. Returns
 * the SVG markup as a string - the caller attaches click handlers to
 * `.timeline-point` after inserting it into the DOM.
 */
function buildTimelineSVG(stops) {
  const W = 900, H = 220, padL = 34, padR = 16, padT = 14, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const dates = stops.map(s => new Date(s.transfer_date).getTime());
  const minDate = Math.min(...dates), maxDate = Math.max(...dates);
  const span = maxDate - minDate;

  const x = (d) => span === 0 ? padL + chartW / 2 : padL + ((d - minDate) / span) * chartW;
  const y = (score) => padT + ((100 - score) / 100) * chartH;

  const gridlines = [0, 25, 50, 75, 100].map(score => `
    <line x1="${padL}" y1="${y(score)}" x2="${W - padR}" y2="${y(score)}" stroke="var(--border)" stroke-width="1" />
    <text x="${padL - 6}" y="${y(score) + 3}" text-anchor="end" font-size="10" fill="var(--muted)">${score}</text>
  `).join("");

  const points = stops.map(s => ({ x: x(new Date(s.transfer_date).getTime()), y: y(s.score) }));
  const line = points.length > 1
    ? `<polyline points="${points.map(p => `${p.x},${p.y}`).join(" ")}" fill="none" stroke="var(--muted)" stroke-width="1.5" opacity="0.5" />`
    : "";

  const dots = stops.map((s, i) => {
    const { x: cx, y: cy } = points[i];
    const color = scoreColor(s.score);
    const circle = s.type === "loan"
      ? `<circle cx="${cx}" cy="${cy}" r="6" fill="var(--panel)" stroke="${color}" stroke-width="3" />`
      : `<circle cx="${cx}" cy="${cy}" r="7" fill="${color}" stroke="var(--panel)" stroke-width="2" />`;
    const label = `${s.from_club} → ${s.to_club} (${s.transfer_date.slice(0, 7)}): ${s.score}`;
    return `<g class="timeline-point" data-player="a" data-index="${i}" style="cursor:pointer">${circle}<title>${label}</title></g>`;
  }).join("");

  const firstYear = stops[0].transfer_date.slice(0, 4);
  const lastYear = stops[stops.length - 1].transfer_date.slice(0, 4);
  const yearLabels = `
    <text x="${padL}" y="${H - 6}" font-size="10" fill="var(--muted)">${firstYear}</text>
    <text x="${W - padR}" y="${H - 6}" text-anchor="end" font-size="10" fill="var(--muted)">${lastYear}</text>
  `;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Career timeline chart">
    ${gridlines}
    ${line}
    ${dots}
    ${peakMarker(stops, points)}
    ${firstYear !== lastYear ? yearLabels : ""}
  </svg>`;
}

/**
 * Both careers plotted on one shared chart, real calendar date on the
 * x-axis (same convention as the solo chart, not age-normalized - this
 * is "who was doing what and when," a genuinely different question from
 * the shape-only comparison nearest_similar_careers answers). Career A
 * keeps the solo chart's circle/ring shapes; Career B reuses square/
 * hollow-square instead, so which dot belongs to which player reads at a
 * glance without needing a color of its own - dot color is still
 * reserved for score, the one visual language every chart on this site
 * shares. A dashed connecting line for B backs up the same distinction
 * for anyone tracing a path rather than reading individual dots.
 */
function buildOverlayTimelineSVG(careerA, careerB) {
  const W = 900, H = 260, padL = 34, padR = 16, padT = 14, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const allStops = [...careerA.stops, ...careerB.stops];
  const dates = allStops.map(s => new Date(s.transfer_date).getTime());
  const minDate = Math.min(...dates), maxDate = Math.max(...dates);
  const span = maxDate - minDate;

  const x = (d) => span === 0 ? padL + chartW / 2 : padL + ((d - minDate) / span) * chartW;
  const y = (score) => padT + ((100 - score) / 100) * chartH;

  const gridlines = [0, 25, 50, 75, 100].map(score => `
    <line x1="${padL}" y1="${y(score)}" x2="${W - padR}" y2="${y(score)}" stroke="var(--border)" stroke-width="1" />
    <text x="${padL - 6}" y="${y(score) + 3}" text-anchor="end" font-size="10" fill="var(--muted)">${score}</text>
  `).join("");

  function seriesMarkup(career, playerKey, squares) {
    const points = career.stops.map(s => ({ x: x(new Date(s.transfer_date).getTime()), y: y(s.score) }));
    const line = points.length > 1
      ? `<polyline points="${points.map(p => `${p.x},${p.y}`).join(" ")}" fill="none" stroke="var(--muted)" stroke-width="1.5" opacity="0.5" ${squares ? 'stroke-dasharray="6 4"' : ""} />`
      : "";
    const dots = career.stops.map((s, i) => {
      const { x: cx, y: cy } = points[i];
      const color = scoreColor(s.score);
      const shape = squares
        ? (s.type === "loan"
            ? `<rect x="${cx - 6}" y="${cy - 6}" width="12" height="12" fill="var(--panel)" stroke="${color}" stroke-width="3" />`
            : `<rect x="${cx - 6}" y="${cy - 6}" width="12" height="12" fill="${color}" stroke="var(--panel)" stroke-width="2" />`)
        : (s.type === "loan"
            ? `<circle cx="${cx}" cy="${cy}" r="6" fill="var(--panel)" stroke="${color}" stroke-width="3" />`
            : `<circle cx="${cx}" cy="${cy}" r="7" fill="${color}" stroke="var(--panel)" stroke-width="2" />`);
      const label = `${career.name}: ${s.from_club} → ${s.to_club} (${s.transfer_date.slice(0, 7)}), ${s.score}`;
      return `<g class="timeline-point" data-player="${playerKey}" data-index="${i}" style="cursor:pointer">${shape}<title>${label}</title></g>`;
    }).join("");
    return line + dots + peakMarker(career.stops, points);
  }

  const sortedDates = [...dates].sort((a, b) => a - b);
  const firstYear = new Date(sortedDates[0]).getFullYear();
  const lastYear = new Date(sortedDates[sortedDates.length - 1]).getFullYear();
  const yearLabels = `
    <text x="${padL}" y="${H - 6}" font-size="10" fill="var(--muted)">${firstYear}</text>
    <text x="${W - padR}" y="${H - 6}" text-anchor="end" font-size="10" fill="var(--muted)">${lastYear}</text>
  `;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Overlaid career timeline comparison">
    ${gridlines}
    ${seriesMarkup(careerA, "a", false)}
    ${seriesMarkup(careerB, "b", true)}
    ${firstYear !== lastYear ? yearLabels : ""}
  </svg>`;
}

/** Render the player header - name/position/current club/span of years for a solo view, or "A vs. B" plus both stop counts once a comparison is active. */
function renderPlayerHeader(careerA, careerB) {
  const allStops = careerB ? [...careerA.stops, ...careerB.stops] : careerA.stops;
  const sorted = [...allStops].sort((x, y) => x.transfer_date.localeCompare(y.transfer_date));
  const years = `${sorted[0].transfer_date.slice(0, 4)}–${sorted[sorted.length - 1].transfer_date.slice(0, 4)}`;
  if (careerB) {
    document.getElementById("player-header").innerHTML = `
      <div class="name" style="font-size:1.3rem;">${careerA.name} vs. ${careerB.name}</div>
      <div class="route">${careerA.stops.length} + ${careerB.stops.length} scored moves combined (${years})</div>
    `;
    return;
  }
  const meta = [careerA.position, careerA.current_club ? `currently at ${careerA.current_club}` : null].filter(Boolean).join(" · ");
  document.getElementById("player-header").innerHTML = `
    <div class="name" style="font-size:1.3rem;">${careerA.name}</div>
    <div class="route">${meta ? meta + " · " : ""}${careerA.stops.length} scored move${careerA.stops.length === 1 ? "" : "s"} (${years})</div>
  `;
}

/** Merge both careers' stops into one date-sorted list, each tagged with which player it belongs to (playerKey/playerName) - the comparison table's row data. */
function mergedStopsWithPlayer(careerA, careerB) {
  const tagged = [
    ...careerA.stops.map(s => ({ ...s, playerKey: "a", playerName: careerA.name })),
    ...careerB.stops.map(s => ({ ...s, playerKey: "b", playerName: careerB.name })),
  ];
  tagged.sort((x, y) => x.transfer_date.localeCompare(y.transfer_date));
  return tagged;
}

/** Render the plain-table fallback/detail list of every stop below the chart - an extra leading Player column once a comparison is active, so a merged, interleaved list still says whose move each row is. */
function renderStopsTable(stops, showPlayerColumn) {
  document.getElementById("stops-table").classList.toggle("compare-mode", showPlayerColumn);
  document.getElementById("stops-table-head").innerHTML = `
    ${showPlayerColumn ? "<th>Player</th>" : ""}
    <th>Date</th><th>Route</th><th>Type</th><th>Age</th><th>Score</th>
  `;
  document.getElementById("stops-body").innerHTML = stops.map((s, i) => `
    <tr data-index="${i}" tabindex="0" role="button" aria-label="View details: ${s.from_club} to ${s.to_club}">
      ${showPlayerColumn ? `<td title="${escapeAttr(s.playerName)}">${s.playerName}</td>` : ""}
      <td>${s.transfer_date.slice(0, 7)}</td>
      <td title="${escapeAttr(s.from_club + " → " + s.to_club)}">${s.from_club} &rarr; ${s.to_club}</td>
      <td>${s.type === "loan" ? "Loan" : "Permanent"}</td>
      <td>${s.age_at_transfer}</td>
      <td style="color:${scoreColor(s.score)}; font-weight:700">${s.score}</td>
    </tr>
  `).join("");
}

/** Swap the legend between solo mode (permanent/loan shape key) and comparison mode (which shape belongs to which player, since dot color is reserved for score in both modes). */
function renderLegend(careerA, careerB) {
  const legend = document.getElementById("timeline-legend");
  if (!careerB) {
    legend.innerHTML = `
      <span><span class="legend-dot legend-dot-permanent"></span> Permanent transfer</span>
      <span><span class="legend-dot legend-dot-loan"></span> Loan</span>
      <span>&#9733; Career peak</span>
      <span class="legend-note">Colored by score: green high, red low</span>
    `;
    return;
  }
  legend.innerHTML = `
    <span>&#9679; / &#9675; ${careerA.name} (permanent/loan)</span>
    <span>&#9632; / &#9633; ${careerB.name} (permanent/loan)</span>
    <span>&#9733; Career peak</span>
    <span class="legend-note">Colored by score: green high, red low</span>
  `;
}

/** Render "similar career shape" suggestions (see /api/players/{id}/career's similar_careers) as clickable pills - clicking one loads it as the comparison side, so the claimed similarity is immediately checkable on the overlay chart rather than just asserted. Clears the section entirely (rather than an empty-state message) when there's nothing to suggest, e.g. a one-stop career with no real "shape" to match. */
function renderSimilarCareers(entries) {
  const container = document.getElementById("similar-careers-section");
  if (!entries.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `
    <h3>Similar Career Shape</h3>
    <p class="surprises-intro">Other players whose career started, ended, and swung a similar way. Click one to compare them side by side.</p>
    ${entries.map(e => `<button type="button" class="similar-career-btn" data-player-id="${e.player_id}">${e.name}</button>`).join("")}
  `;
  container.querySelectorAll(".similar-career-btn").forEach(btn => {
    btn.addEventListener("click", () => loadPlayerB(Number(btn.dataset.playerId)));
  });
}

/** Render one card's score-component breakdown (label + bar + hover tooltip), same shape as browse.js/loans.js. */
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
 * Open the modal and fetch+render the full card for one clicked stop -
 * /api/transfers/detail for a permanent move, /api/loans/detail for a
 * loan, since the two live in separate tables with slightly different
 * field names (success_score/loan_success_score, still_at_club/
 * still_on_loan) - see build_transfer_card/build_loan_card in app/main.py.
 * playerId is explicit (not read from a single "current player" global)
 * since a comparison view has two, and a clicked stop's own player_id
 * depends on which of the two it came from.
 */
async function showCard(stop, playerId) {
  state.openStop = { stop, playerId };
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  cardModal.open();
  try {
    const endpoint = stop.type === "loan" ? "/api/loans/detail" : "/api/transfers/detail";
    const res = await fetch(`${endpoint}?player_id=${playerId}&transfer_date=${stop.transfer_date}`);
    if (!res.ok) throw new Error("Could not load this transfer.");
    const ex = await res.json();
    const score = stop.type === "loan" ? ex.loan_success_score : ex.success_score;
    const stillThere = stop.type === "loan" ? ex.still_on_loan : ex.still_at_club;
    content.innerHTML = `
      <div class="example-card" style="border:none; padding:1.25rem;">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(score)}">${score}</div>
        <div class="tenure-note">
          ${stop.type === "loan" ? "Loan" : "Scored"} over ${tenureDisplay(ex.tenure_days, stop.type)}${stillThere ? " (still there)" : " (before leaving)"}
        </div>
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `;
  } catch (e) {
    content.innerHTML = `<div class="error-box">${e.message}</div>`;
  }
}

/** Close the card modal. */
function closeCard() {
  state.openStop = null;
  cardModal.close();
}

document.getElementById("card-modal-close").addEventListener("click", closeCard);
document.getElementById("card-modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-modal-backdrop") closeCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCard();
});

/**
 * Render the chart/header/legend/table (and, solo-only, similar-career
 * suggestions) from state.careerA/careerB - called after either one
 * changes. Chart points are indexed within their own career's stops
 * array and tagged data-player="a"/"b" (buildTimelineSVG always tags
 * "a", even in solo mode, so one listener setup covers both chart
 * types). Table rows are indexed within whichever list was actually
 * rendered - the merged, playerKey-tagged list in comparison mode, or
 * careerA.stops directly in solo mode, where a row's absent playerKey
 * correctly falls through to careerA's own id below.
 */
function renderTimeline() {
  const { careerA, careerB } = state;
  document.getElementById("empty-state").hidden = true;
  document.getElementById("timeline-section").hidden = false;

  renderPlayerHeader(careerA, careerB);
  renderLegend(careerA, careerB);
  document.getElementById("timeline-chart-wrap").innerHTML = careerB
    ? buildOverlayTimelineSVG(careerA, careerB)
    : buildTimelineSVG(careerA.stops);

  const tableRows = careerB ? mergedStopsWithPlayer(careerA, careerB) : careerA.stops;
  renderStopsTable(tableRows, !!careerB);

  document.querySelectorAll(".timeline-point").forEach(el => {
    el.addEventListener("click", () => {
      const career = el.dataset.player === "b" ? careerB : careerA;
      showCard(career.stops[Number(el.dataset.index)], career.player_id);
    });
  });
  [...document.querySelectorAll("#stops-body tr")].forEach(row => {
    const open = () => {
      const s = tableRows[Number(row.dataset.index)];
      showCard(s, s.playerKey === "b" ? careerB.player_id : careerA.player_id);
    };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      open();
    });
  });

  // Suggestions only make sense before a comparison exists - clicking one
  // once careerB is already set would just silently swap it, surprising
  // rather than useful.
  renderSimilarCareers(careerB ? [] : careerA.similar_careers);
}

/** Fetch and render the primary player's career - starts (or restarts) a solo view, clearing any active comparison, since searching a brand-new primary player makes the old comparison partner unrelated. */
async function loadPlayerA(playerId) {
  const res = await fetch(`/api/players/${playerId}/career`);
  if (!res.ok) return;
  state.careerA = await res.json();
  state.careerB = null;
  document.getElementById("player-search").value = state.careerA.name;
  document.getElementById("player-b-search").value = "";
  document.getElementById("player-b-chip").innerHTML = "";
  document.getElementById("player-b-field").hidden = false;
  renderTimeline();
}

/** Fetch and render the comparison player's career, overlaying it onto the primary player's chart/table/legend. Ignored if it's the same player as careerA - comparing someone with themselves has nothing to show. */
async function loadPlayerB(playerId) {
  if (!state.careerA || playerId === state.careerA.player_id) return;
  const res = await fetch(`/api/players/${playerId}/career`);
  if (!res.ok) return;
  state.careerB = await res.json();
  document.getElementById("player-b-search").value = state.careerB.name;
  document.getElementById("player-b-chip").innerHTML =
    `<span class="selected-chip">${state.careerB.name} <button type="button" id="remove-compare-btn" aria-label="Remove comparison" style="background:none; border:none; color:inherit; cursor:pointer; padding:0 0 0 0.3rem; font:inherit;">&times;</button></span>`;
  document.getElementById("remove-compare-btn").addEventListener("click", removeComparison);
  renderTimeline();
}

/** Drop the active comparison and return to the primary player's own solo view. */
function removeComparison() {
  state.careerB = null;
  document.getElementById("player-b-search").value = "";
  document.getElementById("player-b-chip").innerHTML = "";
  renderTimeline();
}

setupAutocomplete({
  inputId: "player-search",
  listId: "player-list",
  endpoint: "/api/players/career-search",
  renderLabel: (p) => p.name,
  onSelect: (p) => loadPlayerA(p.player_id),
});

setupAutocomplete({
  inputId: "player-b-search",
  listId: "player-b-list",
  endpoint: "/api/players/career-search",
  renderLabel: (p) => p.name,
  onSelect: (p) => loadPlayerB(p.player_id),
});

// A currency change doesn't affect this page (no fees shown directly), but
// a currency-formatted breakdown description inside an open card does -
// refresh it in place, same pattern as browse.js/loans.js.
document.addEventListener("settingschange", () => {
  if (state.openStop) showCard(state.openStop.stop, state.openStop.playerId);
});

// Deep-link support: the homepage's historical example cards (see
// home.js) link here as /player.html?player_id=X so clicking one lands
// straight on that player's real career instead of making a visitor
// search for a name they were just shown a moment ago.
const initialPlayerId = new URLSearchParams(location.search).get("player_id");
if (initialPlayerId) loadPlayerA(Number(initialPlayerId));
