const state = { openStop: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
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
      ? `<circle class="timeline-point" data-index="${i}" cx="${cx}" cy="${cy}" r="6" fill="var(--panel)" stroke="${color}" stroke-width="3" />`
      : `<circle class="timeline-point" data-index="${i}" cx="${cx}" cy="${cy}" r="7" fill="${color}" stroke="var(--panel)" stroke-width="2" />`;
    const label = `${s.from_club} → ${s.to_club} (${s.transfer_date.slice(0, 7)}): ${s.score}`;
    return `<g style="cursor:pointer">${circle}<title>${label}</title></g>`;
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
    ${firstYear !== lastYear ? yearLabels : ""}
  </svg>`;
}

/** Render the player header (name, position, current club, span of years) above the chart. */
function renderPlayerHeader(career) {
  const years = `${career.stops[0].transfer_date.slice(0, 4)}–${career.stops[career.stops.length - 1].transfer_date.slice(0, 4)}`;
  const meta = [career.position, career.current_club ? `currently at ${career.current_club}` : null].filter(Boolean).join(" · ");
  document.getElementById("player-header").innerHTML = `
    <div class="name" style="font-size:1.3rem;">${career.name}</div>
    <div class="route">${meta ? meta + " · " : ""}${career.stops.length} scored move${career.stops.length === 1 ? "" : "s"} (${years})</div>
  `;
}

/** Render the plain-table fallback/detail list of every stop below the chart. */
function renderStopsTable(stops) {
  document.getElementById("stops-body").innerHTML = stops.map((s, i) => `
    <tr data-index="${i}" tabindex="0" role="button" aria-label="View details: ${s.from_club} to ${s.to_club}">
      <td>${s.transfer_date.slice(0, 7)}</td>
      <td>${s.from_club} &rarr; ${s.to_club}</td>
      <td>${s.type === "loan" ? "Loan" : "Permanent"}</td>
      <td>${s.age_at_transfer}</td>
      <td style="color:${scoreColor(s.score)}; font-weight:700">${s.score}</td>
    </tr>
  `).join("");
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
 */
async function showCard(stop) {
  state.openStop = stop;
  const content = document.getElementById("card-modal-content");
  content.innerHTML = "Loading...";
  cardModal.open();
  try {
    const endpoint = stop.type === "loan" ? "/api/loans/detail" : "/api/transfers/detail";
    const res = await fetch(`${endpoint}?player_id=${state.currentPlayerId}&transfer_date=${stop.transfer_date}`);
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

/** Fetch and render one player's full career timeline (chart + table), wiring click handlers on both. */
async function loadCareer(playerId) {
  state.currentPlayerId = playerId;
  const res = await fetch(`/api/players/${playerId}/career`);
  if (!res.ok) return;
  const career = await res.json();

  document.getElementById("empty-state").hidden = true;
  document.getElementById("timeline-section").hidden = false;

  renderPlayerHeader(career);
  document.getElementById("timeline-chart-wrap").innerHTML = buildTimelineSVG(career.stops);
  renderStopsTable(career.stops);

  const open = (i) => showCard(career.stops[i]);
  document.querySelectorAll(".timeline-point").forEach(el => {
    el.addEventListener("click", () => open(Number(el.dataset.index)));
  });
  [...document.querySelectorAll("#stops-body tr")].forEach(row => {
    row.addEventListener("click", () => open(Number(row.dataset.index)));
    row.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      open(Number(row.dataset.index));
    });
  });
}

setupAutocomplete({
  inputId: "player-search",
  listId: "player-list",
  endpoint: "/api/players/career-search",
  renderLabel: (p) => p.name,
  onSelect: (p) => loadCareer(p.player_id),
});

// A currency change doesn't affect this page (no fees shown directly), but
// a currency-formatted breakdown description inside an open card does -
// refresh it in place, same pattern as browse.js/loans.js.
document.addEventListener("settingschange", () => {
  if (state.openStop) showCard(state.openStop);
});
