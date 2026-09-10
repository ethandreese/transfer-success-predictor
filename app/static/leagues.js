const state = { rows: [], openLeague: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Green for a score improvement, red for a decline - score_change is signed around 0, not a 0-100 scale. */
function deltaColor(delta) {
  return delta >= 0 ? "var(--accent)" : "var(--accent-bad)";
}

/** "+2.4" / "-1.5" - always signed so the direction reads at a glance. */
function signed(n, suffix = "") {
  return `${n >= 0 ? "+" : ""}${n}${suffix}`;
}

/** Fetch the current sort's leaderboard and render the table body with per-row click handlers. */
async function loadTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = `<tr><td colspan="6">Loading...</td></tr>`;
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  const res = await fetch(`/api/leagues/trends?sort=${sort}&order=${order}`);
  const data = await res.json();
  state.rows = data.results;

  if (!data.results.length) {
    tbody.innerHTML = `<tr><td colspan="6">No leagues match.</td></tr>`;
    return;
  }
  tbody.innerHTML = data.results.map((r, i) => `
    <tr data-index="${i}" tabindex="0" role="button" aria-label="View trend details: ${r.league}">
      <td>${r.league}</td>
      <td>${r.transfers}</td>
      <td style="color:${scoreColor(r.avg_score)}; font-weight:700">${r.avg_score}</td>
      <td>${formatMoney(r.avg_fee)}</td>
      <td>${r.fee_growth_pct === null ? "—" : signed(r.fee_growth_pct, "%")}</td>
      <td style="color:${r.score_change === null ? "var(--muted)" : deltaColor(r.score_change)}; font-weight:${r.score_change === null ? 400 : 700}">${r.score_change === null ? "—" : signed(r.score_change)}</td>
    </tr>
  `).join("");
  [...tbody.querySelectorAll("tr")].forEach(row => {
    const open = () => showLeague(state.rows[Number(row.dataset.index)]);
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      open();
    });
  });
}

/**
 * Build the early-vs-recent trend chart as an inline SVG: both series
 * (avg fee, avg score) indexed to their own early-period average = 100,
 * so two differently-scaled quantities (euros vs. a 0-100 score) can
 * share one y-axis honestly - a dashed line at 100 marks "unchanged from
 * the early period" for both. Years are evenly spaced (already yearly
 * buckets, unlike Player Timelines' real-date positioning which had to
 * handle multi-year gaps between sparse events).
 */
function buildTrendSVG(byYear, earlyAvgFee, earlyAvgScore) {
  const W = 900, H = 240, padL = 40, padR = 16, padT = 16, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const years = byYear.map(d => d.year);
  const feeIdx = byYear.map(d => earlyAvgFee > 0 ? (d.avg_fee / earlyAvgFee) * 100 : 100);
  const scoreIdx = byYear.map(d => earlyAvgScore > 0 ? (d.avg_score / earlyAvgScore) * 100 : 100);
  const allIdx = [...feeIdx, ...scoreIdx, 100];
  const maxIdx = Math.max(...allIdx) * 1.12;
  const minIdx = Math.min(0, Math.min(...allIdx) * 0.9);

  const x = (i) => years.length === 1 ? padL + chartW / 2 : padL + (i / (years.length - 1)) * chartW;
  const y = (v) => padT + (1 - (v - minIdx) / (maxIdx - minIdx)) * chartH;

  const baseline = `
    <line x1="${padL}" y1="${y(100)}" x2="${W - padR}" y2="${y(100)}" stroke="var(--border)" stroke-dasharray="4 3" stroke-width="1" />
    <text x="${padL}" y="${y(100) - 5}" font-size="10" fill="var(--muted)">100 = ${years[0]} avg</text>
  `;

  function seriesPath(vals, color, label) {
    const pts = vals.map((v, i) => `${x(i)},${y(v)}`).join(" ");
    const dots = vals.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${color}"><title>${label} in ${years[i]}: index ${v.toFixed(0)}</title></circle>`).join("");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" />${dots}`;
  }

  const yearLabels = years.map((yr, i) => `<text x="${x(i)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">${yr}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="240" role="img" aria-label="Fee and score index trend chart">
    ${baseline}
    ${seriesPath(feeIdx, "var(--accent-mid)", "Avg fee index")}
    ${seriesPath(scoreIdx, "var(--accent)", "Avg score index")}
    ${yearLabels}
  </svg>`;
}

/** Plain-language summary of the early-vs-recent comparison, in the same "back the number with a real comparison" style as the rest of the site's explanations. */
function verdictSentence(r) {
  const feeDir = r.fee_growth_pct >= 0 ? "grew" : "fell";
  const scoreDir = r.score_change >= 0 ? "rose" : "dropped";
  return `Between ${r.early_years} and ${r.recent_years}, the average fee in ${r.league} ${feeDir} from
    ${formatMoney(r.early_avg_fee)} to ${formatMoney(r.recent_avg_fee)} (${signed(r.fee_growth_pct, "%")}),
    while the average success score ${scoreDir} from ${r.early_avg_score} to ${r.recent_avg_score}
    (${signed(r.score_change)} pts).`;
}

/** Open the modal and render one league's trend chart + verdict, entirely from data already returned by /api/leagues/trends - no second request needed. */
function showLeague(league) {
  state.openLeague = league;
  const content = document.getElementById("card-modal-content");
  const hasTrend = league.by_year.length > 0;
  content.innerHTML = `
    <div class="example-card" style="border:none; padding:1.25rem;">
      <div class="name">${league.league}</div>
      <div class="route">${league.transfers} scored transfers &middot; ${league.avg_score} avg score &middot; ${formatMoney(league.avg_fee)} avg fee</div>
      ${hasTrend ? `
        <div class="timeline-chart-wrap">${buildTrendSVG(league.by_year, league.early_avg_fee, league.early_avg_score)}</div>
        <div class="timeline-legend">
          <span><span class="legend-dot" style="background:var(--accent-mid); border:none;"></span> Avg fee (indexed)</span>
          <span><span class="legend-dot" style="background:var(--accent); border:none;"></span> Avg score (indexed)</span>
        </div>
        <p class="surprise-banner">${verdictSentence(league)}</p>
      ` : `<p class="surprise-banner">Not enough historical spread yet for a trend (needs at least 6 complete years of data).</p>`}
    </div>
  `;
  cardModal.open();
}

/** Close the league modal. */
function closeCard() {
  state.openLeague = null;
  cardModal.close();
}

document.getElementById("card-modal-close").addEventListener("click", closeCard);
document.getElementById("card-modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-modal-backdrop") closeCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCard();
});

document.getElementById("sort-select").addEventListener("change", loadTable);

document.addEventListener("settingschange", () => {
  loadTable();
  if (state.openLeague) showLeague(state.openLeague);
});

loadTable();
