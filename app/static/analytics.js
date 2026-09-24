const state = { rows: [], data: null };
const cardModal = makeModalAccessible(document.getElementById("card-modal-backdrop"));

/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

/** Turn /api/analytics' columnar scatter ({name: [...], position: [...], ...}) into one row object per transfer - easier for each chart to filter/read than parallel arrays. */
function zipScatter(scatter) {
  const rows = [];
  for (let i = 0; i < scatter.name.length; i++) {
    rows.push({
      player_id: scatter.player_id[i],
      transfer_date: scatter.transfer_date[i],
      name: scatter.name[i],
      position: scatter.position[i],
      transfer_fee: scatter.transfer_fee[i],
      market_value_in_eur: scatter.market_value_in_eur[i],
      age_at_transfer: scatter.age_at_transfer[i],
      success_score: scatter.success_score[i],
      height_in_cm: scatter.height_in_cm[i],
    });
  }
  return rows;
}

/** Position display order shared by the Height vs. Score chart's trend lines and legend - goalkeepers first (fewest transfers, tallest on average), then outfield positions back to front. */
const POSITION_ORDER = ["Goalkeeper", "Defender", "Midfield", "Attack"];
const POSITION_COLORS = {
  Goalkeeper: "var(--pos-goalkeeper)",
  Defender: "var(--pos-defender)",
  Midfield: "var(--pos-midfield)",
  Attack: "var(--pos-attack)",
};

function linearScale(d0, d1, r0, r1) {
  return (v) => d1 === d0 ? (r0 + r1) / 2 : r0 + (v - d0) / (d1 - d0) * (r1 - r0);
}

function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  return (v) => l1 === l0 ? (r0 + r1) / 2 : r0 + (Math.log10(v) - l0) / (l1 - l0) * (r1 - r0);
}

/** Horizontal gridlines + left-edge labels at 0/25/50/75/100 - the y-axis every score-based chart on this page shares. */
function scoreGridlines(y, padL, padR, W) {
  return [0, 25, 50, 75, 100].map(score => `
    <line x1="${padL}" y1="${y(score).toFixed(1)}" x2="${W - padR}" y2="${y(score).toFixed(1)}" stroke="var(--border)" stroke-width="1" />
    <text x="${padL - 6}" y="${(y(score) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">${score}</text>
  `).join("");
}

/**
 * One clickable scatter point: a small visible dot plus a larger
 * (invisible) circle laid on top purely to enlarge the tap/click target -
 * at r=2.3 the visible dot alone is too small to hit reliably, especially
 * by touch. Wrapped in a <g class="scatter-pt"> carrying the transfer's
 * player_id/transfer_date as data attributes, which wireScatterClicks()
 * reads via event delegation to open that transfer's full card.
 */
function scatterPoint(cx, cy, color, opacity, playerId, transferDate, tooltip) {
  return `
    <g class="scatter-pt" data-player-id="${playerId}" data-transfer-date="${transferDate}" style="cursor:pointer">
      <circle cx="${cx}" cy="${cy}" r="6" fill="transparent" />
      <circle cx="${cx}" cy="${cy}" r="2.3" fill="${color}" opacity="${opacity}" />
      <title>${tooltip}</title>
    </g>
  `;
}

/**
 * The binned-average trend line overlaid on a scatter chart: a dashed
 * line in a color distinct from both the scoreColor'd dots and the
 * theme's plain text/border colors (--trend-line by default), with a
 * marker at each underlying bucket. `color` is overridable so the Height
 * vs. Score chart can draw four of these at once, one per position, each
 * in its own --pos-* color (see POSITION_COLORS) - reusing --trend-line
 * for all four would make them indistinguishable from each other and,
 * worse, other charts wouldn't get where "trend-line blue" stops meaning
 * "the sitewide reference line" and starts meaning "this one position."
 * Each marker carries both a `<title>` (a free hover tooltip on desktop,
 * once the browser's own hover delay elapses) and a `data-tooltip`
 * attribute - the real interaction, read by the document-level click
 * listener near the end of this file to show showChartTooltip() on
 * click/tap, since `<title>` alone never fires on a touch device at all
 * (there's no hover state to trigger it). Each marker also gets the same
 * small-visible-dot/larger-invisible-hit-circle treatment as
 * scatterPoint() above - at r=3.5 alone the visible dot is a hard target
 * to land a click on exactly, especially with only 6-12 of them spread
 * across the full chart width.
 */
function trendLinePath(trend, x, y, labelFor, color = "var(--trend-line)") {
  const pts = trend.map(t => `${x(t.x).toFixed(1)},${y(t.avg_score).toFixed(1)}`).join(" ");
  const markers = trend.map(t => {
    const cx = x(t.x).toFixed(1), cy = y(t.avg_score).toFixed(1);
    const label = labelFor(t);
    return `
      <g style="cursor:pointer" data-tooltip="${label}">
        <circle cx="${cx}" cy="${cy}" r="10" fill="transparent" />
        <circle cx="${cx}" cy="${cy}" r="3.5" fill="${color}" stroke="var(--panel)" stroke-width="1" />
        <title>${label}</title>
      </g>
    `;
  }).join("");
  return `
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-dasharray="7 4" stroke-linecap="round" opacity="0.9" />
    ${markers}
  `;
}

/**
 * Fee (log x-axis, since fees span €1k-€200m+) vs. success score. Dots are
 * colored by outcome (scoreColor) and drawn at low opacity since ~4,400
 * points overlap heavily at the low-fee end; the trend line comes from
 * the backend's fee_trend (average score per fee decile), not recomputed
 * here. Free/undisclosed-fee transfers are excluded - they can't sit on
 * a log axis at 0.
 */
function buildFeeScoreChart(rows, trend) {
  const points = rows.filter(r => r.transfer_fee > 0);
  const W = 900, H = 320, padL = 40, padR = 16, padT = 14, padB = 30;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const fees = points.map(p => p.transfer_fee);
  const minFee = Math.min(...fees), maxFee = Math.max(...fees);
  const x = logScale(minFee, maxFee, padL, W - padR);
  const y = (score) => padT + (100 - score) / 100 * chartH;

  const dots = points.map(p => scatterPoint(
    x(p.transfer_fee).toFixed(1), y(p.success_score).toFixed(1), scoreColor(p.success_score), 0.45,
    p.player_id, p.transfer_date, `${p.name}: ${formatMoney(p.transfer_fee)}, score ${p.success_score} (click for details)`,
  )).join("");

  const trendLine = trendLinePath(
    trend.filter(t => t.x >= minFee && t.x <= maxFee), x, y,
    (t) => `Around ${formatMoney(t.x)}: avg score ${t.avg_score} (${t.n.toLocaleString()} transfers)`,
  );

  const ticks = [1e5, 1e6, 1e7, 1e8].filter(t => t >= minFee && t <= maxFee);
  const xLabels = ticks.map(t => `<text x="${x(t).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${formatMoney(t)}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Transfer fee vs success score scatter plot">
    ${scoreGridlines(y, padL, padR, W)}
    ${dots}
    ${trendLine}
    ${xLabels}
  </svg>`;
}

/** Age at transfer (linear x-axis) vs. success score, same dot/trend-line treatment as the fee chart above. Every transfer has an age, so nothing is excluded here. */
function buildAgeScoreChart(rows, trend) {
  const W = 900, H = 320, padL = 40, padR = 16, padT = 14, padB = 30;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const ages = rows.map(r => r.age_at_transfer);
  const minAge = Math.floor(Math.min(...ages)), maxAge = Math.ceil(Math.max(...ages));
  const x = linearScale(minAge, maxAge, padL, W - padR);
  const y = (score) => padT + (100 - score) / 100 * chartH;

  const dots = rows.map(r => scatterPoint(
    x(r.age_at_transfer).toFixed(1), y(r.success_score).toFixed(1), scoreColor(r.success_score), 0.4,
    r.player_id, r.transfer_date, `${r.name}: age ${r.age_at_transfer}, score ${r.success_score} (click for details)`,
  )).join("");

  const trendLine = trendLinePath(
    trend, x, y,
    (t) => `Around age ${t.x.toFixed(1)}: avg score ${t.avg_score} (${t.n.toLocaleString()} transfers)`,
  );

  const xLabels = [];
  for (let a = Math.ceil(minAge / 5) * 5; a <= maxAge; a += 5) {
    xLabels.push(`<text x="${x(a).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${a}</text>`);
  }

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Age at transfer vs success score scatter plot">
    ${scoreGridlines(y, padL, padR, W)}
    ${dots}
    ${trendLine}
    ${xLabels.join("")}
  </svg>`;
}

/**
 * Height (linear x-axis - 160-203cm is too narrow a range to need a log
 * scale) vs. success score, with an overlaid trend line per position
 * (see POSITION_ORDER/POSITION_COLORS and height_trend_by_position on
 * the backend) instead of the fee/age charts' single sitewide line - the
 * whole point of this chart is whether the height-score relationship
 * actually differs by role (a real premium at centre-back/goalkeeper
 * that doesn't hold for attackers, say), which one blended average
 * would hide entirely. `rows`/`trendByPosition` are whatever the "Split
 * by position" filter left in (see renderHeightChart) - all four
 * positions by default, or just one once the reader isolates it; this
 * function itself doesn't know or care which.
 */
function buildHeightScoreChart(rows, trendByPosition) {
  const points = rows.filter(r => r.height_in_cm != null);
  const W = 900, H = 320, padL = 40, padR = 16, padT = 14, padB = 30;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const heights = points.map(p => p.height_in_cm);
  const minH = Math.min(...heights), maxH = Math.max(...heights);
  const x = linearScale(minH, maxH, padL, W - padR);
  const y = (score) => padT + (100 - score) / 100 * chartH;

  const dots = points.map(p => scatterPoint(
    x(p.height_in_cm).toFixed(1), y(p.success_score).toFixed(1), scoreColor(p.success_score), 0.35,
    p.player_id, p.transfer_date, `${p.name}: ${p.height_in_cm}cm, score ${p.success_score} (click for details)`,
  )).join("");

  const trendLines = POSITION_ORDER.map(position => {
    const trend = (trendByPosition[position] || []).filter(t => t.x >= minH && t.x <= maxH);
    if (!trend.length) return "";
    return trendLinePath(
      trend, x, y,
      (t) => `${position} around ${t.x.toFixed(0)}cm: avg score ${t.avg_score} (${t.n.toLocaleString()} transfers)`,
      POSITION_COLORS[position],
    );
  }).join("");

  const ticks = [];
  for (let h = Math.ceil(minH / 10) * 10; h <= maxH; h += 10) ticks.push(h);
  const xLabels = ticks.map(t => `<text x="${x(t).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${t}cm</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Height vs success score scatter plot, by position">
    ${scoreGridlines(y, padL, padR, W)}
    ${dots}
    ${trendLines}
    ${xLabels}
  </svg>`;
}

/**
 * Fee paid vs. market value at the time, both log-scaled on the same
 * domain so the dashed diagonal is a true y=x reference line - above it,
 * a club paid more than market value; below, less. Dots keep the
 * scoreColor treatment so the chart also answers "did bargains or
 * overpays tend to work out better".
 */
function buildFeeValueChart(rows) {
  const points = rows.filter(r => r.transfer_fee > 0 && r.market_value_in_eur > 0);
  const W = 900, H = 340, padL = 50, padR = 16, padT = 14, padB = 30;
  const allVals = points.flatMap(p => [p.transfer_fee, p.market_value_in_eur]);
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const x = logScale(minV, maxV, padL, W - padR);
  const y = logScale(minV, maxV, H - padB, padT);

  const diagonal = `<line x1="${x(minV).toFixed(1)}" y1="${y(minV).toFixed(1)}" x2="${x(maxV).toFixed(1)}" y2="${y(maxV).toFixed(1)}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7" />`;

  const dots = points.map(p => scatterPoint(
    x(p.market_value_in_eur).toFixed(1), y(p.transfer_fee).toFixed(1), scoreColor(p.success_score), 0.45,
    p.player_id, p.transfer_date,
    `${p.name}: ${formatMoney(p.market_value_in_eur)} value, ${formatMoney(p.transfer_fee)} fee, score ${p.success_score} (click for details)`,
  )).join("");

  const ticks = [1e5, 1e6, 1e7, 1e8].filter(t => t >= minV && t <= maxV);
  const xLabels = ticks.map(t => `<text x="${x(t).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${formatMoney(t)}</text>`).join("");
  const yLabels = ticks.map(t => `<text x="${padL - 6}" y="${(y(t) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">${formatMoney(t)}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Transfer fee vs market value scatter plot">
    ${diagonal}
    ${dots}
    ${xLabels}
    ${yLabels}
    <text x="${padL}" y="12" font-size="10" fill="var(--muted)">↑ fee paid</text>
    <text x="${W - padR}" y="${H - padB - 6}" text-anchor="end" font-size="10" fill="var(--muted)">market value →</text>
  </svg>`;
}

/**
 * Transfer count and average fee by year, both indexed to the first
 * year = 100 so two differently-scaled series (a headcount, a euro
 * amount) can share one y-axis honestly - same trick as League Trends'
 * own chart (see leagues.js's buildTrendSVG). The current in-progress
 * year is already excluded server-side (by_year). Each yearly point is
 * click/tap-tooltippable via data-tooltip (see showChartTooltip() near
 * the end of this file) the same way the fee/age charts' trend-line
 * markers are - a <title> alone never fires on a touch device.
 */
function buildVolumeChart(byYear) {
  const W = 900, H = 260, padL = 40, padR = 16, padT = 16, padB = 26;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const years = byYear.map(d => d.year);
  const baseCount = byYear[0].transfers, baseFee = byYear[0].avg_fee;
  const countIdx = byYear.map(d => (d.transfers / baseCount) * 100);
  const feeIdx = byYear.map(d => baseFee > 0 ? (d.avg_fee / baseFee) * 100 : 100);
  const allIdx = [...countIdx, ...feeIdx, 100];
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
    const dots = idxVals.map((v, i) => {
      const cx = x(i).toFixed(1), cy = y(v).toFixed(1);
      const text = `${label} in ${years[i]}: ${formatRaw(rawVals[i])}`;
      return `
        <g style="cursor:pointer" data-tooltip="${text}">
          <circle cx="${cx}" cy="${cy}" r="9" fill="transparent" />
          <circle cx="${cx}" cy="${cy}" r="4" fill="${color}" />
          <title>${text}</title>
        </g>
      `;
    }).join("");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" />${dots}`;
  }

  const yearLabels = years.map((yr, i) => `<text x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">${yr}</text>`).join("");

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Transfer market volume and average fee over time">
    ${baseline}
    ${seriesPath(countIdx, byYear.map(d => d.transfers), "var(--accent)", "Transfers", (v) => `${v} transfers`)}
    ${seriesPath(feeIdx, byYear.map(d => d.avg_fee), "var(--accent-mid)", "Avg fee", formatMoney)}
    ${yearLabels}
  </svg>`;
}

/** Render one transfer's score-component breakdown (label + bar + hover tooltip) as HTML, from the `breakdown` array the API returns. Descriptions run through convertMoneyInText since the backend always formats euro amounts in its prose. Copied from surprises.js/browse.js - same card, same backend endpoint. */
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

/** Open the modal and fetch+render one transfer's full card via /api/transfers/detail - same card Browse/Loans/Model vs Reality use, reached here by clicking/tapping a scatter point instead of a table row. */
async function showCard(playerId, transferDate) {
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

/**
 * Delegate clicks on a chart's wrapper div to whichever .scatter-pt group
 * was hit, rather than attaching a listener per point - a scatter chart
 * here draws thousands of points, and delegation means this is wired once
 * per chart container, not re-wired on every re-render (currency/theme
 * changes just replace the wrapper's innerHTML; the listener stays on the
 * wrapper itself). Only called for the three per-transfer scatter charts
 * (fee/age/market-value) - the market-over-time chart has no .scatter-pt
 * points (nothing to open a transfer card for, it's a yearly aggregate),
 * though its own points are still tooltippable via the separate
 * data-tooltip click listener below.
 */
function wireScatterClicks(containerId) {
  document.getElementById(containerId).addEventListener("click", (e) => {
    const pt = e.target.closest(".scatter-pt");
    if (pt) showCard(pt.dataset.playerId, pt.dataset.transferDate);
  });
}

const chartTooltip = document.getElementById("chart-tooltip");

/**
 * Show the shared chart-tooltip box near a click/tap position, clamped so
 * it never runs off the right or bottom edge of the viewport. This is the
 * real interaction behind every trend-line marker's and market-over-time
 * point's data-tooltip - a native <title> hover tooltip is also present
 * on those same elements as a free bonus for a desktop mouse user willing
 * to wait out the browser's hover delay, but it never fires at all on a
 * touch device (there's no hover state to trigger it), which is why this
 * click-triggered version is the one actually relied on.
 */
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

document.getElementById("card-modal-close").addEventListener("click", () => cardModal.close());
document.getElementById("card-modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-modal-backdrop") cardModal.close();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { cardModal.close(); hideChartTooltip(); }
});
wireScatterClicks("fee-score-chart");
wireScatterClicks("age-score-chart");
wireScatterClicks("height-score-chart");
wireScatterClicks("fee-value-chart");

// Any click on a data-tooltip element (a trend-line marker or a
// market-over-time point) shows its tooltip; any other click - elsewhere
// on the page, including a second tap on the same mark - hides it. One
// listener for every chart on the page, same event-delegation reasoning
// as wireScatterClicks above.
document.addEventListener("click", (e) => {
  const marked = e.target.closest("[data-tooltip]");
  if (marked) showChartTooltip(marked.dataset.tooltip, e.clientX, e.clientY);
  else if (!e.target.closest("#chart-tooltip")) hideChartTooltip();
});

/**
 * Render the Height vs. Score chart, description, and legend from the
 * current "Split by position" filter - all four positions overlaid by
 * default, or just one isolated once the reader picks it from the
 * <select>, which re-plots that position alone rather than leaving the
 * other three lines cluttering a comparison the reader has already
 * decided they don't need. Called from renderAll() (initial load,
 * settingschange) and directly from the filter's own change listener -
 * switching the filter doesn't need every other chart on the page to
 * redraw too.
 */
function renderHeightChart() {
  const { data, rows } = state;
  const filter = document.getElementById("height-position-filter").value;
  const filteredRows = filter ? rows.filter(r => r.position === filter) : rows;
  const trendByPosition = filter ? { [filter]: data.height_trend_by_position[filter] } : data.height_trend_by_position;

  const heightPoints = filteredRows.filter(r => r.height_in_cm != null);
  document.getElementById("height-score-chart").innerHTML = buildHeightScoreChart(filteredRows, trendByPosition);
  document.getElementById("height-score-desc").textContent = filter
    ? `${heightPoints.length.toLocaleString()} ${filter.toLowerCase()} transfers with a recorded height. Click/tap a point on the trend line for that height band's average score and sample size.`
    : `${heightPoints.length.toLocaleString()} transfers with a recorded height. Each position gets its own trend line; click/tap a point on one for that height band's average score and sample size, and compare the four lines' slopes for whether height matters more at some positions than others.`;

  [...document.getElementById("height-score-legend").children].forEach(el => {
    el.hidden = filter !== "" && el.dataset.position !== filter;
  });
}
document.getElementById("height-position-filter").addEventListener("change", renderHeightChart);

/** Render every chart, description, and legend from the already-fetched /api/analytics response - called once on load and again on a settingschange (currency affects several charts' tick/tooltip labels, and there's no harm re-drawing the rest). */
function renderAll() {
  const { data } = state;
  const rows = state.rows;

  // A currency change re-renders every chart's tick/tooltip labels below,
  // but an already-open chart-tooltip was set from a click before the
  // re-render and won't update itself - hide it rather than leave stale
  // text (e.g. a euro amount) on screen after the switch.
  hideChartTooltip();

  const feePoints = rows.filter(r => r.transfer_fee > 0);
  document.getElementById("fee-score-chart").innerHTML = buildFeeScoreChart(rows, data.fee_trend);
  document.getElementById("fee-score-desc").textContent =
    `${feePoints.length.toLocaleString()} transfers with a disclosed fee (free/undisclosed fees excluded, since they can't sit on a log axis). ` +
    `The trend line's average score climbs from ${data.fee_trend[0].avg_score} in the cheapest tenth of fees to ${data.fee_trend[data.fee_trend.length - 1].avg_score} in the priciest tenth; click/tap any point on it for that bucket's exact average and sample size.`;

  document.getElementById("age-score-chart").innerHTML = buildAgeScoreChart(rows, data.age_trend);
  document.getElementById("age-score-desc").textContent =
    `All ${rows.length.toLocaleString()} scored transfers. The trend line's average score falls from ${data.age_trend[0].avg_score} for the youngest transfers to ${data.age_trend[data.age_trend.length - 1].avg_score} for the oldest, so buying young tends to pay off, at least on average. Click/tap any point on it for that bucket's exact numbers.`;

  renderHeightChart();

  const valuePoints = rows.filter(r => r.transfer_fee > 0 && r.market_value_in_eur > 0);
  document.getElementById("fee-value-chart").innerHTML = buildFeeValueChart(rows);
  document.getElementById("fee-value-desc").textContent =
    `${valuePoints.length.toLocaleString()} transfers with both a disclosed fee and a market-value snapshot. Above the dashed line, a club paid more than market value; below it, less. Dot color is still the transfer's outcome score.`;

  document.getElementById("volume-chart").innerHTML = buildVolumeChart(data.by_year);
  document.getElementById("volume-legend").innerHTML = `
    <span><span class="legend-dot" style="background:var(--accent); border:none;"></span> Transfer count (indexed)</span>
    <span><span class="legend-dot" style="background:var(--accent-mid); border:none;"></span> Avg fee (indexed)</span>
  `;
  const years = data.by_year.map(d => d.year);
  document.getElementById("volume-desc").textContent =
    `${years[0]}-${years[years.length - 1]} (the current in-progress year is excluded; see League Trends for why). Both series are indexed to ${years[0]} = 100 so a headcount and a euro amount can share one axis. Click/tap any point for that year's exact number.`;

  document.getElementById("feature-importance-desc").textContent =
    "The trained model's own learned weight for each numeric input (standardized, so every one of these is directly comparable to every other), not an analysis of this page's own data, but literally what the deployed model keys off when it predicts a score. Categorical features (position, foot, origin/destination league) aren't shown, since each expands into many individual category weights, not one number a ranking like this could use.";
  renderFeatureImportance(data.feature_importance);
}

/**
 * "What actually predicts success" - the trained model's own learned
 * feature weights (see /api/analytics' feature_importance), ranked by
 * magnitude. Reuses the exact .explain-row/.explain-bar-fill markup
 * Predict/Compare already use for per-feature contributions - same
 * visual language (green raises the score, red lowers it), just applied
 * to the model's fixed, sitewide coefficients instead of one specific
 * prediction's swap-and-measure contributions. Unlike Predict/Compare's
 * bars (which grow rightward from a shared left edge, comparing factors
 * to each other), these grow outward from a centered zero - the natural
 * reading for a signed coefficient, where left/right is itself part of
 * the meaning, not just magnitude - via .explain-bar-fill.centered's CSS.
 * Bar length is relative to the single largest |coefficient| in the list,
 * not an absolute scale - these are standardized-feature coefficients,
 * useful for ranking against each other but not meaningful as a raw
 * percentage of anything.
 */
function renderFeatureImportance(importance) {
  const maxAbs = Math.max(...importance.map(f => Math.abs(f.coefficient)));
  document.getElementById("feature-importance-list").innerHTML = importance.map(f => {
    const positive = f.coefficient >= 0;
    const halfWidth = Math.abs(f.coefficient) / maxAbs * 50;
    const side = positive ? `left:50%` : `right:50%`;
    return `
      <div class="explain-row">
        <span class="explain-label">${f.label}</span>
        <div class="explain-bar-track">
          <div class="explain-bar-fill centered ${positive ? "pos" : "neg"}" style="${side}; width:${halfWidth}%"></div>
        </div>
        <span class="explain-value">${positive ? "+" : ""}${f.coefficient}</span>
      </div>
    `;
  }).join("");
}

async function load() {
  const res = await fetch("/api/analytics");
  state.data = await res.json();
  state.rows = zipScatter(state.data.scatter);
  renderAll();
}

document.addEventListener("settingschange", () => {
  if (state.data) renderAll();
});

load();
