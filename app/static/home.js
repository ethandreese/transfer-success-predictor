/** Green/amber/red for a 0-100 score, shared by every score display on the page. */
function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
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

/** Fetch the curated showcase cards from /api/examples and render them into #examples. */
async function loadExamples() {
  const el = document.getElementById("examples");
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    if (!data.length) {
      el.textContent = "No examples available.";
      return;
    }
    const years = (days) => (days / 365.25).toFixed(1);
    el.innerHTML = data.map(ex => `
      <div class="example-card">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club} (${ex.transfer_date.slice(0, 7)})</div>
        <div class="score" style="color:${scoreColor(ex.success_score)}">${ex.success_score}</div>
        <div class="tenure-note">
          Scored over ${years(ex.tenure_days)} years at the club${ex.still_at_club ? " (still there)" : " (before leaving)"}
        </div>
        <div class="breakdown">${renderBreakdown(ex.breakdown)}</div>
      </div>
    `).join("");
  } catch (e) {
    el.textContent = "Failed to load examples.";
  }
}

/** Fetch the hypothetical-move predictions from /api/examples/predictions and render them into #hypothetical-examples - the same 4 players as loadExamples() above, each scored for a different destination club instead of their real historical move. A lighter card than .example-card's historical one (no breakdown/tenure-note, since there's no real outcome to describe) - just the route, the predicted score, and the hypothetical fee it was scored at. */
async function loadHypotheticalExamples() {
  const el = document.getElementById("hypothetical-examples");
  try {
    const res = await fetch("/api/examples/predictions");
    const data = await res.json();
    if (!data.length) {
      el.textContent = "No predictions available.";
      return;
    }
    el.innerHTML = data.map(ex => `
      <div class="example-card">
        <div class="name">${ex.name}</div>
        <div class="route">${ex.from_club} &rarr; ${ex.to_club}</div>
        <div class="score" style="color:${scoreColor(ex.success_score)}">${ex.success_score}</div>
        <div class="tenure-note">Predicted for a hypothetical ${formatMoney(ex.transfer_fee)} move</div>
      </div>
    `).join("");
  } catch (e) {
    el.textContent = "Failed to load predictions.";
  }
}

// A currency change doesn't change the underlying data, just how the
// breakdown's money-mentioning descriptions (and the hypothetical fee
// note) display - reload both grids so they redisplay in the new currency.
document.addEventListener("settingschange", () => {
  loadExamples();
  loadHypotheticalExamples();
});

loadExamples();
loadHypotheticalExamples();
