const PAGE_SIZE = 25;
const state = { offset: 0, total: 0 };

function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
}

function currentParams() {
  const [sort, order] = document.getElementById("sort-select").value.split(":");
  const params = new URLSearchParams({
    sort, order,
    limit: PAGE_SIZE,
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

function feeDisplay(fee) {
  if (fee === null || fee === undefined || fee === 0) return "free";
  return `€${(fee / 1_000_000).toFixed(1)}m`;
}

function tenureDisplay(days, stillAtClub) {
  const years = (days / 365.25).toFixed(1);
  return `${years}y${stillAtClub ? " (current)" : ""}`;
}

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
        <td>${feeDisplay(r.transfer_fee)}</td>
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

  const page = Math.floor(state.offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
  document.getElementById("page-info").textContent = `Page ${page} of ${pageCount} (${state.total} transfers)`;
  document.getElementById("prev-page").disabled = state.offset === 0;
  document.getElementById("next-page").disabled = state.offset + PAGE_SIZE >= state.total;
}

function resetAndLoad() {
  state.offset = 0;
  loadTable();
}

function renderBreakdown(breakdown) {
  return breakdown.map(b => `
    <div class="breakdown-row">
      <span class="tooltip-wrap breakdown-label">
        ${b.label}
        <span class="tooltip-box">${b.description}</span>
      </span>
      <div class="breakdown-bar-track">
        <div class="breakdown-bar-fill" style="width:${b.value}%; background:${scoreColor(b.value)}"></div>
      </div>
      <span class="breakdown-value">${b.value}</span>
    </div>
  `).join("");
}

async function showCard(playerId, transferDate) {
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
      <div class="example-card" style="border:none; padding:0;">
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

function closeCard() {
  document.getElementById("card-modal-backdrop").classList.remove("open");
}

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
document.getElementById("league-select").addEventListener("change", resetAndLoad);
document.getElementById("sort-select").addEventListener("change", resetAndLoad);
document.getElementById("prev-page").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - PAGE_SIZE);
  loadTable();
});
document.getElementById("next-page").addEventListener("click", () => {
  state.offset += PAGE_SIZE;
  loadTable();
});

loadFilters();
loadTable();
