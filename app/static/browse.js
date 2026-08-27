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
      <tr>
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
