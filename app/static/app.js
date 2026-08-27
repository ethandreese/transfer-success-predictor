const state = {
  player: null,
  playerClub: null,
  club: null,
};

function scoreColor(score) {
  if (score >= 66) return "var(--accent)";
  if (score >= 40) return "var(--accent-mid)";
  return "var(--accent-bad)";
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

function setupAutocomplete({ inputId, listId, endpoint, onSelect, renderLabel }) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  let debounceTimer = null;

  input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const q = input.value.trim();
    if (q.length < 2) {
      list.classList.remove("open");
      return;
    }
    debounceTimer = setTimeout(async () => {
      const res = await fetch(`${endpoint}?q=${encodeURIComponent(q)}`);
      const items = await res.json();
      if (!items.length) {
        list.classList.remove("open");
        return;
      }
      list.innerHTML = items.map((item, i) => `<div data-idx="${i}">${renderLabel(item)}</div>`).join("");
      list.classList.add("open");
      [...list.children].forEach((child, i) => {
        child.addEventListener("click", () => {
          onSelect(items[i]);
          list.classList.remove("open");
          input.value = renderLabel(items[i]);
        });
      });
    }, 200);
  });

  document.addEventListener("click", (e) => {
    if (e.target !== input) list.classList.remove("open");
  });
}

function updatePredictButton() {
  document.getElementById("predict-btn").disabled = !(state.player && state.club);
}

setupAutocomplete({
  inputId: "player-search",
  listId: "player-list",
  endpoint: "/api/players/search",
  renderLabel: (p) => `${p.name} (${p.position}, ${p.current_club_name})`,
  onSelect: async (p) => {
    state.player = p;
    document.getElementById("player-chip").innerHTML =
      `<span class="selected-chip">${p.name} &middot; age ${Number(p.age_now).toFixed(1)} &middot; ${p.current_club_name}</span>`;
    document.getElementById("age-override").value = Number(p.age_now).toFixed(1);
    if (p.current_club_id) {
      const res = await fetch(`/api/clubs/${p.current_club_id}`);
      state.playerClub = res.ok ? await res.json() : null;
    }
    updatePredictButton();
  },
});

setupAutocomplete({
  inputId: "club-search",
  listId: "club-list",
  endpoint: "/api/clubs/search",
  renderLabel: (c) => `${c.name}`,
  onSelect: (c) => {
    state.club = c;
    document.getElementById("club-chip").innerHTML =
      `<span class="selected-chip">${c.name}</span>`;
    updatePredictButton();
  },
});

document.getElementById("predict-btn").addEventListener("click", async () => {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = "";
  if (!state.player || !state.club) return;

  const fromClub = state.playerClub || {
    domestic_competition_id: state.player.current_club_domestic_competition_id,
    club_value_proxy: 0,
  };

  const payload = {
    age_at_transfer: parseFloat(document.getElementById("age-override").value) || state.player.age_now,
    height_in_cm: state.player.height_in_cm,
    position: state.player.position,
    foot: state.player.foot || "unknown",
    pre_apps: state.player.recent_apps,
    pre_minutes: state.player.recent_minutes,
    pre_goals_p90: state.player.recent_goals_p90,
    pre_ga_p90: state.player.recent_ga_p90,
    pre_mins_per_app: state.player.recent_mins_per_app,
    transfer_fee: (parseFloat(document.getElementById("fee").value) || 0) * 1_000_000,
    value_before: state.player.market_value_in_eur,
    from_domestic_competition_id: fromClub.domestic_competition_id || "unknown",
    to_domestic_competition_id: state.club.domestic_competition_id || "unknown",
    from_total_market_value: fromClub.club_value_proxy || 1,
    to_total_market_value: state.club.club_value_proxy || 1,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }
    const data = await res.json();
    document.getElementById("result").classList.add("open");
    const scoreEl = document.getElementById("score-value");
    scoreEl.textContent = data.success_score;
    scoreEl.style.color = scoreColor(data.success_score);
    const [lo, hi] = data.score_range;
    document.getElementById("score-range-note").textContent =
      `Likely range: ${lo}–${hi}, based on the most similar historical transfers`;
    document.getElementById("mae-value").textContent = data.model_test_mae;
    document.getElementById("r2-value").textContent = data.model_test_r2;
    document.getElementById("comparables-list").innerHTML = data.comparable_transfers.map(c => `
      <div class="comp-row">
        <span>${c.name} (${c.from_club} &rarr; ${c.to_club}, ${c.transfer_date.slice(0, 7)})</span>
        <span style="color:${scoreColor(c.success_score)}">${c.success_score}</span>
      </div>
    `).join("");
    document.getElementById("explanation-list").innerHTML = data.explanation.map(e => {
      const positive = e.contribution >= 0;
      const width = Math.min(Math.abs(e.contribution) * 4, 100);
      return `
        <div class="explain-row">
          <span class="tooltip-wrap explain-label">
            ${e.label}
            <span class="tooltip-box">${e.detail}</span>
          </span>
          <div class="explain-bar-track">
            <div class="explain-bar-fill ${positive ? "pos" : "neg"}" style="width:${width}%"></div>
          </div>
          <span class="explain-value">${positive ? "+" : ""}${e.contribution}</span>
        </div>
      `;
    }).join("");
  } catch (e) {
    errorBox.textContent = e.message;
  }
});

loadExamples();
