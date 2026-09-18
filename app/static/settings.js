// Shared, cross-page settings: currency, theme, and browse/loans page size.
// Loaded before every page's own script (index.html/browse.html/loans.html/
// compare.html) so `formatMoney`/`convertMoneyInText`/settings are ready by
// the time page scripts run. Persisted to localStorage (per-browser only -
// there's no user account for this to sync across devices) and broadcast
// via a "settingschange" DOM event so each page can re-render without a
// full reload when a setting changes.

// Fixed, approximate conversion rates (checked August 2026) - not a live
// feed. Good enough for "roughly how big is this fee in dollars", not for
// anything financial.
// Cold-start indicator: Render's free tier spins the server down after 15
// min idle, so the first request after a quiet spell can take 30-60s
// instead of the usual sub-second response - without this, that shows up as
// a page silently stuck on "Loading..." with no sign anything is happening.
// Wraps window.fetch globally rather than hooking each page's own fetch
// calls, so every page gets it for free - including compare.html, which
// has no fetch at all until the user searches for a player.
(function () {
  let pendingSlow = 0;
  let banner = null;

  function showBanner() {
    if (banner) return;
    banner = document.createElement("div");
    banner.className = "wake-banner";
    banner.innerHTML =
      '<span class="wake-banner-spinner"></span>' +
      "<span>Server is waking up, can take up to a minute.</span>";
    document.body.appendChild(banner);
  }

  function hideBanner() {
    if (!banner) return;
    banner.remove();
    banner = null;
  }

  const originalFetch = window.fetch.bind(window);
  window.fetch = function (...args) {
    let countedSlow = false;
    // A normal warm request finishes in well under this, so a real request
    // never flashes the banner - only a genuine cold start does.
    const timer = setTimeout(() => {
      countedSlow = true;
      pendingSlow++;
      showBanner();
    }, 2500);

    const settle = () => {
      clearTimeout(timer);
      if (countedSlow && --pendingSlow <= 0) hideBanner();
    };

    const result = originalFetch(...args);
    result.then(settle, settle);
    return result;
  };
})();

const EXCHANGE_RATES = { EUR: 1, USD: 1.16, GBP: 0.86 };
const CURRENCY_SYMBOLS = { EUR: "€", USD: "$", GBP: "£" };
const DEFAULT_SETTINGS = { currency: "EUR", theme: "dark", pageSize: 25 };

/** Load saved settings from localStorage, filling in any missing keys with defaults (e.g. after adding a new setting). */
function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("tsp_settings") || "{}") };
  } catch (e) {
    return { ...DEFAULT_SETTINGS };
  }
}

const settings = loadSettings();

/** Persist the current settings object and notify every listener (other code on the page) that something changed. */
function saveSettings() {
  localStorage.setItem("tsp_settings", JSON.stringify(settings));
  applyTheme();
  document.dispatchEvent(new CustomEvent("settingschange", { detail: settings }));
}

/** Stamp the chosen theme onto <html> so style.css's [data-theme="light"] override block applies (or doesn't, for the default dark theme). */
function applyTheme() {
  document.documentElement.setAttribute("data-theme", settings.theme);
}

/**
 * Convert a raw euro amount (in whole euros, e.g. 50_000_000) to the
 * selected currency and format it like the backend's eur_m(): "€50m",
 * "$54.5m" (sub-1-unit amounts get one decimal place), "€2.05b" once the
 * converted amount reaches a billion (no individual transfer fee does, but
 * a club's aggregate spend/resale-profit total on /clubs.html can), or
 * "free" for 0/NaN. Mirrors app/main.py's eur_m() so the two stay visually
 * consistent.
 */
function formatMoney(eurValue) {
  if (eurValue === null || eurValue === undefined || Number.isNaN(eurValue) || eurValue === 0) return "free";
  const symbol = CURRENCY_SYMBOLS[settings.currency];
  const converted = eurValue * EXCHANGE_RATES[settings.currency];
  if (Math.abs(converted) >= 1_000_000_000) {
    return `${symbol}${(converted / 1_000_000_000).toFixed(2)}b`;
  }
  const millions = converted / 1_000_000;
  return millions < 1 ? `${symbol}${millions.toFixed(1)}m` : `${symbol}${millions.toFixed(0)}m`;
}

/**
 * Rewrite every "€X.Ym"/"€Xm"/"€X.Yb" token in backend-generated prose
 * (e.g. a breakdown description like "Bought for €50m, later resold for
 * €80m...") into the selected currency. The backend always formats
 * amounts in EUR (see eur_m() in app/main.py) since that's the dataset's
 * native currency, so this is the only way to make that prose
 * currency-aware without a backend round-trip - a no-op when the setting
 * is already EUR.
 */
function convertMoneyInText(text) {
  if (!text || settings.currency === "EUR") return text;
  return text.replace(/€([\d.]+)([mb])/g, (match, amount, unit) => {
    const eurValue = parseFloat(amount) * (unit === "b" ? 1_000_000_000 : 1_000_000);
    return formatMoney(eurValue);
  });
}

/**
 * Wire a modal backdrop for keyboard/screen-reader use: Tab/Shift+Tab cycle
 * only through the modal's own focusable elements (without this, the page
 * behind a "modal" dialog - nav links, table rows - is still reachable by
 * Tab even though it's visually covered), opening moves focus onto the
 * modal's close button (the one element guaranteed to exist before any
 * async content loads), and closing restores focus to whatever triggered
 * the modal instead of dropping it back to <body>. Call once per modal at
 * page load, then use the returned open()/close() in place of toggling
 * "open" on the backdrop directly.
 */
function makeModalAccessible(backdrop) {
  let opener = null;

  function focusableElements() {
    return [...backdrop.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
      .filter(el => el.offsetParent !== null && !el.disabled);
  }

  backdrop.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const focusable = focusableElements();
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  });

  return {
    // A no-op when already open, not just an idempotent re-add of the
    // "open" class - browse.js/loans.js call open() again to refresh an
    // already-open card's content on a settings change (currency, theme),
    // which would otherwise yank focus back to the close button and
    // overwrite `opener` with whatever was focused in the Settings modal
    // at that moment.
    open() {
      if (backdrop.classList.contains("open")) return;
      opener = document.activeElement;
      backdrop.classList.add("open");
      backdrop.querySelector(".modal-close")?.focus();
    },
    close() {
      backdrop.classList.remove("open");
      opener?.focus();
      opener = null;
    },
  };
}

/** Read the current URL's query string into a plain object - the read half of every list page's URL sync (see writeURLParams below), used once at load to restore a shared/bookmarked view's search/filter/sort/page. */
function readURLParams() {
  return Object.fromEntries(new URLSearchParams(location.search));
}

/**
 * Replace the current URL's query string with `params` (any key whose
 * value is undefined/null/"" is dropped, so an unfiltered view keeps a
 * clean URL rather than trailing empty params) - called after every
 * search/filter/sort/page change on Browse, Loans, Model vs Reality, Club
 * Report Cards, and League Trends, so the address bar always reflects the
 * current view and can be bookmarked or shared as-is. Uses replaceState,
 * not pushState - a live-search box or a paginated table firing this on
 * every keystroke/click would otherwise turn the back button into a
 * click-by-click undo history instead of just leaving the page.
 */
function writeURLParams(params) {
  const url = new URL(location.href);
  url.search = "";
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  });
  history.replaceState(null, "", url);
}

/** Build and insert the settings gear button + its modal into the page. Call once, after the DOM is ready. */
function injectSettingsUI() {
  const btn = document.createElement("button");
  btn.className = "settings-btn";
  btn.id = "settings-btn";
  btn.setAttribute("aria-label", "Settings");
  btn.textContent = "⚙";
  document.body.appendChild(btn);

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.id = "settings-modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-box" role="dialog" aria-modal="true" aria-labelledby="settings-modal-title">
      <button class="modal-close" id="settings-modal-close" aria-label="Close">&times;</button>
      <h2 id="settings-modal-title">Settings</h2>
      <div class="field">
        <label for="currency-select">Currency</label>
        <select id="currency-select">
          <option value="EUR">EUR (&euro;)</option>
          <option value="USD">USD ($)</option>
          <option value="GBP">GBP (&pound;)</option>
        </select>
        <p class="settings-note">Fees and market values shown across the site convert at a fixed, approximate rate, not a live feed.</p>
      </div>
      <div class="field">
        <label for="theme-select">Theme</label>
        <select id="theme-select">
          <option value="dark">Dark</option>
          <option value="light">Light</option>
        </select>
      </div>
      <div class="field">
        <label for="page-size-select">Rows per page (Browse / Loans)</label>
        <select id="page-size-select">
          <option value="25">25</option>
          <option value="50">50</option>
          <option value="100">100</option>
        </select>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  document.getElementById("currency-select").value = settings.currency;
  document.getElementById("theme-select").value = settings.theme;
  document.getElementById("page-size-select").value = String(settings.pageSize);

  const modal = makeModalAccessible(backdrop);
  btn.addEventListener("click", modal.open);
  document.getElementById("settings-modal-close").addEventListener("click", modal.close);
  backdrop.addEventListener("click", (e) => { if (e.target === backdrop) modal.close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.close(); });

  document.getElementById("currency-select").addEventListener("change", (e) => {
    settings.currency = e.target.value;
    saveSettings();
  });
  document.getElementById("theme-select").addEventListener("change", (e) => {
    settings.theme = e.target.value;
    saveSettings();
  });
  document.getElementById("page-size-select").addEventListener("change", (e) => {
    settings.pageSize = parseInt(e.target.value, 10);
    saveSettings();
  });
}

/**
 * Wire every grouped nav item (.nav-dropdown, e.g. "Predict"/"Browse"/
 * "Insights" in the topnav) to open on hover *and* on click - hover for a
 * mouse user (the expected way a grouped nav item behaves), click as the
 * fallback that also works for touch and for a keyboard user tabbing to
 * the toggle and pressing Enter/Space (a native <button> already turns
 * that into a "click"). Only one menu open at a time; leaving the
 * dropdown, clicking outside it, or pressing Escape closes whichever is
 * open.
 *
 * Closing on mouseleave is delayed (CLOSE_DELAY_MS), not immediate. The
 * menu sits directly against the toggle with no gap between them (see
 * .nav-dropdown-menu in style.css - an earlier version used a margin-top
 * there, which sat outside both elements' hit-test area and caused a real
 * dead zone; the visual breathing room now comes from padding-top instead,
 * which stays inside the menu's own hoverable box). The delay is just a
 * small courtesy buffer on top of that fix, for a jittery mouse or a
 * diagonal move that briefly overshoots the menu's edge - entering the
 * dropdown again (or a click) cancels the pending close either way.
 */
function initNavDropdowns() {
  const CLOSE_DELAY_MS = 150;
  const dropdowns = [...document.querySelectorAll(".nav-dropdown")].map(el => ({
    el, toggle: el.querySelector(".nav-dropdown-toggle"), menu: el.querySelector(".nav-dropdown-menu"),
  }));
  let closeTimer = null;

  function closeAll() {
    clearTimeout(closeTimer);
    dropdowns.forEach(({ toggle, menu }) => {
      menu.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  }

  function openOnly(target) {
    clearTimeout(closeTimer);
    dropdowns.forEach(d => {
      const isTarget = d === target;
      d.menu.classList.toggle("open", isTarget);
      d.toggle.setAttribute("aria-expanded", String(isTarget));
    });
  }

  dropdowns.forEach(d => {
    d.el.addEventListener("mouseenter", () => openOnly(d));
    d.el.addEventListener("mouseleave", () => {
      closeTimer = setTimeout(closeAll, CLOSE_DELAY_MS);
    });
    d.toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = d.menu.classList.contains("open");
      isOpen ? closeAll() : openOnly(d);
    });
  });

  document.addEventListener("click", closeAll);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAll(); });
}

applyTheme();
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => { injectSettingsUI(); initNavDropdowns(); });
} else {
  injectSettingsUI();
  initNavDropdowns();
}
