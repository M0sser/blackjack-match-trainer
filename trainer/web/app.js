// Shared pieces for every screen: the tabs, the "working out the best bets" banner,
// the error strip, and small helpers. Each screen script fills its own <section>.
(function () {
  "use strict";

  const SUITS = { S: "♠", H: "♥", D: "♦", C: "♣" };
  const TABS = ["practice", "review", "settings"];
  const tabCallbacks = {};
  const readyCallbacks = [];
  const statusCallbacks = [];
  let status = null;
  let pollTimer = null;
  let errorTimer = null;

  // ---- helpers ---------------------------------------------------------------

  // Build an element: el("button", {class: "btn", onclick: fn}, "Deal")
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
      else if (name === "class") node.className = value;
      else if (name === "html") node.innerHTML = value;
      else if (value === true) node.setAttribute(name, "");
      else node.setAttribute(name, value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  // 1150 -> "1,150"; 1007.5 -> "1,007.5"
  function fmtChips(amount) {
    return Number(amount).toLocaleString("en-US", { maximumFractionDigits: 1 });
  }

  // 0.4123 -> "41.2%". One decimal, because a choice counts as a miss from half a point.
  function fmtChance(chance) {
    return (Math.round(Number(chance) * 1000) / 10).toFixed(1) + "%";
  }

  // "10H" -> a card element; null -> a face-down card
  function cardEl(card) {
    if (!card) return el("span", { class: "card back", "aria-label": "face-down card" });
    const suit = card.slice(-1);
    const rank = card.slice(0, -1);
    const red = suit === "H" || suit === "D";
    return el("span", { class: "card" + (red ? " red" : "") }, rank, el("i", {}, SUITS[suit] || ""));
  }

  function cardHtml(card) {
    return cardEl(card).outerHTML;
  }

  function showError(message) {
    const strip = document.getElementById("error-strip");
    strip.textContent = message;
    strip.classList.add("visible");
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => strip.classList.remove("visible"), 4000);
  }

  // Call the trainer. Resolves with the answer; on a refusal shows the message and rejects.
  async function api(method, path, body, options) {
    let response;
    try {
      response = await fetch(path, {
        method,
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (err) {
      const message = "The trainer is not answering. Is its window still open?";
      if (!(options && options.quiet)) showError(message);
      throw new Error(message);
    }
    let answer = {};
    try { answer = await response.json(); } catch (err) { /* not JSON */ }
    if (!response.ok) {
      const message = answer.error || "Something went wrong inside the trainer.";
      if (!(options && options.quiet)) showError(message);
      const failure = new Error(message);
      failure.status = response.status;
      throw failure;
    }
    return answer;
  }

  // ---- tabs ------------------------------------------------------------------

  function currentTab() {
    const hash = location.hash.replace("#", "");
    return TABS.find((name) => hash.startsWith(name)) || "practice";
  }

  function showTab(name) {
    for (const tab of TABS) {
      document.getElementById("tab-" + tab).hidden = tab !== name;
      const button = document.querySelector('.tab[data-tab="' + tab + '"]');
      button.classList.toggle("active", tab === name);
      button.setAttribute("aria-selected", tab === name ? "true" : "false");
    }
    updateBanner();
    (tabCallbacks[name] || []).forEach((fn) => fn());
  }

  function onTab(name, fn) {
    (tabCallbacks[name] = tabCallbacks[name] || []).push(fn);
  }

  function shownTab() {
    return TABS.find((name) => !document.getElementById("tab-" + name).hidden) || "practice";
  }

  // ---- the brain's status ----------------------------------------------------

  function updateBanner() {
    const banner = document.getElementById("solving");
    const solving = !status || !status.brain.ready;
    banner.hidden = !solving || shownTab() === "settings";
    if (status && solving) {
      document.getElementById("solving-message").textContent = status.brain.message || "";
      document.getElementById("solving-fill").style.width = Math.round((status.brain.progress || 0) * 100) + "%";
    }
  }

  async function refreshStatus() {
    clearTimeout(pollTimer);
    const wasReady = status && status.brain.ready;
    try {
      status = await api("GET", "/api/status", undefined, { quiet: true });
    } catch (err) {
      pollTimer = setTimeout(refreshStatus, 2000);
      return status;
    }
    updateBanner();
    statusCallbacks.forEach((fn) => fn(status));
    if (status.brain.ready) {
      if (!wasReady) readyCallbacks.forEach((fn) => fn(status));
    } else {
      pollTimer = setTimeout(refreshStatus, 1000);
    }
    return status;
  }

  // fn(status) once the brain is ready, and again each time it becomes ready after a re-solve
  function onReady(fn) {
    readyCallbacks.push(fn);
    if (status && status.brain.ready) fn(status);
  }

  // fn(status) every time the status is fetched
  function onStatus(fn) {
    statusCallbacks.push(fn);
    if (status) fn(status);
  }

  window.Trainer = {
    api, showError, fmtChips, fmtChance, cardEl, cardHtml, el,
    onTab, showTab, shownTab, onReady, onStatus, refreshStatus,
    getStatus: () => status,
    BEST_NOTE: "Best means finishing ahead most often against the stand-in opponents in Settings.",
  };

  document.addEventListener("DOMContentLoaded", () => {
    for (const button of document.querySelectorAll(".tab")) {
      button.addEventListener("click", () => {
        history.replaceState(null, "", "#" + button.dataset.tab);
        showTab(button.dataset.tab);
      });
    }
    showTab(currentTab());
    refreshStatus();
  });
})();
