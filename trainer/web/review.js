// The review screen: put in a finished match round by round, let the trainer check that the
// chips add up, then analyze the rounds at once. .
(function () {
  "use strict";

  const T = window.Trainer;
  const el = T.el;
  const ROUNDS = 5;
  const STORE = "match-trainer-review";
  const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
  const KEYS = { a: "A", t: "10", j: "J", q: "Q", k: "K", "0": "10", "1": "A" };
  const SUITS = [["S", "♠ Spades"], ["H", "♥ Hearts"], ["D", "♦ Diamonds"], ["C", "♣ Clubs"]];
  const DID = { stand: "Stood", hit: "Hit", double: "Doubled", split: "Split",
                insurance: "Took insurance", no_insurance: "No insurance" };
  const DO = { stand: "Stand", hit: "Hit", double: "Double", split: "Split",
               insurance: "Insurance", no_insurance: "No insurance" };

  // ---- state ------------------------------------------------------------------

  function blankRound() {
    return { main: "", pp: "", fct: "", overrides: { main: false, pp: false, fct: false },
             player: [null, null], suits: [null, null], upcard: null,
             insurance: false, actions: [], dealer: [] };
  }

  function blankForm() {
    return { rounds: Array.from({ length: ROUNDS }, blankRound), current: 0, opponentFinal: "", won: "unsure" };
  }

  function load() {
    try {
      const form = JSON.parse(localStorage.getItem(STORE));
      if (form && Array.isArray(form.rounds) && form.rounds.length === ROUNDS) {
        // Bets in older saved reviews were entered by hand. Keep those amounts.
        form.rounds.forEach((r) => {
          if (!r.overrides) r.overrides = Object.fromEntries(["main", "pp", "fct"].map((name) =>
            [name, r[name] !== ""]));
        });
        return form;
      }
    } catch (err) { /* nothing saved, or storage is blocked */ }
    return blankForm();
  }

  function save() {
    try { localStorage.setItem(STORE, JSON.stringify(state.form)); } catch (err) { /* storage is blocked */ }
  }

  const state = {
    form: load(),
    slot: null,          // the card slot the picker fills: {kind: player|upcard|action|dealer, i, j}
    pickerOpen: false,
    recommendations: [],
    checks: [],          // the latest /api/review/check answer, one entry per round
    report: null,        // the latest /api/review/analyze answer
    analyzing: false,
  };
  const view = {};       // the containers that get re-rendered

  function round() { return state.form.rounds[state.form.current]; }

  function rules() {
    const status = T.getStatus();
    return (status && status.settings.rules) ||
      { min_bet: 5, bet_step: 5, insurance_offered: true, double_after_split: true, split_by_value: true };
  }

  // ---- reading a round ----------------------------------------------------------

  function value(rank) { return rank === "A" ? 1 : ["10", "J", "Q", "K"].includes(rank) ? 10 : Number(rank); }

  function total(cards) {
    const hard = cards.reduce((sum, rank) => sum + value(rank), 0);
    const soft = cards.includes("A") && hard + 10 <= 21;
    return { total: soft ? hard + 10 : hard, soft };
  }

  function stepOneDone(r) { return Number(r.main) > 0 && r.player[0] && r.player[1] && r.upcard; }
  function needsSuits(r) { return Number(r.pp) > 0 && r.player[0] && r.player[0] === r.player[1]; }
  function insuranceAsked(r) { return r.upcard === "A" && rules().insurance_offered; }

  // Follows the player's hands the way the trainer's engine does, so the form can label split
  // hands and know when the player's part is over. The server's check stays the judge.
  function follow(r) {
    const out = { hands: [], active: 0, over: false, waiting: false, splitAces: false, handOf: [] };
    if (!stepOneDone(r)) return out;
    const hands = [{ cards: [r.player[0], r.player[1]], done: false, fromSplit: false }];
    out.hands = hands;
    let secondCard = null;
    const down = r.dealer[0];
    if (total(r.player).total === 21 || (down && total([r.upcard, down]).total === 21)) out.over = true;

    const advance = () => {
      for (;;) {
        const hand = hands[out.active];
        if (!hand.done && total(hand.cards).total >= 21) hand.done = true;
        if (!hand.done) return;
        if (out.active + 1 === hands.length) { out.over = true; return; }
        out.active += 1;
        hands[out.active].cards.push(secondCard);
        if (out.splitAces) hands[out.active].done = true;
      }
    };

    r.actions.forEach((a, i) => {
      out.handOf[i] = hands.length > 1 ? out.active + 1 : 0;
      if (out.waiting || out.over) return;
      const hand = hands[out.active];
      if (a.action === "stand") {
        hand.done = true;
      } else if (a.action === "split") {
        if (!a.cards[0] || !a.cards[1]) { out.waiting = true; return; }
        const [first, second] = hand.cards;
        out.splitAces = first === "A";
        secondCard = a.cards[1];
        hands.splice(0, 1, { cards: [first, a.cards[0]], done: out.splitAces, fromSplit: true },
                     { cards: [second], done: false, fromSplit: true });
      } else {
        if (!a.card) { out.waiting = true; return; }
        hand.cards.push(a.card);
        if (a.action === "double") hand.done = true;
      }
      advance();
    });
    return out;
  }

  function canAct(r, f, action) {
    if (!f.hands.length || f.over || f.waiting) return false;
    const hand = f.hands[f.active];
    if (action === "double") {
      return hand.cards.length === 2 && !(hand.fromSplit && (f.splitAces || !rules().double_after_split));
    }
    if (action === "split") {
      const [a, b] = hand.cards;
      const pair = rules().split_by_value ? value(a) === value(b) : a === b;
      return f.hands.length === 1 && hand.cards.length === 2 && pair;
    }
    return true;
  }

  // The round as the server reads it, or null while step 1 is not filled in.
  function payload(r) {
    if (!stepOneDone(r)) return null;
    const actions = [];
    for (const a of r.actions) {
      if (a.action === "split" ? !(a.cards[0] && a.cards[1]) : a.action !== "stand" && !a.card) break;
      actions.push(a.action === "split" ? { action: "split", cards: [...a.cards] }
        : a.action === "stand" ? { action: "stand" } : { action: a.action, card: a.card });
    }
    const out = { main: Number(r.main), pp: Number(r.pp) || 0, fct: Number(r.fct) || 0,
                  player: [...r.player], upcard: r.upcard, insurance: insuranceAsked(r) && r.insurance,
                  actions, dealer: [...r.dealer] };
    if (needsSuits(r) && r.suits[0] && r.suits[1]) out.suits = [...r.suits];
    return out;
  }

  // Index of the round after which the chips ran out, or -1.
  function bustOut() {
    const { min_bet: minBet, bet_step: step } = rules();
    return state.checks.findIndex((c) =>
      c && c.complete && Math.floor(c.chips_after / step + 1e-9) * step < minBet);
  }

  function neededRounds() {
    const ended = bustOut();
    return ended >= 0 ? ended + 1 : ROUNDS;
  }

  function readyToAnalyze() {
    const needed = neededRounds();
    for (let i = 0; i < needed; i++) {
      const check = state.checks[i];
      if (!check || !check.ok || !check.complete) return false;
    }
    return true;
  }

  // ---- talking to the server -----------------------------------------------------

  let checkTimer = null;
  let checkNumber = 0;
  let adviceNumber = 0;
  let playAdviceNumber = 0;

  async function loadPlayAdvice() {
    const number = ++playAdviceNumber;
    const current = round(), data = payload(current), f = follow(current);
    const previous = state.form.rounds.slice(0, state.form.current).map(payload);
    if (!data || previous.some((r, i) => !r || !state.checks[i] || !state.checks[i].ok || !state.checks[i].complete)) {
      view.playAdvice.textContent = "Enter the bet and cards, and complete earlier rounds to see a suggested play.";
      return;
    }
    if (f.over || f.waiting) {
      view.playAdvice.textContent = f.waiting ? "Enter the drawn card to see the next suggested play." : "No further playing decision.";
      return;
    }
    view.playAdvice.textContent = "Calculating suggested play…";
    try {
      const result = await T.api("POST", "/api/review/best-play", { rounds: previous, current: data }, { quiet: true });
      if (number !== playAdviceNumber) return;
      if (!result.best) { view.playAdvice.textContent = "No playing decision available."; return; }
      view.playAdvice.replaceChildren(
        el("strong", {}, "Suggested play: " + (DO[result.best] || result.best)),
        el("p", { class: "note" }, "Estimated chance of finishing the match ahead:"),
        ...Object.entries(result.chances).map(([action, probability]) =>
          el("div", {}, (DO[action] || action) + ": " + T.fmtChance(probability))));
    } catch (err) {
      if (number === playAdviceNumber) view.playAdvice.textContent = "Suggested play unavailable. Check the bet and cards entered.";
    }
  }

  function betKey(current) {
    return JSON.stringify(state.form.rounds.slice(0, current).map(payload));
  }

  function currentRecommendation() {
    const advice = state.recommendations[state.form.current];
    return advice && advice.key === betKey(state.form.current) ? advice.bet : null;
  }

  function syncBetFields() {
    for (const name of ["main", "pp", "fct"]) {
      const input = view.bets[name];
      input.value = round()[name];
      input.readOnly = !round().overrides[name];
      input.classList.toggle("rv-default-bet", input.readOnly);
      view.overrides[name].checked = round().overrides[name];
    }
  }

  function showBetAdvice(bet, message) {
    if (bet) {
      view.betAdvice.replaceChildren(
        el("strong", {}, "Suggested main bet: " + T.fmtChips(bet.best)),
        el("p", { class: "note" }, T.fmtChance(bet.best_chance) + " chance of finishing ahead with the suggested bets."));
    } else {
      view.betAdvice.textContent = message;
    }
    for (const [name, key] of [["pp", "best_pp"], ["fct", "best_fct"]]) {
      const amount = bet && bet[key];
      const badge = view.sideAdvice[name];
      badge.className = "rv-side-status" + (bet ? amount > 0 ? " recommended" : " skip" : "");
      badge.textContent = bet ? amount > 0 ? "✓ Recommended · " + T.fmtChips(amount) + " chips"
        : "Skip · Not recommended" : "Waiting for recommendation";
    }
  }

  function applyRecommendation(bet) {
    const r = round();
    let updated = false;
    for (const [name, key] of [["main", "best"], ["pp", "best_pp"], ["fct", "best_fct"]]) {
      if (!r.overrides[name] && r[name] !== bet[key]) {
        r[name] = bet[key];
        updated = true;
      }
    }
    showBetAdvice(bet);
    syncBetFields();
    if (updated) changed();
  }

  async function loadBetAdvice() {
    const number = ++adviceNumber;
    const current = state.form.current;
    const previous = state.form.rounds.slice(0, current).map(payload);
    if (previous.some((r, i) => !r || !state.checks[i] || !state.checks[i].complete || !state.checks[i].ok)) {
      showBetAdvice(null, "Complete the earlier rounds to see the suggested bet.");
      return;
    }
    const cached = currentRecommendation();
    if (cached) { applyRecommendation(cached); return; }
    const key = betKey(current);
    showBetAdvice(null, "Calculating suggested bet…");
    try {
      const bet = await T.api("POST", "/api/review/best-bet", { rounds: previous }, { quiet: true });
      if (number !== adviceNumber) return;
      state.recommendations[current] = { key, bet };
      applyRecommendation(bet);
    } catch (err) {
      if (number === adviceNumber) showBetAdvice(null,
        "Suggested bet unavailable until the trainer is ready and earlier rounds are complete. You can still enter a different amount.");
    }
  }

  function scheduleCheck() {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(runCheck, 300);
  }

  async function runCheck() {
    clearTimeout(checkTimer);
    const number = ++checkNumber;
    try {
      const answer = await T.api("POST", "/api/review/check", { rounds: state.form.rounds.map(payload) }, { quiet: true });
      if (number !== checkNumber) return;
      state.checks = answer.rounds;
    } catch (err) {
      if (number !== checkNumber) return;
      state.checks = [];
    }
    showChecks();
    loadBetAdvice();
    loadPlayAdvice();
  }

  async function analyze() {
    if (!readyToAnalyze() || state.analyzing) return;
    const form = state.form;
    const body = {
      rounds: form.rounds.slice(0, neededRounds()).map(payload),
      opponent_final: form.opponentFinal === "" ? null : Number(form.opponentFinal),
      won: form.won === "won" ? true : form.won === "lost" ? false : null,
    };
    state.analyzing = true;
    showChecks();
    try {
      state.report = await T.api("POST", "/api/review/analyze", body);
      renderReport();
      renderRoundAdvice();
      if (window.innerWidth < 760) view.report.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      /* the error strip already says why */
    } finally {
      state.analyzing = false;
      showChecks();
    }
  }

  // ---- what the user does (the demo hook calls these too) ---------------------------

  // After anything in the form changes: keep it, check it, and drop a report that no longer fits.
  function changed() {
    ++adviceNumber;
    ++playAdviceNumber;
    ++checkNumber;
    // A later automatic bet must be recalculated if an earlier round changes.
    for (let i = state.form.current + 1; i < ROUNDS; i++) {
      state.recommendations[i] = null;
      for (const name of ["main", "pp", "fct"]) {
        if (!state.form.rounds[i].overrides[name]) state.form.rounds[i][name] = "";
      }
    }
    state.checks = state.checks.slice(0, state.form.current);
    save();
    scheduleCheck();
    if (state.report) { state.report = null; renderReport(); renderRoundAdvice(); }
    refresh();
  }

  function selectRound(i) {
    if (i > bustOut() && bustOut() >= 0) return;
    state.form.current = i;
    state.slot = null;
    state.pickerOpen = false;
    save();
    renderRound();
    loadBetAdvice();
    loadPlayAdvice();
  }

  function selectSlot(slot) {
    state.pickerOpen = !sameSlot(state.slot, slot) || !state.pickerOpen;
    state.slot = slot;
    refresh();
    if (state.pickerOpen) (view.picker.querySelector('[aria-pressed="true"]') || view.picker.querySelector(".rv-pick")).focus();
  }

  function setBet(name, amount) {
    round().overrides[name] = true;
    round()[name] = amount === "" ? "" : Number(amount);
    const input = view.bets && view.bets[name];
    if (input && input.value !== String(round()[name])) input.value = round()[name];
    syncBetFields();
    changed();
  }

  function toggleBetOverride(name, enabled) {
    round().overrides[name] = enabled;
    const bet = currentRecommendation();
    if (!enabled) round()[name] = bet ? bet[{ main: "best", pp: "best_pp", fct: "best_fct" }[name]] : "";
    syncBetFields();
    changed();
    if (enabled) { view.bets[name].focus(); view.bets[name].select(); }
  }

  function setSuit(i, suit) { round().suits[i] = suit || null; changed(); }
  function setInsurance(yes) { round().insurance = yes; changed(); }

  function firstEmptyStartSlot(r) {
    if (!r.player[0]) return { kind: "player", i: 0 };
    if (!r.player[1]) return { kind: "player", i: 1 };
    if (!r.upcard) return { kind: "upcard" };
    return null;
  }

  // The rank picker: fill the selected slot, then move to the next one.
  function pickRank(rank) {
    const r = round();
    let slot = state.slot || firstEmptyStartSlot(r) || { kind: "dealer", i: r.dealer.length };
    let next = null;
    if (slot.kind === "player") {
      r.player[slot.i] = rank;
      next = firstEmptyStartSlot(r);
    } else if (slot.kind === "upcard") {
      r.upcard = rank;
      if (rank !== "A") r.insurance = false;
      next = firstEmptyStartSlot(r);
    } else if (slot.kind === "action") {
      const a = r.actions[slot.i];
      if (!a) return;
      if (a.action === "split") {
        a.cards[slot.j] = rank;
        if (slot.j === 0 && !a.cards[1]) next = { kind: "action", i: slot.i, j: 1 };
      } else {
        a.card = rank;
      }
    } else {
      r.dealer[Math.min(slot.i, r.dealer.length)] = rank;
      next = { kind: "dealer", i: Math.min(slot.i, r.dealer.length - 1) + 1 };
    }
    if (!next && slot.kind !== "upcard" && follow(r).over) next = { kind: "dealer", i: r.dealer.length };
    state.slot = next;
    state.pickerOpen = false;
    changed();
    focusSlot(next || slot);
  }

  function addAction(action) {
    const r = round();
    if (!canAct(r, follow(r), action)) return;
    const i = r.actions.length;
    if (action === "split") r.actions.push({ action, cards: [null, null] });
    else if (action === "stand") r.actions.push({ action });
    else r.actions.push({ action, card: null });
    state.slot = action === "stand" ? null : { kind: "action", i, j: 0 };
    if (action === "stand" && follow(r).over) state.slot = { kind: "dealer", i: r.dealer.length };
    state.pickerOpen = action !== "stand";
    changed();
    if (state.pickerOpen) view.picker.querySelector(".rv-pick").focus();
  }

  function undoAction() {
    round().actions.pop();
    state.slot = null;
    state.pickerOpen = false;
    changed();
  }

  function undoDealer() {
    round().dealer.pop();
    state.slot = { kind: "dealer", i: round().dealer.length };
    state.pickerOpen = false;
    changed();
  }

  function setOpponent(amount) {
    state.form.opponentFinal = amount === "" ? "" : Number(amount);
    if (view.opponent.value !== String(amount)) view.opponent.value = amount;
    save();
  }

  function setWon(choice) {
    state.form.won = choice;
    view.won.value = choice;
    save();
  }

  function newReview() {
    ++adviceNumber;
    ++playAdviceNumber;
    state.form = blankForm();
    state.slot = null;
    state.pickerOpen = false;
    state.recommendations = [];
    state.report = null;
    state.checks = [];
    save();
    renderAll();
    scheduleCheck();
  }

  // ---- drawing ---------------------------------------------------------------------

  function sameSlot(a, b) {
    return Boolean(a && b && a.kind === b.kind && (a.i || 0) === (b.i || 0) && (a.j || 0) === (b.j || 0));
  }

  function slotKey(slot) { return [slot.kind, slot.i || 0, slot.j || 0].join("-"); }

  function slotButton(slot) {
    return slot && document.querySelector('[data-card-slot="' + slotKey(slot) + '"]');
  }

  function focusSlot(slot) {
    const button = slotButton(slot);
    if (button) button.focus({ preventScroll: true });
  }

  function closePicker(restoreFocus) {
    state.pickerOpen = false;
    view.picker.hidden = true;
    for (const button of document.querySelectorAll('.rv-slot[aria-expanded="true"]')) {
      button.setAttribute("aria-expanded", "false");
    }
    if (restoreFocus) focusSlot(state.slot);
  }

  function positionPicker() {
    if (!state.pickerOpen || view.picker.hidden) return;
    const anchor = slotButton(state.slot);
    if (!anchor) { closePicker(); return; }
    const rect = anchor.getBoundingClientRect();
    const popup = view.picker.getBoundingClientRect();
    const gap = 8;
    view.picker.style.left = Math.max(gap, Math.min(rect.left, window.innerWidth - popup.width - gap)) + "px";
    const below = rect.bottom + gap;
    const top = below + popup.height <= window.innerHeight - gap ? below : rect.top - popup.height - gap;
    view.picker.style.top = Math.max(gap, Math.min(top, window.innerHeight - popup.height - gap)) + "px";
  }

  function renderPicker() {
    const anchor = slotButton(state.slot);
    view.picker.hidden = !state.pickerOpen || !anchor;
    if (view.picker.hidden) return;
    const label = anchor.getAttribute("aria-label").split(":")[0];
    const rank = anchor.dataset.rank;
    view.picker.setAttribute("aria-label", "Choose a card for " + label.toLowerCase());
    view.picker.replaceChildren(
      el("div", { class: "rv-picker-head" }, el("strong", {}, label),
        el("button", { type: "button", class: "rv-picker-close", "aria-label": "Close card dropdown",
          onclick: () => closePicker(true) }, "×")),
      el("div", { class: "rv-picker" }, RANKS.map((choice) =>
        el("button", { type: "button", class: "rv-pick", "aria-label": "Choose " + choice,
          "aria-pressed": choice === rank ? "true" : "false", onclick: () => pickRank(choice) },
          el("span", { class: "rv-corner", "aria-hidden": "true" }, choice),
          el("span", { class: "rv-rank" }, choice),
          el("span", { class: "rv-corner bottom", "aria-hidden": "true" }, choice)))),
      el("p", { class: "note" }, "Choose a rank or type A, 2–9, T, J, Q, K. Esc closes."));
    positionPicker();
  }

  function slotEl(rank, slot, label) {
    const on = sameSlot(state.slot, slot);
    return el("button", {
      type: "button",
      class: "rv-slot" + (rank ? " filled" : "") + (on ? " on" : ""),
      "aria-label": label + (rank ? ": " + rank : ": empty"),
      "aria-haspopup": "dialog", "aria-controls": "rv-card-dropdown",
      "aria-expanded": on && state.pickerOpen ? "true" : "false",
      "data-card-slot": slotKey(slot), "data-rank": rank || "",
      onfocus: () => { if (!state.pickerOpen) state.slot = slot; },
      onclick: () => selectSlot(slot),
    }, el("span", {}, rank || "+"), el("small", { "aria-hidden": "true" }, "▾"));
  }

  function line(who, ...children) {
    return el("div", { class: "rv-line" }, el("span", { class: "rv-who" }, who), ...children);
  }

  function renderTabs() {
    const ended = bustOut();
    view.tabs.replaceChildren(...state.form.rounds.map((r, i) => {
      const check = state.checks[i];
      const off = ended >= 0 && i > ended;
      const bad = check && !check.ok && stepOneDone(r) && !off;
      const done = check && check.ok && check.complete;
      const mark = bad ? " ⚠" : done ? " ✓" : "";
      return el("button", {
        type: "button",
        class: "rv-tab" + (i === state.form.current ? " on" : "") + (bad ? " bad" : "") + (done ? " done" : ""),
        disabled: off,
        title: off ? "The chips ran out before this round." : null,
        onclick: () => selectRound(i),
      }, "Round " + (i + 1) + mark,
      el("small", {}, done ? T.fmtChips(check.chips_after) : off ? "not played" : "—"));
    }));
  }

  function betField(name, label) {
    const input = el("input", {
      type: "number", min: "0", step: String(rules().bet_step || 5), inputmode: "numeric",
      id: "rv-bet-" + name, readonly: !round().overrides[name], placeholder: "Waiting…",
      value: round()[name] === "" ? "" : String(round()[name]),
      oninput: (event) => setBet(name, event.target.value),
    });
    const override = el("input", { type: "checkbox", checked: round().overrides[name],
      "aria-label": "Change " + label.toLowerCase() + " amount",
      onchange: (event) => toggleBetOverride(name, event.target.checked) });
    view.bets[name] = input;
    view.overrides[name] = override;
    return el("div", { class: "rv-bet-field" }, el("label", { class: "field", for: input.id }, label, input),
      el("label", { class: "rv-override" }, override, "Change amount"));
  }

  // Step 1's cards: player cards, suit pickers for a Perfect Pairs pair, and the up card.
  function renderStart() {
    const r = round();
    const parts = [
      line("Your cards", slotEl(r.player[0], { kind: "player", i: 0 }, "Your first card"),
        slotEl(r.player[1], { kind: "player", i: 1 }, "Your second card")),
    ];
    if (needsSuits(r)) {
      const pickers = [0, 1].map((i) => el("select", {
        "aria-label": "Suit of your " + (i ? "second" : "first") + " card",
        onchange: (event) => setSuit(i, event.target.value),
      }, el("option", { value: "" }, "Suit…"),
      SUITS.map(([letter, name]) => el("option", { value: letter, selected: r.suits[i] === letter }, name))));
      parts.push(line("Suits of your pair", ...pickers));
    }
    parts.push(line("Dealer showed", slotEl(r.upcard, { kind: "upcard" }, "Dealer's up card")));
    view.start.replaceChildren(...parts);
  }

  function actionLabel(a, hand) {
    return (hand ? "Hand " + hand + ": " : "") + DID[a.action];
  }

  // Step 2: insurance, what you did, and the dealer's cards.
  function renderEnd() {
    const r = round();
    if (!stepOneDone(r)) {
      view.end.replaceChildren(el("p", { class: "note" }, "Fill in step 1 first: your bet, your two cards and the dealer's up card."));
      return;
    }
    const f = follow(r);
    const parts = [];
    if (insuranceAsked(r)) {
      parts.push(line("Insurance",
        el("button", { type: "button", class: "rv-move" + (r.insurance ? " on" : ""), onclick: () => setInsurance(true) }, "Yes"),
        el("button", { type: "button", class: "rv-move" + (!r.insurance ? " on" : ""), onclick: () => setInsurance(false) }, "No")));
    }
    parts.push(line("What you did", ...["hit", "stand", "double", "split"].map((action) =>
      el("button", { type: "button", class: "rv-move", disabled: !canAct(r, f, action), onclick: () => addAction(action) },
        DO[action]))));

    const list = r.actions.map((a, i) => {
      const cards = a.action === "split"
        ? [slotEl(a.cards[0], { kind: "action", i, j: 0 }, "Second card of hand 1"),
           slotEl(a.cards[1], { kind: "action", i, j: 1 }, "Second card of hand 2")]
        : a.action === "stand" ? [] : [slotEl(a.card, { kind: "action", i, j: 0 }, "Card for " + DO[a.action])];
      return el("li", {}, el("span", {}, actionLabel(a, f.handOf[i]) + (a.action === "split" ? ", then the hands got" : "")), ...cards);
    });
    if (list.length) {
      parts.push(el("ol", { class: "rv-actions" }, list));
      parts.push(el("div", { class: "rv-line" },
        el("span", { class: "rv-who" }, f.hands.map((h, i) =>
          (f.hands.length > 1 ? "Hand " + (i + 1) + " " : "You have ") + total(h.cards).total).join(" · ")),
        el("button", { type: "button", class: "btn rv-small", onclick: undoAction }, "Undo last move")));
    }

    const dealer = [r.upcard ? el("span", { class: "rv-slot filled fixed", title: "Up card" }, r.upcard) : null];
    r.dealer.forEach((rank, i) => dealer.push(slotEl(rank, { kind: "dealer", i }, i ? "Dealer's card " + (i + 2) : "Dealer's down card")));
    dealer.push(slotEl(null, { kind: "dealer", i: r.dealer.length }, r.dealer.length ? "Next dealer card" : "Dealer's down card"));
    if (r.dealer.length) dealer.push(el("button", { type: "button", class: "btn rv-small", onclick: undoDealer }, "Undo"));
    parts.push(line("Dealer's cards", ...dealer));
    parts.push(el("p", { class: "note" }, "The down card first, then each card the dealer drew."));
    view.end.replaceChildren(...parts);
  }

  // The line under the round: its result and chips, what is still missing, or what is wrong.
  function renderOutcome() {
    const i = state.form.current;
    const r = round();
    const check = state.checks[i];
    const before = state.checks[i - 1];
    let text = "";
    let kind = "";
    if (!stepOneDone(r)) {
      text = "Not filled in yet.";
    } else if (!check) {
      text = "Checking…";
    } else if (!check.ok) {
      text = check.error;
      kind = "miss";
    } else if (check.complete) {
      const net = check.net;
      text = (net > 0 ? "You won " + T.fmtChips(net) : net < 0 ? "You lost " + T.fmtChips(-net) : "You broke even") +
        ". Chips " + T.fmtChips(check.chips_after) + ".";
      kind = "done";
    } else if (before && before.chips_after === null) {
      text = "Finish round " + i + " first to see the chips.";
    } else {
      const f = follow(r);
      const missing = f.waiting ? "pick the card for your last move" : !f.over ? "add what you did"
        : "add the dealer's cards";
      text = "Chips before this round: " + T.fmtChips(check.chips_before) + ". Still to do: " + missing + ".";
    }
    view.outcome.className = "rv-outcome" + (kind ? " " + kind : "");
    view.outcome.textContent = text;
  }

  function renderAnalyze() {
    const ready = readyToAnalyze();
    view.analyze.disabled = !ready || state.analyzing;
    view.analyze.textContent = state.analyzing ? "Analyzing…" : "Analyze match";
    const ended = bustOut();
    view.analyzeNote.textContent = ready
      ? (ended >= 0 && ended < ROUNDS - 1 ? "The chips ran out after round " + (ended + 1) + ", so the match ends there. " : "") +
        "Every round adds up."
      : "Analyze opens once every round of the match is in and adds up.";
  }

  function showChecks() {
    renderTabs();
    renderOutcome();
    renderAnalyze();
  }

  function refresh() {
    renderStart();
    renderEnd();
    renderPicker();
    showChecks();
  }

  function renderRound() {
    view.bets = {};
    view.overrides = {};
    view.sideAdvice = {};
    view.fields.replaceChildren(betField("main", "Your bet"),
      el("p", { class: "note" }, "The suggested amount is filled in automatically. Change it only if you bet differently."));
    view.sideFields.replaceChildren(...[["pp", "Perfect Pairs"], ["fct", "4 Card Total"]].map(([name, label]) => {
      view.sideAdvice[name] = el("p", { class: "rv-side-status", "aria-live": "polite" });
      return el("div", { class: "rv-side-bet" }, el("h3", {}, label), view.sideAdvice[name], betField(name, label + " bet"));
    }));
    showBetAdvice(currentRecommendation(), "Calculating suggested bet…");
    syncBetFields();
    view.roundTitle.textContent = "Round " + (state.form.current + 1);
    refresh();
    renderRoundAdvice();
  }

  function renderRoundAdvice() {
    const round = state.report && state.report.rounds[state.form.current];
    if (!round) { view.roundAdvice.replaceChildren(); view.roundAdvice.hidden = true; return; }
    const bet = round.bet;
    view.roundAdvice.hidden = false;
    view.roundAdvice.replaceChildren(
      el("strong", {}, "Optimal play for round " + round.round),
      el("p", {}, "Best bet: ", ...betText(bet.best, bet.best_pp, bet.best_fct)),
      ...(round.plays.length ? round.plays.map((p) =>
        el("p", {}, (round.plays.some((q) => q.hand === 2) ? "Hand " + p.hand + ": " : "") +
          p.spot + " — " + (DO[p.best] || p.best))) : [el("p", {}, "No playing decision.")])
    );
  }

  function betText(main, pp, fct) {
    const sides = [];
    if (pp) sides.push(T.fmtChips(pp) + " Perfect Pairs");
    if (fct) sides.push(T.fmtChips(fct) + " 4 Card Total");
    return [T.fmtChips(main), sides.length ? el("small", {}, "+ " + sides.join(" + ")) : null];
  }

  function chance(p) { return el("small", {}, T.fmtChance(p) + " ahead"); }

  function mark(fine, ...content) {
    return el("span", { class: fine ? "ok" : "miss" }, ...content, fine ? " ✓" : "");
  }

  function count(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }

  function renderReport() {
    const rep = state.report;
    if (!rep) {
      view.report.replaceChildren(
        el("h2", {}, "The report"),
        el("p", { class: "note" }, "Once every round is in and the chips add up, press Analyze match. " +
          "The best bet and the best play for every round then appear here, next to what you did."),
        el("p", { class: "note" }, "The best bet for the selected round appears above as soon as the earlier rounds are complete."));
      return;
    }
    const rows = rep.rounds.map((r) => {
      const b = r.bet;
      const [bestMain, bestSides] = betText(b.best, b.best_pp, b.best_fct);
      const plays = r.plays.length ? r.plays : null;
      return el("tr", {},
        el("td", {}, String(r.round), el("small", {}, T.fmtChips(r.chips_after))),
        el("td", {}, ...betText(b.yours, b.pp, b.fct), chance(b.your_chance)),
        el("td", {}, mark(b.fine, bestMain), bestSides, chance(b.best_chance)),
        el("td", {}, plays ? plays.map((p) => el("div", { class: "rv-play" },
          DID[p.yours] || p.yours,
          el("small", {}, (r.plays.some((q) => q.hand === 2) ? "Hand " + p.hand + ", " : "") + p.spot),
          chance(p.your_chance))) : el("span", { class: "note" }, "No choice")),
        el("td", {}, plays ? plays.map((p) => el("div", { class: "rv-play" },
          mark(p.fine, DO[p.best] || p.best), chance(p.best_chance))) : "—"));
    });
    const misses = rep.off_bets + rep.off_plays;
    const form = state.form;
    const other = form.opponentFinal === "" ? "" : " The other player finished on " + T.fmtChips(form.opponentFinal) + ".";
    const result = form.won === "won" ? " You won the match." : form.won === "lost" ? " You lost the match." : "";
    const miss = rep.biggest_miss;
    view.report.replaceChildren(
      el("h2", {}, "The report"),
      el("div", { class: "scroll-x" }, el("table", { class: "table rv-table" },
        el("thead", {}, el("tr", {}, ["Round", "Your bet", "Best bet", "Your play", "Best play"].map((h) => el("th", {}, h)))),
        el("tbody", {}, rows))),
      el("p", { class: "note rv-gap" }, "Each % is how often that choice finishes the match ahead. The small number under the round is your chips after it."),
      el("div", { class: "rv-summary" },
        el("p", {}, "You finished on ", el("b", {}, T.fmtChips(rep.final_chips)), "." + other + result + " ",
          misses ? count(rep.off_bets, "bet") + " and " + count(rep.off_plays, "play") + " were off the best line."
            : "Every bet and play was on the best line."),
        miss
          ? el("div", { class: "grade" }, el("strong", {}, "Biggest miss"), el("span", { class: "rv-miss-text" }, miss.text),
              el("span", { class: "note" }, "It cost about " + Math.max(1, Math.round(miss.cost * 100)) + " points of your chance to finish ahead."))
          : el("div", { class: "grade fine" }, el("strong", {}, "No misses"), el("span", { class: "note" }, "Every choice matched the best line.")),
        el("p", { class: "note rv-gap" }, T.BEST_NOTE)),
      el("button", { type: "button", class: "btn rv-gap", onclick: newReview }, "Start a new review"));
  }

  function renderAll() {
    const section = document.getElementById("tab-review");
    view.tabs = el("div", { class: "rv-tabs", role: "group", "aria-label": "Rounds" });
    view.roundTitle = el("span", {});
    view.fields = el("div", { class: "rv-fields" });
    view.sideFields = el("div", {});
    view.start = el("div", {});
    view.end = el("div", {});
    view.outcome = el("div", { class: "rv-outcome", "aria-live": "polite" });
    view.roundAdvice = el("div", { class: "rv-summary", hidden: true, "aria-live": "polite" });
    view.betAdvice = el("div", { class: "rv-summary", "aria-live": "polite" }, "Calculating best bet…");
    view.playAdvice = el("div", { class: "rv-summary", "aria-live": "polite" });
    view.analyze = el("button", { type: "button", class: "btn primary", onclick: analyze }, "Analyze match");
    view.analyzeNote = el("p", { class: "note" });
    view.report = el("div", { class: "box rv-report" });
    view.picker = el("div", { id: "rv-card-dropdown", class: "rv-card-dropdown", role: "dialog", hidden: true });

    const opponent = view.opponent = el("input", {
      type: "number", min: "0", step: "5", inputmode: "numeric", placeholder: "—",
      value: state.form.opponentFinal === "" ? "" : String(state.form.opponentFinal),
      oninput: (event) => setOpponent(event.target.value),
    });
    const won = view.won = el("select", { onchange: (event) => setWon(event.target.value) },
      [["won", "I won"], ["lost", "I lost"], ["unsure", "Not sure"]].map(([v, text]) =>
        el("option", { value: v, selected: state.form.won === v }, text)));

    const entry = el("div", { class: "box rv-entry" },
      view.tabs,
      view.roundAdvice,
      el("div", { class: "rv-workspace" },
        el("div", { class: "rv-hand-entry" },
          view.betAdvice,
          el("div", { class: "rv-stage" },
            el("div", { class: "rv-label" }, view.roundTitle, " · Step 1 · Start of the hand"),
            view.fields, view.start,
            el("p", { class: "note rv-gap" }, "Click a card box to choose its rank."), view.playAdvice),
          el("div", { class: "rv-stage" },
            el("div", { class: "rv-label" }, "Step 2 · How it ended"),
            view.end, view.outcome)),
        el("aside", { class: "rv-side-panel", "aria-label": "Side bets" },
          el("h2", {}, "Side bets"),
          el("p", { class: "note" }, "Suggested side bets are included automatically. A 0 means skip this bet."),
          view.sideFields)),
      el("div", { class: "rv-finish" },
        el("label", { class: "field" }, "Other player's final chips, if you saw them", opponent),
        el("label", { class: "field" }, "How it went", won),
        view.analyze),
      view.analyzeNote,
      el("button", { type: "button", class: "btn rv-small rv-gap", onclick: newReview }, "Start a new review"));

    section.replaceChildren(
      el("div", { class: "rv-head" },
        el("h2", {}, "Review your match"),
        el("p", { class: "note" }, "Enter each round as you go to see the next best bet. " +
          "Once the match is complete, analyze it to compare every bet and play.")),
      el("div", { class: "rv-grid" }, entry, view.report), view.picker);
    renderRound();
    renderReport();
  }

  // ---- keys ------------------------------------------------------------------------

  document.addEventListener("keydown", (event) => {
    if (T.shownTab() !== "review" || event.ctrlKey || event.metaKey || event.altKey) return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
    if (state.pickerOpen) {
      if (event.key === "Escape") { event.preventDefault(); closePicker(true); return; }
      if (event.key === "Tab") { closePicker(true); return; }
      const buttons = [...view.picker.querySelectorAll(".rv-pick")];
      const index = buttons.indexOf(document.activeElement);
      const offsets = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 };
      if (event.key in offsets || event.key === "Home" || event.key === "End") {
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
          : Math.max(0, Math.min(buttons.length - 1, index + offsets[event.key]));
        buttons[next].focus();
        return;
      }
    }
    const key = event.key.toLowerCase();
    const rank = KEYS[key] || (/^[2-9]$/.test(key) ? key : null);
    if (!rank) return;
    event.preventDefault();
    pickRank(rank);
  });

  document.addEventListener("pointerdown", (event) => {
    if (state.pickerOpen && !event.target.closest(".rv-card-dropdown, .rv-slot")) closePicker();
  });
  window.addEventListener("resize", positionPicker);
  document.addEventListener("scroll", positionPicker, true);
  T.onTab("practice", () => closePicker());
  T.onTab("settings", () => closePicker());

  // ---- the test hook: /#review-demo --------------------------------------------------

  // The five-round example from the mockup (tests/test_review.py, MATCH); it ends on 1,700.
  const DEMO = [
    { main: 100, player: ["5", "6"], up: "6", actions: [["double", "K"]], dealer: ["10", "8"] },
    { main: 300, player: ["K", "Q"], up: "8", actions: [["stand"]], dealer: ["9"] },
    { main: 200, player: ["9", "7"], up: "10", actions: [["stand"]], dealer: ["9"] },
    { main: 100, player: ["A", "7"], up: "9", actions: [["stand"]], dealer: ["10"] },
    { main: 500, player: ["K", "Q"], up: "6", actions: [["stand"]], dealer: ["10", "7"] },
  ];

  async function demo() {
    newReview();
    DEMO.forEach((m, i) => {
      selectRound(i);
      setBet("main", m.main);
      [...m.player, m.up].forEach(pickRank);
      for (const [action, card] of m.actions) {
        addAction(action);
        if (card) pickRank(card);
      }
      selectSlot({ kind: "dealer", i: 0 });
      m.dealer.forEach(pickRank);
    });
    setOpponent(1450);
    setWon("won");
    await runCheck();
    await analyze();
    if (state.report) document.title = "review done";  }

  // ---- start -----------------------------------------------------------------------

  renderAll();
  scheduleCheck();
  T.onStatus(() => { if (!state.checks.length) return; showChecks(); });
  T.onReady(() => { state.recommendations = []; loadBetAdvice(); });
  T.onReady(loadPlayAdvice);
  if (location.hash === "#review-demo") {
    let demoStarted = false;
    T.onReady(() => { if (!demoStarted) { demoStarted = true; demo(); } });
  }
})();
