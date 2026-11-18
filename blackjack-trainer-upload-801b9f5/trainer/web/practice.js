// The practice table: start a match, bet, play, see the grades, and the end-of-match summary.
// Everything is drawn into <section id="tab-practice"> from the last state the trainer sent.
(function () {
  "use strict";

  const T = window.Trainer;
  const el = T.el;
  const fmt = T.fmtChips;

  const CHIPS = [5, 10, 25, 100, 500, 1000];
  const TARGETS = { main: "Main bet", pp: "Perfect Pairs", fct: "4 Card Total" };
  const ACTIONS = { hit: "Hit", stand: "Stand", double: "Double", split: "Split",
    insurance: "Insurance", no_insurance: "No insurance" };
  const ACTION_KEYS = { hit: "H", stand: "S", double: "D", split: "P", insurance: "I", no_insurance: "N" };
  const OUTCOMES = { win: "won", lose: "lost", push: "push", blackjack: "blackjack", bust: "bust" };
  const MODES = {
    coach: { name: "Coach", key: "C", text: "Grades after every round." },
    match: { name: "Match", key: "M", text: "Grades only when all rounds are done, like a real match." },
  };

  const root = document.getElementById("tab-practice");

  const state = {
    ready: false,               // the brain is solved, so practice routes answer
    loaded: false,              // the first state request has come back
    match: null,                // the last state the trainer sent; null when no match is running
    receivedAt: 0,              // performance.now() when `match` arrived, for the local clock
    mode: null,                 // the Coach / Match choice on the start screen
    target: "main",             // where chip presses go
    draft: { main: 0, pp: 0, fct: 0 },
    draftRound: null,           // the round the draft bet belongs to
    busy: false,
    pending: Promise.resolve(), // settles when the request in flight has been answered and drawn
    lastRefetch: -Infinity,
  };

  // ---- talking to the trainer ------------------------------------------------

  // One request at a time; the screen keeps its last good state when one is refused.
  function request(method, path, body, quiet) {
    if (state.busy) return state.pending;
    state.busy = true;
    state.pending = T.api(method, path, body, { quiet: !!quiet })
      .then(accept, (err) => { if (err.status === 404) state.match = null; })
      .finally(() => {
        state.busy = false;
        state.loaded = true;
        render();
      });
    return state.pending;
  }

  function accept(answer) {
    state.match = answer;
    state.receivedAt = performance.now();
    if (answer.save_error) T.showError(answer.save_error);
    if (answer.phase === "bet" && state.draftRound !== answer.round) {
      state.draft = { main: 0, pp: 0, fct: 0 };
      state.target = "main";
      state.draftRound = answer.round;
    }
  }

  const loadState = () => request("GET", "/api/practice/state", undefined, true);
  const startMatch = () => request("POST", "/api/practice/start", { mode: chosenMode() });
  const nextRound = () => request("POST", "/api/practice/next", { step: state.match.step });

  function deal() {
    if (state.draft.main <= 0) {
      T.showError("Add chips to your main bet first.");
      return;
    }
    request("POST", "/api/practice/bet", { ...state.draft, step: state.match.step });
  }

  function act(action) {
    if (!state.match.legal.includes(action)) {
      T.showError(ACTIONS[action] + " is not possible right now.");
      return;
    }
    request("POST", "/api/practice/action", { action, step: state.match.step });
  }

  // ---- the bet being built ---------------------------------------------------

  function draftTotal() {
    return state.draft.main + state.draft.pp + state.draft.fct;
  }

  function betStep() {
    const status = T.getStatus();
    return (status && status.settings.rules.bet_step) || 5;
  }

  function addChip(value) {
    if (draftTotal() + value > state.match.chips + 1e-9) {
      T.showError("That is more than your " + fmt(state.match.chips) + " chips.");
      return;
    }
    state.draft[state.target] += value;
    render();
  }

  function clearBet() {
    state.draft = { main: 0, pp: 0, fct: 0 };
    state.target = "main";
    render();
  }

  function allIn() {
    const step = betStep();
    const room = state.match.chips - state.draft.pp - state.draft.fct;
    state.draft.main = Math.floor(room / step + 1e-9) * step;
    render();
  }

  function sameBetPossible() {
    const last = state.match.last_bet;
    return !!last && last.main + last.pp + last.fct <= state.match.chips + 1e-9;
  }

  function sameBet() {
    if (!sameBetPossible()) {
      T.showError("Your last bet is not possible with the chips you have.");
      return;
    }
    const last = state.match.last_bet;
    state.draft = { main: last.main, pp: last.pp, fct: last.fct };
    render();
  }

  function selectTarget(name) {
    state.target = name;
    render();
  }

  function chooseMode(mode) {
    state.mode = mode;
    render();
  }

  function chosenMode() {
    if (state.mode) return state.mode;
    const status = T.getStatus();
    return (status && status.settings.default_mode) || "coach";
  }

  // ---- keys ------------------------------------------------------------------

  // The command a key stands for in the current phase, or null.
  function commandFor(key) {
    const m = state.match;
    const phase = m ? m.phase : "start";
    if (phase === "start" || phase === "match_over") {
      if (key === "Enter") return startMatch;
      if (key === "c") return () => chooseMode("coach");
      if (key === "m") return () => chooseMode("match");
    } else if (phase === "bet") {
      const chip = "123456".indexOf(key);
      if (key.length === 1 && chip >= 0) return () => addChip(CHIPS[chip]);
      if (key === "a") return allIn;
      if (key === "c") return clearBet;
      if (key === "r") return sameBet;
      if (key === "Enter") return deal;
    } else if (phase === "insurance" || phase === "play") {
      const action = Object.keys(ACTION_KEYS).find((name) => ACTION_KEYS[name].toLowerCase() === key);
      const fits = phase === "insurance" ? ["insurance", "no_insurance"] : ["hit", "stand", "double", "split"];
      if (fits.includes(action)) return () => act(action);
    } else if (phase === "round_over") {
      if (key === "Enter") return nextRound;
    }
    return null;
  }

  function onKey(event) {
    if (event.ctrlKey || event.altKey || event.metaKey || event.repeat) return;
    if (!state.ready || !state.loaded || T.shownTab() !== "practice") return;
    if (event.target instanceof Element && event.target.closest("input, select, textarea")) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    const command = commandFor(key);
    if (!command) return;
    event.preventDefault();
    if (!state.busy) command();
  }

  // ---- the clock -------------------------------------------------------------

  function clockLeft() {
    const clock = state.match.clock;
    const gone = clock.running ? (performance.now() - state.receivedAt) / 1000 : 0;
    return Math.max(0, clock.left - gone);
  }

  function fmtClock(seconds) {
    const whole = Math.ceil(seconds);
    return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
  }

  function tickClock() {
    const node = document.getElementById("pr-clock");
    if (!node || !state.match) return;
    const left = clockLeft();
    node.textContent = fmtClock(left);
    node.classList.toggle("low", left < 60);
    // At zero, ask the trainer what happens next (a Match is forfeit), at most every few seconds.
    const now = performance.now();
    if (left <= 0 && state.match.clock.running && !state.busy && now - state.lastRefetch > 5000) {
      state.lastRefetch = now;
      loadState();
    }
  }

  // ---- drawing: shared pieces ------------------------------------------------

  function button(label, key, onclick, options) {
    const opts = options || {};
    // The second click of a double-click would land on whatever button is drawn next, so drop it.
    const once = (event) => { if (event.detail <= 1) onclick(); };
    return el("button", { class: "btn " + (opts.cls || ""), type: "button", disabled: !!opts.disabled,
      onclick: once, title: key ? "Key: " + key : null }, label, key ? el("kbd", {}, key) : null);
  }

  function signed(amount) {
    if (amount > 0) return "+" + fmt(amount);
    if (amount < 0) return "−" + fmt(-amount);
    return "0";
  }

  function betText(bet) {
    let text = fmt(bet.main);
    if (bet.pp) text += " + " + fmt(bet.pp) + " Perfect Pairs";
    if (bet.fct) text += " + " + fmt(bet.fct) + " 4 Card Total";
    if (bet.insurance) text += " + " + fmt(bet.insurance) + " insurance";
    return text;
  }

  function netText(net) {
    if (net > 0) return "You won " + fmt(net);
    if (net < 0) return "You lost " + fmt(-net);
    return "No chips won or lost";
  }

  function statusBar(m) {
    const clock = el("b", { id: "pr-clock", class: "pr-clock" }, fmtClock(clockLeft()));
    const showBet = m.bet && (m.phase === "insurance" || m.phase === "play");
    return el("div", { class: "pr-bar" },
      el("span", {}, "Round ", el("b", {}, m.round), " of ", m.rounds),
      el("span", { class: "pr-mode" }, MODES[m.mode].name),
      el("span", {}, "Clock ", clock),
      showBet ? el("span", {}, "Bet ", el("b", {}, fmt(m.bet.main))) : null,
      el("span", {}, "Chips ", el("b", {}, fmt(m.chips))));
  }

  function handLabel(hand, index, count) {
    const parts = [count > 1 ? "Hand " + (index + 1) : "You"];
    parts.push((hand.soft && hand.total <= 21 ? "soft " : "") + hand.total);
    if (count > 1 || (state.match.bet && hand.stake !== state.match.bet.main)) parts.push("stake " + fmt(hand.stake));
    if (hand.outcome) parts.push(OUTCOMES[hand.outcome] || hand.outcome);
    return parts.join(" · ");
  }

  function tableArea(m) {
    let dealerLabel = "Dealer";
    if (m.dealer_total !== null) dealerLabel += " · " + m.dealer_total + (m.dealer_total > 21 ? " · bust" : "");
    return el("div", { class: "pr-table" },
      el("div", { class: "pr-hand" },
        el("div", { class: "pr-label" }, dealerLabel),
        el("div", { class: "cards" }, m.dealer.map(T.cardEl))),
      el("div", { class: "pr-hands" },
        m.hands.map((hand, i) => el("div", { class: "pr-hand" + (hand.active ? " active" : "") },
          el("div", { class: "pr-label" }, handLabel(hand, i, m.hands.length),
            hand.active && m.hands.length > 1 ? el("span", { class: "pr-turn" }, "your turn") : null),
          el("div", { class: "cards" }, hand.cards.map(T.cardEl))))));
  }

  function gradeTitle(g) {
    if (g.kind === "bet") return "Bet · " + g.spot;
    const what = g.kind === "insurance" ? "Insurance" : "Play";
    return what + " · " + g.spot + (g.hand > 1 ? " (hand " + g.hand + ")" : "");
  }

  function gradeList(grades) {
    return el("div", { class: "pr-grades" }, grades.map((g) =>
      el("div", { class: "grade" + (g.fine ? " fine" : "") },
        el("div", { class: "pr-grade-head" },
          el("b", {}, gradeTitle(g)),
          g.fine ? el("span", { class: "ok" }, "✓ fine") : el("span", { class: "miss" }, "miss")),
        el("div", { class: "pr-grade-cols" },
          el("span", {}, "Yours: ", el("b", {}, g.yours), " ",
            el("span", { class: "pr-chance" }, T.fmtChance(g.your_chance))),
          el("span", {}, "Best: ", el("b", {}, g.best), " ",
            el("span", { class: "pr-chance" }, T.fmtChance(g.best_chance)))),
        el("span", { class: "note" }, g.reason))));
  }

  function bestNote() {
    return el("p", { class: "note pr-best-note" },
      "Percentages are the chance of finishing ahead. " + T.BEST_NOTE);
  }

  function keysBox(lines) {
    return el("div", { class: "box pr-keys" },
      el("h3", {}, "Keys"),
      el("ul", {}, lines.map(([keys, text]) => el("li", {}, el("kbd", {}, keys), " ", text))));
  }

  function layout(felt, aside) {
    return el("div", { class: "pr-layout" }, felt, el("div", { class: "pr-aside" }, aside));
  }

  function modeChoice() {
    const chosen = chosenMode();
    return el("div", { class: "pr-modes", role: "radiogroup", "aria-label": "Mode" },
      Object.entries(MODES).map(([mode, info]) =>
        el("button", { type: "button", role: "radio", class: "pr-mode-btn" + (mode === chosen ? " selected" : ""),
          "aria-checked": mode === chosen ? "true" : "false", onclick: () => chooseMode(mode) },
          el("b", {}, info.name, el("kbd", {}, info.key)), el("span", {}, info.text))));
  }

  // ---- drawing: one function per phase ---------------------------------------

  function renderWaiting() {
    return el("div", { class: "box" },
      el("h2", {}, "Practice table"),
      el("p", { class: "note" }, "The table opens as soon as the best bets have been worked out."));
  }

  function renderStart() {
    return el("div", { class: "box pr-start" },
      el("h2", {}, "Start a match"),
      el("p", {}, "Five rounds against a hidden stand-in opponent. Each round you bet, then play, "
        + "and every bet and play is graded by how often it finishes ahead."),
      modeChoice(),
      button("Start", "Enter", startMatch, { cls: "primary" }));
  }

  function renderBet(m) {
    const d = state.draft;
    const room = m.chips - draftTotal();
    const felt = el("div", { class: "felt pr-felt" },
      statusBar(m),
      el("button", { type: "button", class: "pr-target pr-main" + (state.target === "main" ? " selected" : ""),
        onclick: () => selectTarget("main") },
        el("span", { class: "pr-small" }, "Your bet"),
        el("span", { class: "pr-big" }, fmt(d.main))),
      el("div", { class: "pr-chips" }, CHIPS.map((value, i) =>
        el("button", { type: "button", class: "chip chip-" + value, disabled: value > room + 1e-9,
          title: "Key: " + (i + 1), onclick: () => addChip(value) },
          fmt(value), el("kbd", {}, i + 1)))),
      el("div", { class: "pr-row" },
        button("Clear", "C", clearBet, { cls: "ghost" }),
        button("All in", "A", allIn, { cls: "ghost" }),
        m.last_bet ? button("Same bet", "R", sameBet, { cls: "ghost", disabled: !sameBetPossible() }) : null),
      el("div", { class: "pr-side" }, ["pp", "fct"].map((name) =>
        el("button", { type: "button", class: "pr-target" + (state.target === name ? " selected" : ""),
          onclick: () => selectTarget(name) }, el("span", {}, TARGETS[name]), el("b", {}, fmt(d[name]))))),
      el("p", { class: "pr-small pr-adding" }, "Chips go on: ", el("b", {}, TARGETS[state.target])),
      button("Deal", "Enter", deal, { cls: "primary pr-wide", disabled: d.main <= 0 }));
    return layout(felt, keysBox([
      ["1–6", "add a 5, 10, 25, 100, 500 or 1,000 chip"],
      ["A", "all in"], ["C", "clear"], ["R", "same bet as last round"], ["Enter", "deal"],
    ]));
  }

  function renderHandPhase(m) {
    const names = m.phase === "insurance" ? ["insurance", "no_insurance"] : ["hit", "stand", "double", "split"];
    const felt = el("div", { class: "felt pr-felt" },
      statusBar(m),
      tableArea(m),
      m.phase === "insurance" ? el("p", { class: "pr-ask" }, "The dealer shows an ace. Insurance?") : null,
      el("div", { class: "pr-row" }, names.map((name) =>
        button(ACTIONS[name], ACTION_KEYS[name], () => act(name), { disabled: !m.legal.includes(name) }))));
    const lines = m.phase === "insurance"
      ? [["I", "insurance"], ["N", "no insurance"]]
      : [["H", "hit"], ["S", "stand"], ["D", "double"], ["P", "split"]];
    return layout(felt, keysBox(lines));
  }

  function resultLine(m) {
    const r = m.result;
    const extras = [];
    if (m.bet.pp) extras.push("Perfect Pairs " + signed(r.pp));
    if (m.bet.fct) extras.push("4 Card Total " + signed(r.fct));
    if (m.bet.insurance) extras.push("Insurance " + signed(r.insurance));
    return el("div", { class: "pr-result" },
      el("div", {}, netText(r.net)),
      extras.length ? el("div", { class: "pr-small" }, "Main " + signed(r.main) + " · " + extras.join(" · ")) : null);
  }

  function renderRoundOver(m) {
    const last = m.round >= m.rounds;
    const felt = el("div", { class: "felt pr-felt" },
      statusBar(m),
      tableArea(m),
      resultLine(m),
      button(last ? "See the result" : "Next round", "Enter", nextRound, { cls: "primary pr-wide" }));
    const aside = m.grades
      ? el("div", { class: "box" }, el("h3", {}, "Round " + m.round + " grades"), gradeList(m.grades), bestNote())
      : el("div", { class: "box" }, el("h3", {}, "Grades"),
        el("p", { class: "note" }, "In Match mode the grades are kept until all rounds are done."));
    return layout(felt, aside);
  }

  function outcomeWords(s) {
    if (s.outcome === "forfeit") return "Time ran out, so the match is forfeit: you finished with a loss.";
    return "You finished " + s.outcome;
  }

  function renderMatchOver(m) {
    const s = m.summary;
    const grades = s.rounds.flatMap((r) => r.grades);
    const fine = grades.filter((g) => g.fine).length;
    return el("div", { class: "pr-end" },
      el("div", { class: "box pr-end-head pr-" + s.outcome },
        el("h2", {}, outcomeWords(s)),
        el("p", { class: "pr-totals" },
          el("span", {}, "You ", el("b", {}, fmt(s.you))),
          el("span", {}, "Opponent ", el("b", {}, fmt(s.opponent)))),
        el("p", {}, "Your opponent was ", el("b", {}, s.opponent_name), "."),
        el("p", { class: "note" }, "Playing the best bets and plays from the start finishes ahead "
          + T.fmtChance(s.start_chance) + " of the time. You made " + fine + " of " + grades.length
          + " choices the best way or close to it."),
        modeChoice(),
        button("Play again", "Enter", startMatch, { cls: "primary" })),
      el("div", { class: "box" },
        el("h3", {}, "Round by round"),
        s.rounds.map((r) => el("div", { class: "pr-round" },
          el("div", { class: "pr-round-head" },
            el("b", {}, "Round " + r.round),
            el("span", {}, "Bet " + betText(r.bet)),
            el("span", {}, "Net ", el("b", { class: r.net > 0 ? "ok" : r.net < 0 ? "miss" : "" }, signed(r.net))),
            el("span", {}, fmt(r.chips_after) + " chips after")),
          gradeList(r.grades))),
        s.rounds.length ? bestNote() : el("p", { class: "note" }, "No rounds were finished.")));
  }

  const PHASES = {
    bet: renderBet,
    insurance: renderHandPhase,
    play: renderHandPhase,
    round_over: renderRoundOver,
    match_over: renderMatchOver,
  };

  function render() {
    let view;
    if (!state.ready || !state.loaded) view = renderWaiting();
    else if (!state.match) view = renderStart();
    else view = PHASES[state.match.phase](state.match);
    if (state.match && state.match.saved_file) view.append(el("p", { class: "note rv-gap" },
      "Saved automatically on this computer · data/matches"));
    root.replaceChildren(view);
    tickClock();
  }

  // ---- test hooks: /#autoplay and /#autoplay-coach ---------------------------

  // Plays one match to the end through the key handler only, awaiting each answer.
  async function autoplay(mode) {
    const press = async (key) => {
      await state.pending;
      document.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
      await state.pending;
    };
    let started = false;
    for (let step = 0; step < 400; step++) {
      const m = state.match;
      const phase = m ? m.phase : "start";
      if (phase === "start" || phase === "match_over") {
        if (started) break;
        started = true;
        await press(mode === "coach" ? "c" : "m");
        await press("Enter");
      } else if (phase === "bet") {
        await press(m.chips >= 100 ? "4" : "a");
        await press("Enter");
      } else if (phase === "insurance") {
        await press("n");
      } else if (phase === "play") {
        const hand = m.hands.find((h) => h.active) || m.hands[0];
        await press(hand.total >= 17 ? "s" : "h");
      } else if (phase === "round_over") {
        await press("Enter");
      }
    }
    document.title = "autoplay done";
  }

  function autoplayMode() {
    if (location.hash === "#autoplay") return "match";
    if (location.hash === "#autoplay-coach") return "coach";
    return null;
  }

  // ---- start up --------------------------------------------------------------

  let autoplayed = false;
  T.onReady(() => {
    state.ready = true;
    state.loaded = false;
    render();
    loadState().then(() => {
      const mode = autoplayMode();
      if (mode && !autoplayed) {
        autoplayed = true;
        autoplay(mode);
      }
    });
  });
  // While the brain is re-solved the old match is gone and practice routes refuse; wait.
  T.onStatus((status) => {
    if (!status.brain.ready && state.ready) {
      state.ready = false;
      render();
    }
  });
  T.onTab("practice", () => { if (state.ready && !state.busy) loadState(); });
  document.addEventListener("keydown", onKey);
  setInterval(tickClock, 250);
  render();
})();
