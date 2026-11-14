// The settings screen: the few rules that can change, the stand-in opponents, and the default mode.
(function () {
  "use strict";

  const T = window.Trainer;
  const el = T.el;
  const STYLE_NAMES = { book: "Plays by the book", cautious: "Plays cautiously", dealer_copy: "Copies the dealer" };

  const state = { filled: false, saving: false, defaults: { rule: {}, style: {} } };
  const fields = {};     // every input and select, by name
  let ruleBox = null;
  let styleBox = null;
  let saveButton = null;
  let savedNote = null;

  // ---- talking to the server ---------------------------------------------------------

  async function save() {
    let body;
    try {
      body = readForm();
    } catch (err) {
      T.showError(err.message);
      return;
    }
    state.saving = true;
    saveButton.disabled = true;
    savedNote.textContent = "";
    try {
      const answer = await T.api("POST", "/api/settings", body);
      fill(answer.settings);
      savedNote.textContent = answer.resolving
        ? "Saved. The trainer is working out the best bets again for these settings."
        : "Saved.";
      if (answer.resolving) T.refreshStatus();
    } catch (err) {
      /* the error strip already says why */
    } finally {
      state.saving = false;
      saveButton.disabled = false;
    }
  }

  // ---- the form ----------------------------------------------------------------------

  function number(name, label) {
    const value = Number(fields[name].value);
    if (fields[name].value.trim() === "" || !Number.isFinite(value)) throw new Error(label + " has to be a number.");
    return value;
  }

  function readForm() {
    const weights = (prefix, names, what) => Object.fromEntries(names.map((name) =>
      [name, number(prefix + name, "The weight for " + what(name))]));
    return {
      rules: {
        min_bet: number("min_bet", "The minimum bet"),
        tie_credit: Number(fields.tie_credit.value),
        double_after_split: fields.double_after_split.checked,
        split_by_value: fields.split_by_value.value === "value",
        insurance_offered: fields.insurance_offered.checked,
        clock_seconds: number("clock_seconds", "The clock length"),
      },
      rule_weights: weights("rule:", Object.keys(state.defaults.rule), (n) => n),
      style_weights: weights("style:", Object.keys(state.defaults.style), (n) => STYLE_NAMES[n] || n),
      default_mode: fields.mode_match.checked ? "match" : "coach",
    };
  }

  // Put saved settings into the fields.
  function fill(settings) {
    const rules = settings.rules;
    fields.min_bet.value = rules.min_bet;
    fields.tie_credit.value = String(rules.tie_credit);
    fields.double_after_split.checked = rules.double_after_split;
    fields.split_by_value.value = rules.split_by_value ? "value" : "rank";
    fields.insurance_offered.checked = rules.insurance_offered;
    fields.clock_seconds.value = rules.clock_seconds;
    for (const [name, weight] of Object.entries(settings.rule_weights)) {
      if (fields["rule:" + name]) fields["rule:" + name].value = weight;
    }
    for (const [name, weight] of Object.entries(settings.style_weights)) {
      if (fields["style:" + name]) fields["style:" + name].value = weight;
    }
    fields.mode_coach.checked = settings.default_mode !== "match";
    fields.mode_match.checked = settings.default_mode === "match";
  }

  function resetWeights() {
    for (const [name, weight] of Object.entries(state.defaults.rule)) fields["rule:" + name].value = weight;
    for (const [name, weight] of Object.entries(state.defaults.style)) fields["style:" + name].value = weight;
    savedNote.textContent = "The defaults are back in the form. Press Save to use them.";
  }

  function weightField(key, label, fallback) {
    fields[key] = el("input", { type: "number", min: "0", step: "1", inputmode: "decimal" });
    return el("label", { class: "field" }, label, fields[key], el("span", { class: "st-default" }, "default " + fallback));
  }

  // The weight fields depend on the names the server sends, so they are drawn from the status.
  function renderWeights(status) {
    state.defaults = { rule: status.default_rule_weights, style: status.default_style_weights };
    ruleBox.replaceChildren(...Object.entries(state.defaults.rule).map(([name, weight]) =>
      weightField("rule:" + name, name, weight)));
    styleBox.replaceChildren(...Object.entries(state.defaults.style).map(([name, weight]) =>
      weightField("style:" + name, STYLE_NAMES[name] || name, weight)));
  }

  function check(name, label) {
    fields[name] = el("input", { type: "checkbox" });
    return el("label", { class: "st-check" }, fields[name], label);
  }

  function render() {
    fields.min_bet = el("input", { type: "number", min: "5", step: "5", inputmode: "numeric" });
    fields.tie_credit = el("select", {},
      el("option", { value: "0.5" }, "Half a win"), el("option", { value: "0" }, "A loss"), el("option", { value: "1" }, "A win"));
    fields.split_by_value = el("select", {},
      el("option", { value: "value" }, "By value (10 with K)"),
      el("option", { value: "rank" }, "By rank (K with K)"));
    fields.clock_seconds = el("input", { type: "number", min: "30", max: "3600", step: "10", inputmode: "numeric" });
    fields.mode_coach = el("input", { type: "radio", name: "st-mode", value: "coach" });
    fields.mode_match = el("input", { type: "radio", name: "st-mode", value: "match" });
    ruleBox = el("div", { class: "st-weights" });
    styleBox = el("div", { class: "st-weights" });
    saveButton = el("button", { type: "button", class: "btn primary", onclick: save }, "Save");
    savedNote = el("p", { class: "note", "aria-live": "polite" });

    document.getElementById("tab-settings").replaceChildren(
      el("div", { class: "box" },
        el("h2", {}, "Rules"),
        el("div", { class: "st-grid" },
          el("label", { class: "field" }, "Minimum bet", fields.min_bet),
          el("label", { class: "field" }, "Equal chips at the end count as", fields.tie_credit),
          el("label", { class: "field" }, "Which pairs can be split", fields.split_by_value),
          el("label", { class: "field" }, "Clock length, in seconds", fields.clock_seconds)),
        el("div", { class: "st-checks" },
          check("double_after_split", "Double after a split is allowed"),
          check("insurance_offered", "Insurance is offered when the dealer shows an ace"))),
      el("div", { class: "box" },
        el("h2", {}, "Opponents"),
        el("p", {}, "These are guesses about how real opponents bet. Every best bet depends on them."),
        el("p", { class: "note" }, "Each number says how common that kind of opponent is, compared with the others. 0 leaves it out."),
        el("h3", { class: "st-sub" }, "How they bet"),
        ruleBox,
        el("h3", { class: "st-sub" }, "How they play their cards"),
        styleBox,
        el("button", { type: "button", class: "btn st-reset", onclick: resetWeights }, "Reset to defaults")),
      el("div", { class: "box" },
        el("h2", {}, "Default mode"),
        el("div", { class: "st-checks" },
          el("label", { class: "st-check" }, fields.mode_coach, "Coach: grades after every round, the clock waits between rounds"),
          el("label", { class: "st-check" }, fields.mode_match, "Match: the clock keeps running, grades at the end"))),
      el("div", { class: "st-save" }, saveButton, savedNote,
        el("p", { class: "note" }, "Changing the rules or the opponents means the best bets are worked out again, which takes a minute or two.")));
  }

  // ---- start -------------------------------------------------------------------------

  render();
  T.onStatus((status) => {
    if (state.filled) return;           // never overwrite what the user is typing
    state.filled = true;
    renderWeights(status);
    fill(status.settings);
    if (location.hash === "#settings-demo") document.title = "settings done";
  });
})();
