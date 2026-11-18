"""Fit a challenger on an initial review cohort, then compare frozen betting models."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from trainer.brain import Brain, _combos
from trainer.errors import TrainerError
from trainer.history import write_record
from trainer.opponents import build_population
from trainer.review import _finished_replays, _history_class, _replay_or_message
from trainer.settings import Settings, population_from
from trainer.solver import final_distributions

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL = DATA / "calibration" / "model.json"
REPORT = MODEL.with_name("comparison.json")


def rounds_from(record):
    rounds = []
    for raw in record["form"]["rounds"]:
        if not all(raw["player"]):
            break
        r = {key: raw[key] for key in ("main", "pp", "fct", "actions", "player", "upcard", "insurance", "dealer")}
        if r["pp"] and all(raw["suits"]):
            r["suits"] = raw["suits"]
        r["blackjack_outcome"] = raw.get("blackjackOutcome", "blackjack")
        if raw.get("fctResult") not in ("", None):
            r["fct_result"] = float(raw["fctResult"])
        rounds.append(r)
    return rounds


def eligible(record, settings):
    if record.get("status") != "complete":
        raise ValueError("unfinished")
    if Settings.from_dict(record["settings"]).rules != settings.rules:
        raise ValueError("different rules")
    form = record["form"]
    final = form["opponentFinal"]
    if final in (None, "") or not math.isfinite(float(final)) or float(final) <= 0:
        raise ValueError("missing or nonpositive opponent score")
    final = float(final)
    rounds = rounds_from(record)
    replays = _finished_replays(settings.rules, rounds)
    if len(replays) != settings.rules.rounds:
        raise ValueError("incomplete opponent card history")
    chips = replays[-1][0]["chips_after"]
    expected = "won" if chips > final else "lost" if chips < final else "tied"
    if form["won"] != expected:
        raise ValueError("uncertain or conflicting outcome")
    return rounds, replays, final


def rule_likelihoods(brain, settings, history, final):
    # Smooth the discrete chip grid with a Laplace kernel of width 10% of the opening stack.
    # A common normalizing constant cancels in model comparisons.
    width = settings.rules.starting_chips / 10
    prefixes = _combos(history[:-1])
    scores = []
    for label in settings.rule_weights:
        population = build_population({label: 1}, settings.style_weights)
        score = 0.
        for prefix, prob in prefixes:
            finals = final_distributions(settings.rules, population, brain.tables, prefix)
            for g, p in history[-1].items():
                chips, weights = finals[g]
                score += prob * p * np.dot(weights, np.exp(-np.abs(chips - final) / width))
        scores.append(float(score))
    return np.asarray(scores)


def fit_weights(prior, likelihoods):
    # One fractional observation per match. The existing weights encode prior match counts.
    weights = prior.copy()
    for sample in likelihoods:
        posterior = prior * sample
        weights += posterior / posterior.sum()
    return weights


def policy_replay(brain, rounds, history, opponent):
    chips = float(brain.rules.starting_chips)
    bets = []
    for i, raw in enumerate(rounds):
        if brain.rules.floor_bet(chips) < brain.rules.min_bet:
            break
        main, pp, fct = brain.grade_bet(i + 1, chips, history[:i], brain.rules.min_bet).best
        bets.append([main, pp, fct])
        replay, error = _replay_or_message(brain.rules, i + 1, raw | dict(main=main, pp=pp, fct=fct), chips)
        if error or replay.round_.result is None:
            return {"bets": bets, "unscorable": error or "missing cards"}
        chips = replay.round_.result.chips_after
    return {"bets": bets, "chips": chips, "result": "win" if chips > opponent else "loss" if chips < opponent else "tie"}


def run(cutoff=None):
    saved = (json.loads(p.read_text()) for p in (DATA / "reviews").glob("*.json"))
    records = sorted((d for d in saved if d.get("kind") == "review" and "form" in d),
                     key=lambda d: d["created_at"])
    if cutoff:
        if MODEL.exists():
            raise SystemExit("A frozen model already exists. Run without --fit-through to compare.")
        anchor = next((d for d in records if d["id"] == cutoff), None)
        if anchor is None:
            raise SystemExit("Unknown cutoff review ID.")
        cutoff_time = anchor["created_at"]
        settings = Settings.from_dict(anchor["settings"])
    else:
        model = json.loads(MODEL.read_text())
        cutoff_time = model["cutoff_time"]
        settings = Settings.from_dict(model["baseline"])
    baseline = Brain.prepare(settings.rules, population_from(settings), DATA / "cache")
    prior = np.array(list(settings.rule_weights.values()))
    training, future, excluded = [], [], Counter()
    for record in records:
        old = record["created_at"] <= cutoff_time
        if old and not cutoff:
            continue
        try:
            rounds, replays, final = eligible(record, settings)
        except (ValueError, TrainerError) as err:
            excluded[("training" if old else "comparison") + ": " + str(err)] += 1
            continue
        history = [_history_class(baseline, replay) for _, replay in replays]
        sample = rule_likelihoods(baseline, settings, history, final)
        if old:
            training.append((record["id"], sample))
        else:
            future.append((record["id"], rounds, history, final, sample))
    if cutoff:
        if len(training) < 2:
            raise SystemExit("Need at least two eligible training reviews.")
        samples = [sample for _, sample in training]
        fitted = fit_weights(prior, samples)
        loo = []
        for i, sample in enumerate(samples):
            heldout = fit_weights(prior, samples[:i] + samples[i+1:])
            loo.append(float(np.log(np.average(sample, weights=heldout) / np.average(sample, weights=prior))))
        model = {"cutoff_time": cutoff_time, "baseline": settings.to_dict(),
                 "candidate_weights": dict(zip(settings.rule_weights, fitted)), "training_ids": [k for k, _ in training],
                 "training_exclusions": {k: v for k, v in excluded.items() if k.startswith("training:")}, "leave_one_out_mean_log_gain": float(np.mean(loo))}
        write_record(MODEL, model)
    revised = Settings.from_dict(model["baseline"] | {"rule_weights": model["candidate_weights"]})
    candidate = Brain.prepare(revised.rules, population_from(revised), DATA / "cache")
    rows = []
    for key, rounds, history, final, sample in future:
        rows.append({"review_id": key, "opponent_final": final,
                     "forecast_log_gain": float(np.log(np.average(sample, weights=list(revised.rule_weights.values())) /
                                                       np.average(sample, weights=prior))),
                     "baseline": policy_replay(baseline, rounds, history, final),
                     "candidate": policy_replay(candidate, rounds, history, final)})
    report = {"training_matches": len(model["training_ids"]), "leave_one_out_mean_log_gain": model["leave_one_out_mean_log_gain"],
              "opening_bets": [brain.grade_bet(1, settings.rules.starting_chips, [], settings.rules.min_bet).best
                               for brain in (baseline, candidate)],
              "comparison_exclusions": {k: v for k, v in excluded.items() if k.startswith("comparison:")},
              "new_matches": rows}
    write_record(REPORT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-through", metavar="REVIEW_ID", help="freeze the initial cohort through this review")
    run(parser.parse_args().fit_through)
