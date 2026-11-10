"""Optuna TPE baseline: a standard optimizer that learns from results but has no prior knowledge.

Ask/tell: TPE proposes a full config, it goes through tools.run_experiment as
changes against the current best (like random search: same guardrails,
duplicate check, accept rule, budget and log), and the dev loss goes back to
TPE. The start config is told as trial 0. TPE's best is not the result: the
accept rule's best is, as for every searcher.

Encoding of the search space for TPE: a nullable parameter gets a "<name>__null"
flag plus its value, and each temperature-dynamics feature is its own in/out
flag, mirroring random search's sampling. TPE starts with 3 random trials
instead of Optuna's default 10, so it is already modelling at the agent's
compute (4-7 experiments; NOTES.md, Design v2).

Usage: python main.py --searcher optuna --seed 0
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

from src import tools
from src.search_space import SEARCH_SPACE, START_PARAMS, to_overrides

STARTUP_TRIALS = 3
MAX_REJECTED_IN_A_ROW = 100  # as in random search


def distributions() -> dict[str, BaseDistribution]:
    dists: dict[str, BaseDistribution] = {}
    for name, p in SEARCH_SPACE.items():
        if p.nullable:
            dists[f"{name}__null"] = CategoricalDistribution([True, False])
        if p.kind == "int":
            dists[name] = IntDistribution(int(p.low), int(p.high), log=p.log, step=p.step)
        elif p.kind == "float":
            dists[name] = FloatDistribution(p.low, p.high, log=p.log)
        elif p.kind == "bool":
            dists[name] = CategoricalDistribution([True, False])
        elif p.kind == "choice":
            dists[name] = CategoricalDistribution(list(p.choices))
        else:  # subset
            for choice in p.choices:
                dists[f"{name}__{choice}"] = CategoricalDistribution([True, False])
    return dists


def decode(values: dict[str, Any]) -> dict[str, Any]:
    """TPE's flat values -> search-space params."""
    params: dict[str, Any] = {}
    for name, p in SEARCH_SPACE.items():
        if p.kind == "subset":
            params[name] = [c for c in p.choices if values[f"{name}__{c}"]]
        elif p.nullable and values[f"{name}__null"]:
            params[name] = None
        else:
            params[name] = values[name]
    return params


def encode(params: dict[str, Any]) -> dict[str, Any]:
    """Search-space params -> TPE's flat values (a null parameter's value slot gets its lower bound)."""
    values: dict[str, Any] = {}
    for name, p in SEARCH_SPACE.items():
        if p.kind == "subset":
            values.update({f"{name}__{c}": c in params[name] for c in p.choices})
            continue
        if p.nullable:
            values[f"{name}__null"] = params[name] is None
        values[name] = (int(p.low) if p.kind == "int" else p.low) if params[name] is None else params[name]
    return values


def run(seed: int, experiments: int | None = None) -> tools.SearchState:
    """Ask, run, tell until a budget is used up (or too many proposals in a row were rejected)."""
    state = tools.start_search(f"optuna_s{seed}_{datetime.now():%Y%m%d-%H%M%S}", "optuna", seed,
                               experiments=experiments)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=STARTUP_TRIALS, seed=seed))
    dists = distributions()
    study.add_trial(optuna.trial.create_trial(params=encode(START_PARAMS), distributions=dists,
                                              value=state.best.dev_loss))
    losses = {_key(START_PARAMS): state.best.dev_loss}  # canonical config -> dev loss, to answer repeats
    rejected_in_a_row = 0
    while state.budget.exhausted() is None and rejected_in_a_row < MAX_REJECTED_IN_A_ROW:
        trial = study.ask(dists)
        params = decode(trial.params)
        changes = {k: v for k, v in params.items() if v != state.best_params[k]}
        result = tools.run_experiment(state, changes, hypothesis="optuna TPE", expected_change_pct=None)
        if "dev_loss" in result:
            losses[_key(params)] = result["dev_loss"]
            study.tell(trial, result["dev_loss"])
            rejected_in_a_row = 0
        elif "rejected" in result and _key(params) in losses:  # a repeat (or the best itself): TPE learns its loss
            study.tell(trial, losses[_key(params)])
            rejected_in_a_row += 1
        else:  # failed GEFCom run, or a rejection without a known loss
            study.tell(trial, state=TrialState.FAIL)
            rejected_in_a_row += "rejected" in result
    stop_reason = state.budget.exhausted() or "too many rejected proposals"
    tools.finish(state, state.best_id, "optuna TPE: best by the accept rule", stop_reason=stop_reason)
    return state


def _key(params: dict[str, Any]) -> str:
    return json.dumps(to_overrides(params), sort_keys=True)
