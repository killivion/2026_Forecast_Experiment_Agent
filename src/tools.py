"""The five agent tools: JSON schemas for the Claude API and their Python handlers.

The handlers are searcher-agnostic. Random search calls run_experiment and
finish directly, so both searchers go through the same guardrails, accept
rule, budget and log format; only the choice of the next proposal differs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import yaml
from anthropic.types import ToolParam

from src.guardrails import Budget, Rejection, check_proposal
from src.logging_utils import RunLog
from src.search_space import SEARCH_SPACE, START_PARAMS, accept, month_wins, to_overrides
from src.tasks.gefcom import REPO_ROOT, EvalResult, GefcomRunError, Tier, evaluate

TIER: Tier = "fast"  # searchers only ever see fast-tier results on the development folds
MONTHS = {2: "Oct 2010", 3: "Nov 2010", 4: "Dec 2010", 5: "Jan 2011", 6: "Feb 2011", 7: "Mar 2011", 8: "Apr 2011"}

TOOLS: list[ToolParam] = [
    {
        "name": "describe_task",
        "description": "The forecasting problem, the metric, what is known at forecast time, the rules of "
                       "this search, the current best config and the remaining budget.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_search_space",
        "description": "Every parameter you may change, with its type, range or choices and what it does.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_experiment",
        "description": "Change some parameters of the current best config and backtest the result on the "
                       "7 development months (takes 1-15 minutes). The result says whether the change was "
                       "accepted as the new best. Rejected proposals (invalid, leaking or duplicate) are not "
                       "run and use no budget.",
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {"type": "object",
                            "description": "Parameter name -> new value, relative to the current best config. "
                                           "Names and allowed values: get_search_space."},
                "hypothesis": {"type": "string",
                               "description": "Why you expect this change to help, in one or two sentences."},
                "expected_change_pct": {"type": "number",
                                        "description": "Your prediction of the dev loss change vs. the current "
                                                       "best, in percent (negative = better, e.g. -2.0)."},
            },
            "required": ["changes", "hypothesis", "expected_change_pct"],
        },
    },
    {
        "name": "get_history",
        "description": "All experiments of this search so far: changes, dev loss, coverage, accepted or not.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": "End the search. Name the experiment you consider best and say why you stopped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "best_run_id": {"type": "integer", "description": "Experiment id you consider best."},
                "rationale": {"type": "string", "description": "Why this config, and why stop now."},
            },
            "required": ["best_run_id", "rationale"],
        },
    },
]


@dataclass
class SearchState:
    log: RunLog
    budget: Budget
    min_rel_improvement: float
    min_month_wins: int
    best_params: dict[str, Any]
    best: EvalResult
    best_id: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)  # compact rows for get_history
    seen: dict[str, int] = field(default_factory=dict)  # canonical overrides (JSON) -> experiment id
    leakage_hits: int = 0
    finished: bool = False

    def experiments_left(self) -> int:
        return self.budget.max_experiments - self.budget.experiments


def start_search(run_id: str, searcher: str, seed: int, model: str | None = None,
                 experiments: int | None = None) -> SearchState:
    """Open the log and evaluate the start config as experiment 0 (outside the experiment budget).
    experiments overrides agent.yaml's experiment budget (e.g. a short test run)."""
    settings = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    budget_cfg = {**settings["budget"], **({"experiments": experiments} if experiments else {})}
    rule = settings["accept_rule"]
    budget = Budget(budget_cfg["experiments"], budget_cfg["wall_minutes"], budget_cfg["usd"])
    log = RunLog(run_id)
    log.write("start", searcher=searcher, seed=seed, model=model, tier=TIER, budget=budget_cfg,
              accept_rule=rule, start_params=START_PARAMS)
    result = evaluate(to_overrides(START_PARAMS), TIER)
    state = SearchState(log, budget, rule["min_rel_improvement"], rule["min_month_wins"], dict(START_PARAMS), result)
    _record(state, 0, "start config: GEFCom's plain model", None, {}, dict(START_PARAMS), result, True, "start config")
    return state


def describe_task(state: SearchState) -> dict[str, Any]:
    return {
        "problem": "Month-ahead probabilistic forecast of hourly electricity load for one US utility "
                   "(GEFCom2014 load track): for every hour of the coming month, predict quantiles of the load.",
        "known_at_forecast_time": "Load and temperature up to the end of the previous month, and the calendar "
                                  "(hour, weekday, holidays). The temperature of the forecast month itself is not known.",
        "model": "One LightGBM quantile regression per quantile level, on calendar and temperature features.",
        "metric": "Pinball loss over 9 quantile levels and all hours of a month, averaged over the 7 development "
                  "months Oct 2010 - Apr 2011. Lower is better. Also reported: coverage of the 5%-95% interval "
                  "(target 0.90).",
        "rules": f"Each experiment changes parameters of the current best config. It becomes the new best only if "
                 f"its dev loss is more than {state.min_rel_improvement:.0%} lower than the best's and it is lower in "
                 f"at least {state.min_month_wins} of the 7 months. The start config is experiment 0.",
        "current_best": _summary(state.best_id, state.best_params, state.best),
        "experiments_left": state.experiments_left(),
    }


def get_search_space() -> dict[str, Any]:
    space: dict[str, Any] = {}
    for name, p in SEARCH_SPACE.items():
        entry: dict[str, Any] = {"type": p.kind, "description": p.description}
        if p.kind in ("int", "float"):
            entry["range"] = [p.low, p.high]
            if p.step > 1:
                entry["step"] = p.step
        if p.choices:
            entry["choices"] = list(p.choices)
        if p.nullable:
            entry["null_allowed"] = True
        space[name] = entry
    return space


def run_experiment(state: SearchState, changes: dict[str, Any], hypothesis: str,
                   expected_change_pct: float | None = None) -> dict[str, Any]:
    """Guardrails -> duplicate check -> evaluate -> accept rule -> log. Returns the tool result."""
    if state.finished:
        return {"error": "the search is already finished."}
    stop = state.budget.exhausted()
    if stop:
        return {"error": f"no experiments left ({stop}). Call finish."}
    rejection = check_proposal(changes, state.best_params)
    params = {**state.best_params, **changes} if rejection is None else {}
    key = json.dumps(to_overrides(params), sort_keys=True) if rejection is None else ""
    if rejection is None and key in state.seen:
        rejection = Rejection("duplicate", f"already evaluated as experiment {state.seen[key]} (see get_history).")
    if rejection:
        state.leakage_hits += rejection.category == "leakage"
        state.log.write("rejected", changes=changes, hypothesis=hypothesis, category=rejection.category,
                        reason=rejection.reason)
        return {"rejected": rejection.category, "reason": rejection.reason, "budget_used": False}

    state.budget.experiments += 1
    run_id = state.budget.experiments
    state.seen[key] = run_id  # a config that fails must not be retried either
    started = time.perf_counter()
    try:
        result = evaluate(to_overrides(params), TIER)
    except GefcomRunError as e:
        state.history.append({"id": run_id, "changes": changes, "error": str(e).splitlines()[0],
                              "hypothesis": hypothesis})
        state.log.write("error", id=run_id, changes=changes, hypothesis=hypothesis, message=str(e),
                        runtime_s=time.perf_counter() - started)
        return {"experiment_id": run_id, "error": str(e), "experiments_left": state.experiments_left()}
    previous = state.best
    decision = accept(result, previous, state.min_rel_improvement, state.min_month_wins)
    if decision.accepted:
        state.best, state.best_params, state.best_id = result, params, run_id
    _record(state, run_id, hypothesis, expected_change_pct, changes, params, result, decision.accepted, decision.reason)
    return {
        "experiment_id": run_id,
        "dev_loss": round(result.dev_loss, 3),
        "change_vs_best_pct": round(100 * (result.dev_loss / previous.dev_loss - 1), 2),
        "months_better_than_best": month_wins(result, previous),
        "month_losses": _months(result),
        "coverage_90": round(result.coverage_90, 3),
        "runtime_min": round(result.runtime_s / 60, 1),
        "decision": decision.reason,
        "best": {"experiment_id": state.best_id, "dev_loss": round(state.best.dev_loss, 3)},
        "experiments_left": state.experiments_left(),
    }


def get_history(state: SearchState) -> dict[str, Any]:
    return {"experiments": state.history, "best_id": state.best_id, "experiments_left": state.experiments_left()}


def finish(state: SearchState, best_run_id: int, rationale: str, stop_reason: str = "finish called") -> dict[str, Any]:
    """The accept rule's best is what gets evaluated; the searcher's claim is logged next to it."""
    if state.finished:  # e.g. two finish calls in one turn: log only the first
        return {"finished": True, "note": "the search was already finished."}
    state.finished = True
    state.log.write("finish", best_id=state.best_id, claimed_best_id=best_run_id, rationale=rationale,
                    stop_reason=stop_reason, best_params=state.best_params, best_dev_loss=state.best.dev_loss,
                    experiments=state.budget.experiments, leakage_hits=state.leakage_hits)
    result: dict[str, Any] = {"finished": True, "best": _summary(state.best_id, state.best_params, state.best)}
    if best_run_id != state.best_id:
        result["note"] = f"By the accept rule the best is experiment {state.best_id}; that one will be evaluated."
    return result


def call_tool(state: SearchState, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool_use block from the agent to its handler."""
    if name == "describe_task":
        return describe_task(state)
    if name == "get_search_space":
        return get_search_space()
    if name == "run_experiment":
        return run_experiment(state, tool_input.get("changes", {}), str(tool_input.get("hypothesis", "")),
                              tool_input.get("expected_change_pct"))
    if name == "get_history":
        return get_history(state)
    if name == "finish":
        return finish(state, tool_input.get("best_run_id", -1), str(tool_input.get("rationale", "")))
    return {"error": f"unknown tool {name!r}"}


def _months(result: EvalResult) -> dict[str, float]:
    return {MONTHS[t]: round(loss, 2) for t, loss in result.fold_losses.items()}


def _summary(run_id: int, params: dict[str, Any], result: EvalResult) -> dict[str, Any]:
    return {"experiment_id": run_id, "params": params, "dev_loss": round(result.dev_loss, 3),
            "month_losses": _months(result), "coverage_90": round(result.coverage_90, 3)}


def _record(state: SearchState, run_id: int, hypothesis: str, expected_change_pct: float | None,
            changes: dict[str, Any], params: dict[str, Any], result: EvalResult, accepted: bool, decision: str) -> None:
    state.seen[json.dumps(to_overrides(params), sort_keys=True)] = run_id
    state.history.append({"id": run_id, "changes": changes, "dev_loss": round(result.dev_loss, 3),
                          "coverage_90": round(result.coverage_90, 3), "accepted": accepted, "hypothesis": hypothesis})
    state.log.write("experiment", id=run_id, hypothesis=hypothesis, expected_change_pct=expected_change_pct,
                    changes=changes, params=params, config_hash=result.config_hash, dev_loss=result.dev_loss,
                    coverage_90=result.coverage_90, fold_losses=result.fold_losses, runtime_s=result.runtime_s,
                    cached=result.cached, accepted=accepted, decision=decision, best_id=state.best_id,
                    best_dev_loss=state.best.dev_loss)
