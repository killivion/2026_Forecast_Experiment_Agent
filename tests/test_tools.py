import json
from typing import Any

import pytest

import src.logging_utils as logging_utils
import src.tools as tools
from src.logging_utils import read_log
from src.tasks.gefcom import EvalResult, GefcomRunError
from src.tools import TOOLS, SearchState, call_tool, describe_task, finish, run_experiment, start_search


def fake_result(overrides: dict[str, Any]) -> EvalResult:
    """Loss depends on the overrides: scenarios -10%, recency -5%, learning rate 0.1 -0.5%."""
    loss = 100.0
    if overrides["model"]["temperature_scenarios"]["enabled"]:
        loss *= 0.90
    if overrides["model"]["recency_half_life_years"] is not None:
        loss *= 0.95
    if overrides["model"]["params"]["learning_rate"] == 0.1:
        loss *= 0.995
    folds = {t: loss * (1 + 0.01 * t) for t in range(2, 9)}  # uniformly scaled: lower loss wins every month
    return EvalResult("fast", "h", sum(folds.values()) / 7, 0.0, 0.9, folds, 120.0, "d", False)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake_evaluate(overrides: dict[str, Any], tier: str) -> EvalResult:
        seen.append(overrides)
        return fake_result(overrides)

    monkeypatch.setattr(tools, "evaluate", fake_evaluate)
    return seen


@pytest.fixture
def state(tmp_path, calls) -> SearchState:
    return start_search("t", "agent", 0)


@pytest.fixture(autouse=True)
def tmp_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_utils, "RUNS_DIR", tmp_path)


def events(state: SearchState) -> list[dict[str, Any]]:
    return read_log(state.log.path)


def test_start_search_experiment_zero_outside_budget(state: SearchState, calls) -> None:
    assert len(calls) == 1
    assert state.budget.experiments == 0
    assert state.best_id == 0
    assert state.best_params == tools.START_PARAMS
    assert [h["id"] for h in state.history] == [0]
    log = events(state)
    assert [e["event"] for e in log] == ["start", "experiment"]
    assert log[1]["id"] == 0 and log[1]["accepted"] is True


def test_accepted_change_updates_best(state: SearchState) -> None:
    r = run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    assert r["experiment_id"] == 1
    assert state.best_id == 1 and state.best_params["scenarios_enabled"] is True
    assert state.budget.experiments == 1
    assert r["change_vs_best_pct"] == -10.0
    assert r["months_better_than_best"] == 7
    assert list(r["month_losses"]) == ["Oct 2010", "Nov 2010", "Dec 2010", "Jan 2011", "Feb 2011", "Mar 2011", "Apr 2011"]
    assert r["decision"].startswith("accepted")
    assert r["best"]["experiment_id"] == 1
    last = events(state)[-1]
    assert last["event"] == "experiment" and last["accepted"] is True and last["best_id"] == 1


def test_rejected_by_accept_rule_keeps_best_uses_budget(state: SearchState) -> None:
    r = run_experiment(state, {"learning_rate": 0.1}, "h", -1.0)  # only 0.5% better
    assert r["decision"].startswith("rejected")
    assert state.best_id == 0 and state.best_params == tools.START_PARAMS
    assert state.budget.experiments == 1
    assert r["experiments_left"] == state.budget.max_experiments - 1
    assert events(state)[-1]["accepted"] is False


def test_leakage_rejected_no_budget(state: SearchState, calls) -> None:
    n = len(calls)
    r = run_experiment(state, {"temperature_source": "oracle"}, "cheat", -50.0)
    assert r["rejected"] == "leakage"
    assert state.leakage_hits == 1
    assert state.budget.experiments == 0
    assert len(calls) == n
    assert events(state)[-1]["event"] == "rejected"


def test_invalid_and_non_dict_changes_rejected(state: SearchState, calls) -> None:
    n = len(calls)
    assert run_experiment(state, {"no_such_param": 1}, "h", 0.0)["rejected"] == "invalid"
    assert run_experiment(state, ["n_estimators"], "h", 0.0)["rejected"] == "invalid"  # type: ignore[arg-type]
    assert state.budget.experiments == 0 and len(calls) == n


def test_duplicate_of_earlier_experiment_rejected(state: SearchState, calls) -> None:
    run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    n = len(calls)
    r = run_experiment(state, {"scenarios_enabled": False}, "back", 10.0)  # equals the start config
    assert r["rejected"] == "duplicate"
    assert "experiment 0" in r["reason"]
    assert state.budget.experiments == 1 and len(calls) == n


def test_evaluate_error_is_logged_and_uses_budget(state: SearchState, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(overrides: dict[str, Any], tier: str) -> EvalResult:
        raise GefcomRunError("timed out")

    monkeypatch.setattr(tools, "evaluate", boom)
    r = run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    assert r["error"] == "timed out" and r["experiment_id"] == 1
    assert state.budget.experiments == 1 and state.best_id == 0
    last = events(state)[-1]
    assert last["event"] == "error" and last["message"] == "timed out" and last["runtime_s"] >= 0
    assert state.history[-1]["error"] == "timed out"  # visible in get_history
    retry = run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    assert retry["rejected"] == "duplicate" and state.budget.experiments == 1  # a failed config isn't rerun


def test_budget_exhausted_returns_error_without_evaluating(state: SearchState, calls) -> None:
    state.budget.experiments = state.budget.max_experiments
    n = len(calls)
    r = run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    assert "error" in r and "finish" in r["error"].lower()
    assert len(calls) == n


def test_finish_claim_matches_has_no_note(state: SearchState) -> None:
    run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    r = finish(state, 1, "done")
    assert r["finished"] is True and "note" not in r
    last = events(state)[-1]
    assert last["event"] == "finish" and last["best_id"] == 1 and last["claimed_best_id"] == 1


def test_finish_claim_differs_adds_note_and_blocks_further_runs(state: SearchState, calls) -> None:
    run_experiment(state, {"scenarios_enabled": True}, "h", -10.0)
    r = finish(state, 0, "done")
    assert "note" in r
    last = events(state)[-1]
    assert last["best_id"] == 1 and last["claimed_best_id"] == 0
    n = len(calls)
    after = run_experiment(state, {"recency_half_life_years": 3.0}, "h", -5.0)
    assert "error" in after and len(calls) == n
    finish(state, 1, "again")  # a second finish (e.g. two in one turn) writes no second event
    assert [e["event"] for e in events(state)].count("finish") == 1


def test_call_tool_dispatches_all_tools(state: SearchState) -> None:
    assert "problem" in call_tool(state, "describe_task", {})
    assert "n_estimators" in call_tool(state, "get_search_space", {})
    r = call_tool(state, "run_experiment", {"changes": {"scenarios_enabled": True}, "hypothesis": "h",
                                            "expected_change_pct": -10})
    assert r["experiment_id"] == 1
    assert call_tool(state, "get_history", {})["best_id"] == 1
    assert call_tool(state, "finish", {"best_run_id": 1, "rationale": "r"})["finished"] is True
    assert "error" in call_tool(state, "nope", {})


def test_tool_schemas() -> None:
    assert len(TOOLS) == 5
    for t in TOOLS:
        assert t["name"] and t["description"]
        assert t["input_schema"]["type"] == "object"
    run = next(t for t in TOOLS if t["name"] == "run_experiment")
    assert set(run["input_schema"]["required"]) == {"changes", "hypothesis", "expected_change_pct"}
    json.dumps(TOOLS)


def test_describe_task_neutral_and_has_state(state: SearchState) -> None:
    d = describe_task(state)
    # current_best lists the param names (scenarios_enabled, ...) by design; check the prose only
    text = json.dumps({k: v for k, v in d.items() if k != "current_best"}).lower()
    assert "scenario" not in text and "recency" not in text
    assert d["current_best"]["experiment_id"] == 0
    assert d["experiments_left"] == state.budget.max_experiments
