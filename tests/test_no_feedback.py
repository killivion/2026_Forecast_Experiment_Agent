from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import src.no_feedback as nf
import src.tools as tools
from src.search_space import START_PARAMS, to_overrides
from tests.test_agent import FakeClient, env, events, fake_result, response, tool_use  # noqa: F401 (env: autouse fixture)


def item(**changes: Any) -> dict[str, Any]:
    return {"changes": changes, "hypothesis": "h", "expected_change_pct": -1}


def plan(*items: Any) -> SimpleNamespace:
    return response(tool_use("p", "plan_experiments", experiments=list(items)))


@pytest.fixture
def evaluated(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(tools, "evaluate", lambda o, tier: (seen.append(o), fake_result(o))[1])
    monkeypatch.setattr(nf, "load_dotenv", lambda *a, **k: None)
    return seen


def finish_of(state: tools.SearchState) -> dict[str, Any]:
    return [e for e in events(state) if e["event"] == "finish"][0]


def test_plan_runs_in_order_and_is_logged(evaluated: list[dict[str, Any]]) -> None:
    FakeClient.script = [plan(item(max_depth=5), item(scenarios_enabled=True))]
    state = nf.run(0)

    log = events(state)
    assert [e["event"] for e in log if e["event"] in ("plan", "experiment")] == ["experiment", "plan", "experiment", "experiment"]
    assert len(log[[e["event"] for e in log].index("plan")]["experiments"]) == 2
    assert [e["id"] for e in log if e["event"] == "experiment"] == [0, 1, 2]
    assert [e["searcher"] for e in log if e["event"] == "start"] == ["no_feedback"]
    assert state.best_id == 2 and finish_of(state)["stop_reason"] == "plan used up (2 planned)"
    assert FakeClient.requests[0]["tools"] == [nf.PLAN_TOOL]


def test_changes_are_converted_to_full_config_vs_start(evaluated: list[dict[str, Any]]) -> None:
    # item 1 is accepted (new best has scenarios on); item 2 is planned vs. START and does not repeat that change,
    # so it must run as start + max_depth=5, i.e. scenarios back off.
    FakeClient.script = [plan(item(scenarios_enabled=True), item(max_depth=5))]
    nf.run(0)

    assert len(evaluated) == 3
    assert evaluated[2] == to_overrides({**START_PARAMS, "max_depth": 5})
    assert evaluated[2]["model"]["temperature_scenarios"]["enabled"] is False


def test_budget_caps_the_plan(evaluated: list[dict[str, Any]]) -> None:
    FakeClient.script = [plan(item(max_depth=5), item(max_depth=6), item(max_depth=7))]
    state = nf.run(0, experiments=2)

    assert state.budget.experiments == 2 and len(evaluated) == 3  # start + 2
    assert finish_of(state)["stop_reason"] == state.budget.exhausted() != ""


def test_no_tool_use_finishes_with_reason(evaluated: list[dict[str, Any]]) -> None:
    FakeClient.script = [response(SimpleNamespace(type="text", text="no"), stop="end_turn")]
    state = nf.run(0)

    assert finish_of(state)["stop_reason"] == "model stopped without a plan: end_turn"
    assert state.budget.experiments == 0 and state.best_id == 0


def test_invalid_items_rejected_without_budget(evaluated: list[dict[str, Any]]) -> None:
    FakeClient.script = [plan({"changes": ["max_depth"], "hypothesis": "h", "expected_change_pct": 0},
                              item(no_such_param=1), "garbage", item(max_depth=5))]
    state = nf.run(0)

    rejected = [e for e in events(state) if e["event"] == "rejected"]
    assert len(rejected) >= 2 and {e["category"] for e in rejected} <= {"invalid", "duplicate"}
    assert state.budget.experiments == 1 and len(evaluated) == 2  # only max_depth=5 ran
    assert finish_of(state)["stop_reason"] == "plan used up (4 planned)"
