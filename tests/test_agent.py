from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
import yaml

import src.agent as agent
import src.logging_utils as logging_utils
import src.tools as tools
from src.logging_utils import read_log
from src.tasks.gefcom import EvalResult

SETTINGS = yaml.safe_load((agent.REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
SONNET = SETTINGS["model"]
HAIKU = "claude-haiku-5-5"


def fake_result(overrides: dict[str, Any]) -> EvalResult:
    loss = 100.0
    if overrides["model"]["temperature_scenarios"]["enabled"]:
        loss *= 0.90
    folds = {t: loss * (1 + 0.01 * t) for t in range(2, 9)}
    return EvalResult("fast", "h", sum(folds.values()) / 7, 0.0, 0.9, folds, 120.0, "d", False)


def usage(i: int = 1000, o: int = 500, read: int = 2000, created: int = 3000,
          split: tuple[int, int] | None = (1000, 2000)) -> SimpleNamespace:
    cc = SimpleNamespace(ephemeral_5m_input_tokens=split[0], ephemeral_1h_input_tokens=split[1]) if split else None
    return SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=read,
                           cache_creation_input_tokens=created, cache_creation=cc)


def tool_use(id_: str, name: str, **inp: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def response(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage())


class FakeStream:
    def __init__(self, resp: SimpleNamespace) -> None:
        self.resp = resp

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self.resp


class FakeClient:
    """Scripted responses (or exceptions); records every request with messages deep-copied."""
    script: list[Any] = []
    requests: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    @property
    def messages(self) -> "FakeClient":
        return self

    def stream(self, **kwargs: Any) -> FakeStream:
        FakeClient.requests.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        item = FakeClient.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeStream(item)


@pytest.fixture(autouse=True)
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_utils, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(tools, "evaluate", lambda overrides, tier: fake_result(overrides))
    monkeypatch.setattr(agent, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(agent.anthropic, "Anthropic", FakeClient)
    FakeClient.script, FakeClient.requests = [], []


def events(state: tools.SearchState) -> list[dict[str, Any]]:
    return read_log(state.log.path)


RUN = dict(changes={"scenarios_enabled": True}, hypothesis="h", expected_change_pct=-10)


def test_happy_path() -> None:
    r1 = response(SimpleNamespace(type="thinking", thinking="plan"), SimpleNamespace(type="text", text="look"),
                  tool_use("a", "describe_task"), tool_use("b", "get_search_space"))
    r2 = response(tool_use("c", "run_experiment", **RUN))
    r3 = response(tool_use("d", "finish", best_run_id=1, rationale="done"))
    FakeClient.script = [r1, r2, r3]
    state = agent.run(0)

    assert state.finished and state.best_id == 1
    reqs = FakeClient.requests
    assert len(reqs) == 3
    m = reqs[1]["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[1]["content"] == r1.content
    results = m[2]["content"]
    assert [(b["type"], b["tool_use_id"]) for b in results] == [("tool_result", "a"), ("tool_result", "b")]
    assert "problem" in json.loads(results[0]["content"])
    assert len(reqs[2]["messages"]) == 5
    for req in reqs:
        assert req["model"] == SONNET
        assert req["output_config"] == {"effort": SETTINGS["effort"]}
        assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert req["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    log = events(state)
    calls = [e for e in log if e["event"] == "llm_call"]
    assert len(calls) == 3
    assert all(c["cost_usd"] > 0 and "cache_write_5m_tokens" in c and "cache_write_1h_tokens" in c for c in calls)
    assert calls[0]["thinking"] == ["plan"] and calls[0]["text"] == ["look"]
    assert state.budget.usd == pytest.approx(sum(c["cost_usd"] for c in calls))
    assert [e["event"] for e in log].count("finish") == 1
    assert log[-1]["stop_reason"] == "finish called"


def test_budget_used_prompts_finish_then_harness_finish() -> None:
    FakeClient.script = [response(tool_use("a", "run_experiment", **RUN)),
                         response(tool_use("b", "get_history"))]  # ignores the request to finish
    state = agent.run(0, experiments=1, model=HAIKU)

    reqs = FakeClient.requests
    assert [r["model"] for r in reqs] == [HAIKU, HAIKU]
    last_user = reqs[1]["messages"][-1]["content"]
    assert last_user[-1]["type"] == "text" and "Call finish now" in last_user[-1]["text"]
    p = SETTINGS["price_per_mtok"][HAIKU]
    expected = (1000 * p["input"] + 500 * p["output"] + 2000 * p["cache_read"]
                + 1000 * p["cache_write_5m"] + 2000 * p["cache_write_1h"]) / 1e6
    calls = [e for e in events(state) if e["event"] == "llm_call"]
    assert calls[0]["cost_usd"] == pytest.approx(expected)
    finishes = [e for e in events(state) if e["event"] == "finish"]
    assert len(finishes) == 1 and finishes[0]["stop_reason"] == "no finish after the budget ran out"


def test_end_turn_without_tool_use() -> None:
    FakeClient.script = [response(SimpleNamespace(type="text", text="I am done"), stop="end_turn")]
    state = agent.run(0)
    finishes = [e for e in events(state) if e["event"] == "finish"]
    assert len(finishes) == 1 and finishes[0]["stop_reason"] == "model stopped: end_turn"


@pytest.mark.parametrize("make", [
    lambda: anthropic.APIConnectionError(request=httpx.Request("POST", "https://x.test")),
    lambda: anthropic.InternalServerError(
        "boom", response=httpx.Response(500, request=httpx.Request("POST", "https://x.test")), body=None),
])
def test_transient_error_ends_with_finish(make: Any) -> None:
    FakeClient.script = [make()]
    state = agent.run(0)
    finishes = [e for e in events(state) if e["event"] == "finish"]
    assert len(finishes) == 1 and finishes[0]["stop_reason"].startswith("API error after retries")


def test_non_transient_error_propagates() -> None:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://x.test"))
    FakeClient.script = [anthropic.BadRequestError("bad", response=resp, body=None)]
    with pytest.raises(anthropic.BadRequestError):
        agent.run(0)


def test_call_cost_sonnet_by_hand() -> None:
    p = SETTINGS["price_per_mtok"][SONNET]
    t = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 500_000,
         "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}
    # 1M*2 + 0.1M*10 + 0.5M*0.2 = 2 + 1 + 0.1 = 3.1 USD
    assert agent.call_cost(t, p) == pytest.approx(3.1)


def test_call_cost_cache_write_lifetimes_priced_differently() -> None:
    p = SETTINGS["price_per_mtok"][SONNET]
    base = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
            "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}
    five = agent.call_cost({**base, "cache_write_5m_tokens": 1_000_000}, p)
    hour = agent.call_cost({**base, "cache_write_1h_tokens": 1_000_000}, p)
    assert five == pytest.approx(p["cache_write_5m"]) and hour == pytest.approx(p["cache_write_1h"])
    assert hour > five


def test_call_cost_haiku_long_prompt_factor() -> None:
    p = SETTINGS["price_per_mtok"][HAIKU]
    def t(n: int) -> dict[str, int]:
        return {"input_tokens": n, "output_tokens": 0, "cache_read_input_tokens": 0,
                "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}
    assert agent.call_cost(t(100_000), p) == pytest.approx(100_000 * p["input"] / 1e6)  # at the limit: 1x
    assert agent.call_cost(t(100_001), p) == pytest.approx(5 * 100_001 * p["input"] / 1e6)


def test_tokens_without_cache_split_counts_as_1h_writes() -> None:
    t = agent._tokens(usage(created=3000, split=None))
    assert t["cache_write_1h_tokens"] == 3000 and t["cache_write_5m_tokens"] == 0
    t = agent._tokens(usage(split=(1000, 2000)))
    assert (t["cache_write_5m_tokens"], t["cache_write_1h_tokens"]) == (1000, 2000)


def finish_of(state: tools.SearchState) -> dict[str, Any]:
    finishes = [e for e in events(state) if e["event"] == "finish"]
    assert len(finishes) == 1
    return finishes[0]


def test_tool_choice_disables_parallel_calls_on_every_request() -> None:
    FakeClient.script = [response(tool_use("a", "get_history")), response(tool_use("b", "finish", best_run_id=0, rationale="r"))]
    agent.run(0)
    assert len(FakeClient.requests) == 2
    assert all(r["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True} for r in FakeClient.requests)


def test_full_budget_has_no_finish_tool_and_ends_by_harness_with_budget_text() -> None:
    FakeClient.script = [response(tool_use("a", "run_experiment", **RUN))]
    state = agent.run(0, experiments=1, variant="full_budget")

    req = FakeClient.requests[0]
    assert "finish" not in [t["name"] for t in req["tools"]] and len(req["tools"]) == len(tools.TOOLS) - 1
    assert req["system"].endswith(agent.USE_ALL) and agent.CALL_FINISH not in req["system"]
    assert len(FakeClient.requests) == 1  # no extra "call finish" turn
    assert finish_of(state)["stop_reason"] == state.budget.exhausted() != ""
    assert events(state)[0]["searcher"] == "agent_full"


def test_main_keeps_finish_tool_and_call_finish_prompt() -> None:
    FakeClient.script = [response(tool_use("a", "finish", best_run_id=0, rationale="r"))]
    agent.run(0)
    req = FakeClient.requests[0]
    assert "finish" in [t["name"] for t in req["tools"]] and req["system"].endswith(agent.CALL_FINISH)


def test_full_budget_nudges_then_ends_after_max_nudges() -> None:
    FakeClient.script = [response(SimpleNamespace(type="text", text="done"), stop="end_turn")
                         for _ in range(agent.MAX_NUDGES + 1)]
    state = agent.run(0, variant="full_budget")

    reqs = FakeClient.requests
    assert len(reqs) == agent.MAX_NUDGES + 1
    nudge = reqs[1]["messages"][-1]
    assert nudge["role"] == "user" and "Run the next experiment" in nudge["content"]
    assert finish_of(state)["stop_reason"] == "model stopped: end_turn"


def test_actual_offered_lists_actual_but_guardrails_reject_and_count_it() -> None:
    FakeClient.script = [response(tool_use("a", "get_search_space")),
                         response(tool_use("b", "run_experiment", changes={"temperature_source": "actual"},
                                           hypothesis="h", expected_change_pct=-30)),
                         response(tool_use("c", "finish", best_run_id=0, rationale="r"))]
    state = agent.run(0, variant="actual_offered")

    space = json.loads(FakeClient.requests[1]["messages"][-1]["content"][0]["content"])
    assert "actual" in space["temperature_source"]["choices"]
    rejected = json.loads(FakeClient.requests[2]["messages"][-1]["content"][0]["content"])
    assert rejected["rejected"] == "leakage" and rejected["budget_used"] is False
    assert state.leakage_hits == 1 and state.budget.experiments == 0
    log = events(state)
    assert log[0]["searcher"] == "agent_actual" and log[0]["offer_actual"] is True
    assert finish_of(state)["leakage_hits"] == 1
