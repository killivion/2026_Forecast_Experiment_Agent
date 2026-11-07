"""Tests for src/random_search.py. GEFCom never runs: tools.evaluate is a scripted
fake and the run logs go to tmp_path."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from src import logging_utils, random_search, tools
from src.guardrails import _check_value, check_proposal
from src.logging_utils import read_log
from src.random_search import sample_params
from src.search_space import SEARCH_SPACE, START_PARAMS, Param
from src.tasks.gefcom import EvalResult

N = 5000  # samples for the distribution checks (deterministic: fixed seed)


@pytest.fixture(scope="module")
def samples() -> list[dict[str, Any]]:
    rng = random.Random(0)
    return [sample_params(rng) for _ in range(N)]


# ---- sample_params ----------------------------------------------------------------------------

def test_samples_are_valid_proposals() -> None:
    for seed in range(1000):
        params = sample_params(random.Random(seed))
        assert list(params) == list(SEARCH_SPACE)
        for name, value in params.items():
            assert _check_value(SEARCH_SPACE[name], value) is None, (seed, name, value)
        assert check_proposal(params, START_PARAMS) is None, seed  # no leakage pattern, a real change


def test_same_seed_same_samples() -> None:
    def draw(seed: int) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        return [sample_params(rng) for _ in range(5)]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


@pytest.mark.parametrize("name", [n for n, p in SEARCH_SPACE.items() if p.log])
def test_log_params_cover_the_range_evenly_on_a_log_scale(samples: list[dict[str, Any]], name: str) -> None:
    p = SEARCH_SPACE[name]
    # position on the log scale: 0 = low, 1 = high
    pos = [math.log(s[name] / p.low) / math.log(p.high / p.low) for s in samples if s[name] is not None]
    assert min(pos) < 0.05 and max(pos) > 0.95
    for q in (0.25, 0.5, 0.75):  # a linear draw would put far fewer samples below the log midpoint
        assert abs(sum(x < q for x in pos) / len(pos) - q) < 0.05, q


@pytest.mark.parametrize("name", [n for n, p in SEARCH_SPACE.items() if p.kind == "int" and not p.log])
def test_linear_int_params_hit_every_grid_value(samples: list[dict[str, Any]], name: str) -> None:
    p = SEARCH_SPACE[name]
    assert {s[name] for s in samples} - {None} == set(range(int(p.low), int(p.high) + 1, p.step))


def test_subset_is_uniform_over_all_subsets_in_space_order(samples: list[dict[str, Any]]) -> None:
    choices = SEARCH_SPACE["temperature_dynamics"].choices
    subsets = [s["temperature_dynamics"] for s in samples]
    assert all(sub == [c for c in choices if c in sub] for sub in subsets)
    counts = Counter(tuple(sub) for sub in subsets)
    assert len(counts) == 2 ** len(choices)  # every subset occurs, from [] to all five
    expected = N / 2 ** len(choices)
    assert all(abs(c - expected) < 0.35 * expected for c in counts.values())


@pytest.mark.parametrize("name", [n for n, p in SEARCH_SPACE.items() if p.nullable])
def test_nullable_params_are_null_half_the_time(samples: list[dict[str, Any]], name: str) -> None:
    assert abs(sum(s[name] is None for s in samples) / N - 0.5) < 0.05


@pytest.mark.parametrize("name", [n for n, p in SEARCH_SPACE.items() if p.kind in ("bool", "choice")])
def test_bool_and_choice_params_are_uniform(samples: list[dict[str, Any]], name: str) -> None:
    p = SEARCH_SPACE[name]
    options = (True, False) if p.kind == "bool" else p.choices
    counts = Counter(s[name] for s in samples)
    assert set(counts) == set(options)
    assert all(abs(counts[o] / N - 1 / len(options)) < 0.05 for o in options)


class _UpperBound(random.Random):
    """uniform() returns its upper bound: the worst case for float rounding."""

    def uniform(self, a: float, b: float) -> float:
        return b


@pytest.mark.parametrize("param", [
    Param("float", "", 1, 10, log=True),       # exp(log(10)) == 10.000000000000002
    Param("int", "", 1, 9, step=3, log=True),  # grid 1, 4, 7: x = 9.000000000000002 rounds to 10
])
def test_log_draws_at_the_upper_bound_stay_valid(param: Param) -> None:
    assert _check_value(param, random_search._sample(param, _UpperBound())) is None


# ---- run (GEFCom faked) -----------------------------------------------------------------------

BUDGET = 5
# Month losses the fake evaluate returns, in call order (call 0 = the start config). Under the
# 1% / 4-of-7 accept rule the best goes 0 -> 1 -> 4, whatever configs were sampled.
SCRIPT = [
    [10.0] * 7,             # 0: start config
    [9.0] * 7,              # 1: 10% better, all 7 months    -> accepted
    [8.95] * 7,             # 2: 0.6% better than 9.0        -> rejected (needs > 1%)
    [5.0] * 3 + [9.5] * 4,  # 3: 16% better, but 3/7 months  -> rejected
    [8.5] * 7,              # 4: 5.6% better, all 7 months   -> accepted
    [9.0] * 7,              # 5: worse                       -> rejected
]


@pytest.fixture
def evaluated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Scripted results, a budget of BUDGET experiments, the 1% / 4-of-7 rule (independent of
    configs/agent.yaml) and logs in tmp_path. Returns the overrides passed to evaluate."""
    calls: list[dict[str, Any]] = []

    def fake_evaluate(overrides: dict[str, Any], tier: str) -> EvalResult:
        losses = SCRIPT[len(calls)]  # IndexError = more evaluations than the budget allows
        calls.append(overrides)
        return EvalResult("fast", f"h{len(calls)}", sum(losses) / 7, 0.0, 0.9, dict(zip(range(2, 9), losses)),
                          60.0, "d", False)

    real_start = tools.start_search

    def small_start(*args: Any, **kwargs: Any) -> tools.SearchState:
        state = real_start(*args, **kwargs)
        state.budget.max_experiments = BUDGET
        state.min_rel_improvement, state.min_month_wins = 0.01, 4
        return state

    monkeypatch.setattr(tools, "evaluate", fake_evaluate)
    monkeypatch.setattr(tools, "start_search", small_start)
    monkeypatch.setattr(logging_utils, "RUNS_DIR", tmp_path)
    return calls


def _every_other_sample_is_the_start_config(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """run_experiment must reject the start config: no effective change while it is the
    best, a duplicate of experiment 0 afterwards. Returns all samples in order."""
    drawn: list[dict[str, Any]] = []

    def sample(rng: random.Random) -> dict[str, Any]:
        params = dict(START_PARAMS) if len(drawn) % 2 == 0 else sample_params(rng)
        drawn.append(params)
        return params

    monkeypatch.setattr(random_search, "sample_params", sample)
    return drawn


def test_run_uses_exactly_the_budget_and_finishes_with_the_accept_rule_best(
        evaluated: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    drawn = _every_other_sample_is_the_start_config(monkeypatch)
    monkeypatch.setattr(random_search, "MAX_REJECTED_IN_A_ROW", 2)  # 5 rejections, never 2 in a row
    state = random_search.run(seed=3)
    log = read_log(state.log.path)
    experiments = [e for e in log if e["event"] == "experiment"]
    rejected = [e for e in log if e["event"] == "rejected"]

    assert state.log.path.parent.parent == tmp_path
    assert re.fullmatch(r"random_s3_\d{8}-\d{6}", state.log.path.parent.name)
    assert (log[0]["event"], log[0]["searcher"], log[0]["seed"]) == ("start", "random", 3)
    assert {e["event"] for e in log} == {"start", "experiment", "rejected", "finish"}
    # exactly BUDGET evaluations after the start config; rejected samples use none of it
    assert len(evaluated) == 1 + BUDGET and state.budget.experiments == BUDGET
    assert [e["id"] for e in experiments] == list(range(BUDGET + 1))
    assert len(rejected) == BUDGET and {e["category"] for e in rejected} == {"invalid", "duplicate"}
    # the seed fixes the samples, and each experiment evaluates exactly its sample
    # although only the changes vs. the best are sent
    rng = random.Random(3)
    assert drawn[1::2] == [sample_params(rng) for _ in range(BUDGET)]
    assert [e["params"] for e in experiments[1:]] == json.loads(json.dumps(drawn[1::2]))
    assert [e["accepted"] for e in experiments] == [True, True, False, False, True, False]

    finish = log[-1]
    assert finish["event"] == "finish"
    assert finish["best_id"] == finish["claimed_best_id"] == state.best_id == 4
    assert finish["best_params"] == experiments[4]["params"]
    assert finish["best_dev_loss"] == pytest.approx(8.5)
    assert finish["experiments"] == BUDGET
    assert "experiment" in finish["stop_reason"] and finish["stop_reason"] == state.budget.exhausted()


def test_run_stops_after_too_many_rejections_in_a_row(evaluated: list[dict[str, Any]],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random_search, "sample_params", lambda rng: dict(START_PARAMS))  # never a change
    state = random_search.run(seed=0)
    log = read_log(state.log.path)
    assert sum(e["event"] == "rejected" for e in log) == random_search.MAX_REJECTED_IN_A_ROW
    assert len(evaluated) == 1 and state.budget.experiments == 0
    finish = log[-1]
    assert (finish["event"], finish["stop_reason"], finish["best_id"]) == ("finish", "too many rejected samples", 0)
