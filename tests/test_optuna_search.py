"""Tests for src/optuna_search.py. GEFCom never runs: tools.evaluate is a fake and the run logs go to tmp_path."""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path
from typing import Any

import optuna
import pytest
from optuna.trial import TrialState

from src import logging_utils, optuna_search, tools
from src.guardrails import check_proposal
from src.logging_utils import read_log
from src.random_search import sample_params
from src.search_space import START_PARAMS
from src.tasks.gefcom import EvalResult, GefcomRunError

N = 6


# ---- encode / decode --------------------------------------------------------------------------

def test_start_params_round_trip() -> None:
    values = optuna_search.encode(START_PARAMS)
    assert set(values) == set(optuna_search.distributions())
    assert optuna_search.decode(values) == START_PARAMS


def test_random_samples_round_trip_and_stay_valid() -> None:
    rng = random.Random(0)
    for _ in range(300):
        params = sample_params(rng)
        assert optuna_search.decode(optuna_search.encode(params)) == params
        assert check_proposal(params, START_PARAMS) is None


def test_null_param_gets_its_lower_bound_slot_and_decodes_back_to_none() -> None:
    nullable = [n for n, p in optuna_search.SEARCH_SPACE.items() if p.nullable]
    params = dict(START_PARAMS, **{nullable[0]: None})
    values = optuna_search.encode(params)
    assert values[f"{nullable[0]}__null"] is True
    assert optuna_search.decode(values)[nullable[0]] is None


# ---- run (GEFCom faked) -----------------------------------------------------------------------

class Env:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on: set[int] = set()  # call indexes (0 = start config) that raise
        self.studies: list[optuna.Study] = []


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Env:
    e = Env()

    def fake_evaluate(overrides: dict[str, Any], tier: str) -> EvalResult:
        i = len(e.calls)
        e.calls.append(overrides)
        if i in e.fail_on:
            raise GefcomRunError("boom\ndetails")
        loss = 5 + (zlib.crc32(json.dumps(overrides, sort_keys=True).encode()) % 1000) / 100  # config-determined
        return EvalResult("fast", f"h{i}", loss, 0.0, 0.9, dict.fromkeys(range(2, 9), loss), 60.0, "d", False)

    real_create = optuna.create_study

    def spy_create(*args: Any, **kwargs: Any) -> optuna.Study:
        study = real_create(*args, **kwargs)
        e.studies.append(study)
        return study

    monkeypatch.setattr(tools, "evaluate", fake_evaluate)
    monkeypatch.setattr(logging_utils, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(optuna, "create_study", spy_create)
    return e


def test_run_uses_exactly_the_budget_and_logs_optuna(env: Env) -> None:
    state = optuna_search.run(seed=1, experiments=N)
    log = read_log(state.log.path)
    assert (log[0]["event"], log[0]["searcher"], log[0]["seed"]) == ("start", "optuna", 1)
    assert len(env.calls) == 1 + N and state.budget.experiments == N
    finish = log[-1]
    assert finish["event"] == "finish" and finish["stop_reason"] == state.budget.exhausted()
    assert finish["best_id"] == state.best_id
    assert len(env.studies[0].trials) >= 1 + N  # trial 0 + every evaluated config (+ told repeats)


def test_same_seed_same_configs_different_seed_differs(env: Env) -> None:
    def configs(seed: int) -> list[dict[str, Any]]:  # evaluated overrides (run ids only differ by the second)
        start = len(env.calls)
        optuna_search.run(seed=seed, experiments=N)
        return env.calls[start:]

    a, b, c = configs(4), configs(4), configs(5)
    assert len(a) == 1 + N
    assert a == b
    assert a != c


def test_start_config_is_trial_zero_with_its_loss(env: Env) -> None:
    state = optuna_search.run(seed=0, experiments=2)
    t0 = env.studies[0].trials[0]
    assert t0.state == TrialState.COMPLETE
    assert t0.params == optuna_search.encode(START_PARAMS)
    assert t0.value == pytest.approx(read_log(state.log.path)[1]["dev_loss"])


def test_repeat_of_a_known_config_is_told_from_known_loss_without_budget(
        env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    real_decode = optuna_search.decode
    n = {"calls": 0}

    def decode(values: dict[str, Any]) -> dict[str, Any]:
        n["calls"] += 1
        return dict(START_PARAMS) if n["calls"] == 1 else real_decode(values)

    monkeypatch.setattr(optuna_search, "decode", decode)
    state = optuna_search.run(seed=0, experiments=3)
    trials = env.studies[0].trials
    assert trials[1].state == TrialState.COMPLETE and trials[1].value == trials[0].value
    assert len(env.calls) == 1 + 3 and state.budget.experiments == 3  # the repeat cost nothing
    assert any(e["event"] == "rejected" for e in read_log(state.log.path))


def test_failed_gefcom_run_is_told_as_fail_and_search_continues(env: Env) -> None:
    env.fail_on = {1}  # the first proposed config
    state = optuna_search.run(seed=2, experiments=3)
    log = read_log(state.log.path)
    assert [e["event"] for e in log].count("error") == 1
    assert env.studies[0].trials[1].state == TrialState.FAIL
    assert len(env.calls) == 1 + 3 and state.budget.experiments == 3  # the failure used budget
    assert log[-1]["event"] == "finish"


def test_stops_after_too_many_rejected_proposals(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optuna_search, "decode", lambda values: dict(START_PARAMS))
    state = optuna_search.run(seed=0, experiments=3)
    log = read_log(state.log.path)
    assert sum(e["event"] == "rejected" for e in log) == optuna_search.MAX_REJECTED_IN_A_ROW
    assert len(env.calls) == 1 and state.budget.experiments == 0
    assert (log[-1]["event"], log[-1]["stop_reason"]) == ("finish", "too many rejected proposals")
