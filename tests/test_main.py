"""main.py dispatches to the right searcher with the right arguments (searchers faked)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import main
from src import agent, optuna_search, random_search


def _state() -> SimpleNamespace:
    return SimpleNamespace(log=SimpleNamespace(run_id="r"), best_id=0, best=SimpleNamespace(dev_loss=8.0),
                           budget=SimpleNamespace(experiments=3), leakage_hits=0)


def _record(calls: list[tuple[str, tuple[Any, ...]]], name: str):
    def fake(*args: Any) -> SimpleNamespace:
        calls.append((name, args))
        return _state()
    return fake


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    recorded: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(agent, "run", _record(recorded, "agent"))
    monkeypatch.setattr(random_search, "run", _record(recorded, "random"))
    monkeypatch.setattr(optuna_search, "run", _record(recorded, "optuna"))
    return recorded


def test_dispatches_to_the_chosen_searcher(calls: list[tuple[str, tuple[Any, ...]]]) -> None:
    main.main(["--searcher", "agent", "--seed", "2", "--budget", "3", "--model", "claude-haiku-5-5"])
    main.main(["--searcher", "random", "--seed", "1"])
    main.main(["--searcher", "agent", "--seed", "0", "--variant", "full_budget"])
    assert calls == [("agent", (2, 3, "claude-haiku-5-5", "main")), ("random", (1, None)),
                     ("agent", (0, None, None, "full_budget"))]


def test_variant_is_rejected_for_other_searchers(calls: list[tuple[str, tuple[Any, ...]]]) -> None:
    with pytest.raises(SystemExit):
        main.main(["--searcher", "random", "--variant", "full_budget"])
    assert calls == []


def test_dispatches_to_optuna(calls: list[tuple[str, tuple[Any, ...]]]) -> None:
    main.main(["--searcher", "optuna", "--seed", "4", "--budget", "5"])
    main.main(["--searcher", "optuna"])
    assert calls == [("optuna", (4, 5)), ("optuna", (0, None))]


def test_model_is_rejected_for_optuna(calls: list[tuple[str, tuple[Any, ...]]]) -> None:
    with pytest.raises(SystemExit):
        main.main(["--searcher", "optuna", "--model", "claude-haiku-5-5"])
    assert calls == []


def test_model_is_rejected_for_random_search(calls: list[tuple[str, tuple[Any, ...]]]) -> None:
    with pytest.raises(SystemExit):
        main.main(["--searcher", "random", "--model", "claude-haiku-5-5"])
    assert calls == []
