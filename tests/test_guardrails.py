"""Tests for src/guardrails.py: one test per rejection path of check_proposal, plus Budget."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src import guardrails
from src.guardrails import Budget, check_proposal
from src.search_space import START_PARAMS


def _category(changes: dict[str, Any]) -> str | None:
    rejection = check_proposal(changes, START_PARAMS)
    return rejection.category if rejection else None


@pytest.mark.parametrize("changes", [
    {"temperature_source": "actual"},
    {"temperature_source": "ACTUAL"},
    {"leakage": {"temperature_source": "actual"}},
    {"use_oracle": True},
    {"folds": "holdout"},
    {"development_last_test_task": 15},
    {"extended": True},
])
def test_leakage_rejected(changes: dict[str, Any]) -> None:
    assert _category(changes) == "leakage"


def test_unknown_key_lists_allowed_names() -> None:
    rejection = check_proposal({"colsample": 0.5}, START_PARAMS)
    assert rejection and rejection.category == "invalid"
    assert "colsample" in rejection.reason and "max_depth" in rejection.reason


@pytest.mark.parametrize("changes", [
    {"max_depth": 99},
    {"learning_rate": 0.5},
    {"max_depth": 5.0},  # float for an int param
    {"max_depth": True},  # bool for an int param
    {"n_estimators": None},  # not nullable
    {"scenario_combine": "median"},
    {"temperature_dynamics": ["temp_ma24", "bogus"]},
    {"temperature_dynamics": ["temp_ma24", "temp_ma24"]},
    {"temperature_dynamics": [["temp_ma24"]]},  # unhashable item: rejected, not a crash
    {"scenario_window_days": 7},  # off the step-2 grid
])
def test_invalid_values_rejected(changes: dict[str, Any]) -> None:
    assert _category(changes) == "invalid"


@pytest.mark.parametrize("changes", [
    {},
    {"max_depth": 4},
    {"scenario_window_days": 2},  # unused while scenarios are off
    {"temperature_dynamics": []},
])
def test_no_effective_change_rejected(changes: dict[str, Any]) -> None:
    rejection = check_proposal(changes, START_PARAMS)
    assert rejection and rejection.category == "invalid" and "no effective change" in rejection.reason


def test_nullable_params_accept_null() -> None:
    current = {**START_PARAMS, "scenarios_enabled": True, "recency_half_life_years": 3.0, "scenario_max_years": 5}
    assert check_proposal({"recency_half_life_years": None}, current) is None
    assert check_proposal({"scenario_max_years": None}, current) is None


def test_valid_proposal_returns_none() -> None:
    assert check_proposal({"scenarios_enabled": True, "scenario_window_days": 4}, START_PARAMS) is None


# ---- Budget -----------------------------------------------------------------------------------

def _budget(**kwargs: Any) -> Budget:
    return Budget(max_experiments=10, max_wall_minutes=60, max_usd=5.0, started=0.0, **kwargs)


def _set_clock(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(guardrails, "time", SimpleNamespace(monotonic=lambda: seconds))


def test_budget_not_exhausted_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_clock(monkeypatch, 0.0)
    assert _budget().exhausted() is None


def test_budget_experiment_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_clock(monkeypatch, 0.0)
    assert "experiment" in (_budget(experiments=10).exhausted() or "")


def test_budget_token_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_clock(monkeypatch, 0.0)
    assert _budget(usd=4.99).exhausted() is None
    assert "API budget" in (_budget(usd=5.0).exhausted() or "")


def test_budget_wall_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_clock(monkeypatch, 59 * 60)
    assert _budget().exhausted() is None
    _set_clock(monkeypatch, 60 * 60)
    assert "wall-time" in (_budget().exhausted() or "")
