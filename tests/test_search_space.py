"""Tests for src/search_space.py: to_overrides, START_PARAMS consistency, accept."""
from __future__ import annotations

import json
from typing import Any

import pytest

from src.guardrails import _check_value
from src.search_space import SEARCH_SPACE, START_PARAMS, accept, to_overrides
from src.tasks.gefcom import EvalResult


def _result(dev_loss: float, losses: list[float], tier: str = "fast") -> EvalResult:
    return EvalResult(tier, "h", dev_loss, 0.0, 0.9, dict(zip(range(2, 9), losses)), 0.0, "d", False)  # type: ignore[arg-type]


def _scenarios(**params: Any) -> dict[str, Any]:
    return to_overrides({**START_PARAMS, **params})["model"]["temperature_scenarios"]


# ---- to_overrides -----------------------------------------------------------------------------

def test_to_overrides_start_params() -> None:
    overrides = to_overrides(START_PARAMS)
    model = overrides["model"]
    assert model["params"]["num_leaves"] == 2 ** START_PARAMS["max_depth"]
    assert model["temperature_scenarios"]["shift_days"] == list(range(-6, 7))
    assert model["temperature_scenarios"]["enabled"] is False
    assert model["recency_half_life_years"] is None
    assert model["temperature_dynamics"] == []
    assert overrides["leakage"] == {"temperature_source": "climatology"}


def test_to_overrides_window_edges() -> None:
    assert _scenarios(scenarios_enabled=True, scenario_window_days=0)["shift_days"] == [0]
    assert _scenarios(scenarios_enabled=True, scenario_window_days=4)["shift_days"] == [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def _same_overrides(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same JSON, i.e. the same GEFCom config hash."""
    return json.dumps(to_overrides(a), sort_keys=True) == json.dumps(to_overrides(b), sort_keys=True)


def test_to_overrides_is_canonical() -> None:
    on = {**START_PARAMS, "scenarios_enabled": True}
    # scenario settings are unused while scenarios are off ...
    assert _same_overrides({**START_PARAMS, "scenario_window_days": 2, "scenario_combine": "quantile_average",
                            "scenario_max_years": 3}, START_PARAMS)
    # ... and temperature_source while they are on
    assert _same_overrides({**on, "temperature_source": "last_year"}, on)
    assert not _same_overrides({**on, "scenario_window_days": 2}, on)
    # feature order and int-vs-float don't create a new config
    assert _same_overrides({**START_PARAMS, "temperature_dynamics": ["temp_ma72", "temp_ma24"]},
                           {**START_PARAMS, "temperature_dynamics": ["temp_ma24", "temp_ma72"]})
    assert _same_overrides({**START_PARAMS, "recency_half_life_years": 2}, {**START_PARAMS, "recency_half_life_years": 2.0})


# ---- START_PARAMS vs SEARCH_SPACE -------------------------------------------------------------

def test_start_params_match_search_space() -> None:
    assert set(START_PARAMS) == set(SEARCH_SPACE)
    for name, value in START_PARAMS.items():
        assert _check_value(SEARCH_SPACE[name], value) is None, name


# ---- accept -----------------------------------------------------------------------------------

BEST = _result(10.0, [10.0] * 7)


def test_accept_needs_improvement_and_wins() -> None:
    decision = accept(_result(9.0, [9.0] * 7), BEST, 0.05, 4)
    assert decision.accepted and decision.reason.startswith("accepted") and "7/7" in decision.reason
    assert "(10.0% lower;" in decision.reason


def test_reason_says_higher_for_a_worse_candidate() -> None:
    decision = accept(_result(11.0, [11.0] * 7), BEST, 0.05, 4)  # read by the agent: no "-10.0% lower"
    assert not decision.accepted and "(10.0% higher;" in decision.reason and "0/7" in decision.reason


def test_accept_rejects_improvement_exactly_at_threshold() -> None:
    decision = accept(_result(9.5, [9.5] * 7), BEST, 0.05, 4)  # (10 - 9.5) / 10 == 0.05
    assert not decision.accepted and decision.reason.startswith("rejected")


def test_accept_rejects_too_few_wins_despite_large_improvement() -> None:
    decision = accept(_result(5.0, [1.0] * 3 + [20.0] * 4), BEST, 0.05, 4)
    assert not decision.accepted and decision.reason.startswith("rejected") and "3/7" in decision.reason


def test_accept_rejects_mismatched_tiers() -> None:
    with pytest.raises(ValueError):
        accept(_result(9.0, [9.0] * 7, tier="confirm"), BEST, 0.05, 4)
