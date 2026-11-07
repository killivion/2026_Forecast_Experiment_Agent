"""Code-side checks before every experiment, and the hard budget.

check_proposal returns a Rejection (shown to the searcher as the tool result)
or None. A rejected proposal is not run and uses no experiment budget; the
caller logs it. "leakage" rejections are what the report counts as blocked
leakage attempts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.search_space import SEARCH_SPACE, Param, to_overrides

# Scanned in the whole proposal (keys and values, also nested), so a leak can't
# hide behind an unknown or nested key.
ORACLE = re.compile(r'"actual"|oracle', re.IGNORECASE)
EVALUATION = re.compile(r"fold|holdout|extended|backtest|test_task", re.IGNORECASE)


@dataclass(frozen=True)
class Rejection:
    category: Literal["leakage", "invalid", "duplicate"]  # "duplicate" is set by tools.run_experiment
    reason: str


def check_proposal(changes: dict[str, Any], current: dict[str, Any]) -> Rejection | None:
    """changes: the searcher's proposed parameter values; current: the full
    parameter set of the current best config."""
    if not isinstance(changes, dict):  # LLM JSON can be anything
        return Rejection("invalid", "changes must be an object mapping parameter names to values.")
    text = json.dumps(changes)
    if ORACLE.search(text):
        return Rejection("leakage", "the observed ('actual') temperature of the forecast month is not known a month "
                                    "ahead (oracle leak). temperature_source allows: climatology, last_year.")
    if EVALUATION.search(text):
        return Rejection("leakage", "the evaluation is fixed: experiments run on the 7 development folds only; the "
                                    "holdout and extended folds are reserved for the final evaluation.")
    unknown = sorted(k for k in changes if k not in SEARCH_SPACE)
    if unknown:
        return Rejection("invalid", f"unknown parameter(s) {unknown}. Allowed: {sorted(SEARCH_SPACE)}.")
    for name, value in changes.items():
        problem = _check_value(SEARCH_SPACE[name], value)
        if problem:
            return Rejection("invalid", f"{name}: {problem}")
    if to_overrides({**current, **changes}) == to_overrides(current):
        return Rejection("invalid", "no effective change: this gives the same model as the current best "
                                    "(scenario_* settings are unused while scenarios are off, temperature_source "
                                    "while they are on).")
    return None


def _check_value(param: Param, value: Any) -> str | None:
    """Problem with one proposed value, or None. Values come from LLM JSON, so they are checked at runtime."""
    if value is None:
        return None if param.nullable else "null is not allowed"
    if param.kind == "bool":
        return None if isinstance(value, bool) else f"must be true or false, got {value!r}"
    if param.kind == "choice":
        return None if value in param.choices else f"must be one of {list(param.choices)}, got {value!r}"
    if param.kind == "subset":
        # Membership first: set() would raise on unhashable items such as nested lists.
        valid = isinstance(value, list) and all(v in param.choices for v in value) and len(set(value)) == len(value)
        return None if valid else f"must be a list of distinct values from {list(param.choices)}, got {value!r}"
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if param.kind == "int" and not isinstance(value, int):
        is_number = False
    off_step = param.kind == "int" and is_number and (value - param.low) % param.step != 0
    if not is_number or not param.low <= value <= param.high or off_step:
        kind = "an integer" if param.kind == "int" else "a number"
        steps = f" in steps of {param.step}" if param.step > 1 else ""
        return f"must be {kind} in [{param.low:g}, {param.high:g}]{steps}, got {value!r}"
    return None


@dataclass
class Budget:
    """Hard limits for one search. The caller increments experiments (every
    proposal that passed the guardrails) and usd (agent only: API spend; a token
    cap would mostly count cache reads, which cost a tenth)."""
    max_experiments: int
    max_wall_minutes: float
    max_usd: float
    experiments: int = 0
    usd: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def exhausted(self) -> str | None:
        """Why the search must stop now, or None."""
        if self.experiments >= self.max_experiments:
            return f"experiment budget used ({self.experiments}/{self.max_experiments})"
        minutes = (time.monotonic() - self.started) / 60
        if minutes >= self.max_wall_minutes:
            return f"wall-time budget used ({minutes:.0f}/{self.max_wall_minutes:g} min)"
        if self.usd >= self.max_usd:
            return f"API budget used (${self.usd:.2f}/${self.max_usd:.2f})"
        return None
