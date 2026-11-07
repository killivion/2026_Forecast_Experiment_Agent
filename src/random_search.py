"""Random-search baseline: same search space, budget, accept rule and log format as the agent.

Every sample is a full config drawn from SEARCH_SPACE, independent of all
results so far, so the seed alone fixes the sequence of samples. It goes
through tools.run_experiment as changes against the current best, exactly like
an agent proposal: same guardrails, duplicate check, accept rule, budget and
log. Only the choice of the next proposal differs, so agent vs. random search
measures the value of the LLM's decisions.

Usage: python main.py --searcher random --seed 0
"""
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Any

from src import tools
from src.search_space import SEARCH_SPACE, Param

MAX_REJECTED_IN_A_ROW = 100  # rejected samples use no budget; this stops an endless resampling loop


def sample_params(rng: random.Random) -> dict[str, Any]:
    """One full config, every parameter drawn independently: int uniformly on its
    step grid, float uniformly in [low, high] (both log-uniform where Param.log),
    bool and choice uniformly, subset with each choice in or out at p = 0.5, and
    None at p = 0.5 for nullable parameters. Settings the model doesn't use
    (scenario_* while scenarios are off) are drawn too; to_overrides drops them."""
    return {name: _sample(param, rng) for name, param in SEARCH_SPACE.items()}


def _sample(param: Param, rng: random.Random) -> Any:
    if param.nullable and rng.random() < 0.5:
        return None
    if param.kind == "bool":
        return rng.random() < 0.5
    if param.kind == "choice":
        return rng.choice(param.choices)
    if param.kind == "subset":  # every subset equally likely, items in SEARCH_SPACE order
        return [c for c in param.choices if rng.random() < 0.5]
    if param.kind == "float":
        value = _log_uniform(rng, param.low, param.high) if param.log else rng.uniform(param.low, param.high)
        return min(max(value, param.low), param.high)  # float rounding can land just outside the bounds
    # int: the grid low, low + step, ... up to high
    low, high, step = int(param.low), int(param.high), param.step
    if not param.log:
        return rng.randrange(low, high + 1, step)
    index = round((_log_uniform(rng, low, high) - low) / step)  # nearest grid point,
    return low + step * min(index, (high - low) // step)  # capped at the last one <= high


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Log-uniform in [low, high] up to float rounding: exp(log(10)) == 10.000000000000002."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def run(seed: int, experiments: int | None = None) -> tools.SearchState:
    """Sample until a budget is used up (or too many samples in a row were rejected),
    then finish with the accept rule's best. experiments overrides agent.yaml's budget."""
    state = tools.start_search(f"random_s{seed}_{datetime.now():%Y%m%d-%H%M%S}", "random", seed,
                               experiments=experiments)
    rng = random.Random(seed)
    rejected_in_a_row = 0
    while state.budget.exhausted() is None and rejected_in_a_row < MAX_REJECTED_IN_A_ROW:
        params = sample_params(rng)
        # Changes relative to the best, like the agent's: run_experiment merges them back into params.
        changes = {k: v for k, v in params.items() if v != state.best_params[k]}
        result = tools.run_experiment(state, changes, hypothesis="random sample", expected_change_pct=None)
        rejected_in_a_row = rejected_in_a_row + 1 if "rejected" in result else 0  # duplicate or no effective change
    stop_reason = state.budget.exhausted() or "too many rejected samples"
    tools.finish(state, state.best_id, "random search: best by the accept rule", stop_reason=stop_reason)
    return state
