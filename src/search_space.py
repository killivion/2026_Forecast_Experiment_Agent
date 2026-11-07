"""What a searcher may change, and when a result counts as a new best.

Searchers (the agent and random search) work with flat, named parameters;
to_overrides maps them onto GEFCom's nested config. Anything not listed in
SEARCH_SPACE cannot be expressed at all: the quantile grid, the folds and the
data stay fixed. Ranges and the accept rule are justified in NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.tasks.gefcom import EvalResult

Kind = Literal["int", "float", "bool", "choice", "subset"]


@dataclass(frozen=True)
class Param:
    kind: Kind
    description: str
    low: float = 0.0  # int / float bounds, inclusive
    high: float = 0.0
    step: int = 1  # int only: allowed values are low, low + step, ...
    log: bool = False  # sample on a log scale (random search)
    choices: tuple[str, ...] = ()  # choice / subset
    nullable: bool = False  # null allowed (= off / use all)


SEARCH_SPACE: dict[str, Param] = {
    "n_estimators": Param("int", "Boosting rounds per quantile model. Runtime grows linearly.", 100, 600, log=True),
    "learning_rate": Param("float", "Shrinkage per boosting round.", 0.02, 0.2, log=True),
    "max_depth": Param("int", "Tree depth; num_leaves is set to 2**max_depth.", 3, 7),
    "min_child_samples": Param("int", "Minimum training rows per leaf.", 10, 300, log=True),
    "recency_half_life_years": Param(
        "float", "Training rows weighted 0.5**(age_years / half_life); null = equal weights. "
                 "Makes a run about 3x slower.", 1, 8, log=True, nullable=True),
    "scenarios_enabled": Param(
        "bool", "Predict the month under temperature scenarios (observed temperatures of earlier years) "
                "instead of a single point temperature input."),
    "scenario_window_days": Param(
        "int", "Scenario shifts of -N..+N days around the same date in each earlier year (N even).", 0, 14, step=2),
    "scenario_combine": Param(
        "choice", "How the scenario forecasts are combined.", choices=("mixture", "quantile_average")),
    "scenario_max_years": Param(
        "int", "Use only the N most recent earlier years as scenarios; null = all (9-10).", 1, 9, nullable=True),
    "temperature_source": Param(
        "choice", "Point temperature input, used only when scenarios are off.", choices=("climatology", "last_year")),
    "temperature_dynamics": Param(
        "subset", "Extra features of the temperature path (moving averages, lag, daily max/min).",
        choices=("temp_ma24", "temp_lag24", "temp_ma72", "temp_day_max", "temp_day_min")),
}

# GEFCom's "plain" ablation: no scenarios, no recency weighting, dev.yaml's tree settings.
START_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_samples": 20,  # LightGBM's default, made explicit
    "recency_half_life_years": None,
    "scenarios_enabled": False,
    "scenario_window_days": 6,  # dev.yaml has 7; in the fast tier both give the shifts [-6, -4, ..., 6]
    "scenario_combine": "mixture",
    "scenario_max_years": None,
    "temperature_source": "climatology",
    "temperature_dynamics": [],
}


def to_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Flat search-space parameters -> nested overrides for gefcom.evaluate.

    Canonical, so two parameter sets give the same overrides (and cache entry)
    exactly when they give the same model: settings the model doesn't use are
    written with their start values (scenario_* while scenarios are off,
    temperature_source while they are on), floats as floats, features sorted."""
    p = {name: float(v) if SEARCH_SPACE[name].kind == "float" and v is not None else v for name, v in params.items()}
    unused = [n for n in SEARCH_SPACE if n.startswith("scenario_")] if not p["scenarios_enabled"] else ["temperature_source"]
    p.update({name: START_PARAMS[name] for name in unused})
    window = p["scenario_window_days"]
    return {
        "leakage": {"temperature_source": p["temperature_source"]},
        "model": {
            "params": {
                "n_estimators": p["n_estimators"],
                "learning_rate": p["learning_rate"],
                "max_depth": p["max_depth"],
                "num_leaves": 2 ** p["max_depth"],
                "min_child_samples": p["min_child_samples"],
            },
            "temperature_scenarios": {
                "enabled": p["scenarios_enabled"],
                "shift_days": list(range(-window, window + 1)),
                "combine": p["scenario_combine"],
                "max_years": p["scenario_max_years"],
            },
            "recency_half_life_years": p["recency_half_life_years"],
            "temperature_dynamics": sorted(p["temperature_dynamics"]),
        },
    }


def month_wins(a: EvalResult, b: EvalResult) -> int:
    """Number of dev months in which a's loss is lower than b's."""
    return sum(a.fold_losses[t] < b.fold_losses[t] for t in a.fold_losses)


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str  # shown to the searcher


def accept(candidate: EvalResult, best: EvalResult, min_rel_improvement: float, min_month_wins: int) -> Decision:
    """New best only if the dev loss is more than min_rel_improvement lower
    than the best's AND the candidate wins at least min_month_wins dev months."""
    if candidate.tier != best.tier:
        raise ValueError(f"cannot compare a {candidate.tier} result with a {best.tier} result")
    improvement = (best.dev_loss - candidate.dev_loss) / best.dev_loss
    wins = month_wins(candidate, best)
    accepted = improvement > min_rel_improvement and wins >= min_month_wins
    reason = (f"{'accepted' if accepted else 'rejected'}: dev loss {candidate.dev_loss:.3f} vs best "
              f"{best.dev_loss:.3f} ({improvement:.1%} lower, needs > {min_rel_improvement:.1%}); "
              f"won {wins}/{len(best.fold_losses)} months (needs >= {min_month_wins})")
    return Decision(accepted, reason)
