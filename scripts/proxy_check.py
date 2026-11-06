"""Is the fast tier a usable proxy for the confirm tier?

Runs GEFCom's four ablations (as overrides on dev.yaml) in both tiers and
compares, within each tier only: the ranking by dev loss, and for every pair
of configs how many of the 7 dev months the better one wins (what the accept
rule uses). Losses are never compared across tiers (different quantile grids).

Usage: python -m scripts.proxy_check   (~40 min; run nothing else in parallel)
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from src.tasks.gefcom import EvalResult, Tier, evaluate

ABLATIONS: dict[str, dict[str, Any]] = {
    "plain": {"model": {"temperature_scenarios": {"enabled": False}, "recency_half_life_years": None}},
    "recency_only": {"model": {"temperature_scenarios": {"enabled": False}}},
    "scenarios_only": {"model": {"recency_half_life_years": None}},
    "final": {},
}


def month_wins(a: EvalResult, b: EvalResult) -> int:
    """Number of dev months where a's loss is lower than b's."""
    return sum(a.fold_losses[t] < b.fold_losses[t] for t in a.fold_losses)


def main() -> None:
    tiers: tuple[Tier, Tier] = ("fast", "confirm")
    results = {tier: {name: evaluate(overrides, tier) for name, overrides in ABLATIONS.items()} for tier in tiers}
    ranking = {tier: sorted(ABLATIONS, key=lambda n: results[tier][n].dev_loss) for tier in tiers}

    print(f"{'config':<16}{'fast loss':>10}{'rank':>6}{'fast min':>10}{'confirm loss':>14}{'rank':>6}{'confirm min':>13}")
    for name in ABLATIONS:
        f, c = results["fast"][name], results["confirm"][name]
        print(f"{name:<16}{f.dev_loss:>10.3f}{ranking['fast'].index(name) + 1:>6}{f.runtime_s / 60:>10.1f}"
              f"{c.dev_loss:>14.3f}{ranking['confirm'].index(name) + 1:>6}{c.runtime_s / 60:>13.1f}")
    print("RANKING MATCH" if ranking["fast"] == ranking["confirm"] else "RANKING MISMATCH")

    print("\nMonths won by the first config (of 7), per tier:")
    for a, b in combinations(ABLATIONS, 2):
        wins = [month_wins(results[tier][a], results[tier][b]) for tier in tiers]
        flag = "" if (wins[0] >= 4) == (wins[1] >= 4) else "  <- 4-of-7 verdict differs"
        print(f"  {a} vs {b}: fast {wins[0]}, confirm {wins[1]}{flag}")


if __name__ == "__main__":
    main()
