"""Run one search on the GEFCom2014 development folds.

python main.py --searcher agent|no_feedback|random --seed 0 [--budget 20] [--model claude-haiku-5-5]
               [--variant main|full_budget|actual_offered]
"""
from __future__ import annotations

import argparse
from typing import get_args

from src import agent, no_feedback, random_search


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--searcher", choices=["agent", "no_feedback", "random"], required=True)
    parser.add_argument("--seed", type=int, default=0, help="random: fixes the samples; LLM searchers: run label only")
    parser.add_argument("--budget", type=int, help="experiments (default: configs/agent.yaml)")
    parser.add_argument("--model", help="LLM searchers only, e.g. claude-haiku-5-5 for a cheap test run")
    parser.add_argument("--variant", choices=get_args(agent.Variant), default="main", help="agent only")
    args = parser.parse_args(argv)
    if args.searcher == "random" and args.model:
        parser.error("--model only applies to the LLM searchers")
    if args.searcher != "agent" and args.variant != "main":
        parser.error("--variant only applies to --searcher agent")
    if args.searcher == "agent":
        state = agent.run(args.seed, args.budget, args.model, args.variant)
    elif args.searcher == "no_feedback":
        state = no_feedback.run(args.seed, args.budget, args.model)
    else:
        state = random_search.run(args.seed, args.budget)
    print(f"{state.log.run_id}: best experiment {state.best_id}, dev loss {state.best.dev_loss:.3f}, "
          f"{state.budget.experiments} experiments, leakage attempts blocked: {state.leakage_hits}")


if __name__ == "__main__":
    main()
