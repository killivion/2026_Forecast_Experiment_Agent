"""No-feedback control: the agent's model plans the whole search up front, without seeing any result.

It gets the same information the agent reads before its first experiment
(describe_task and get_search_space, including the start config's losses) and
returns up to the experiment budget of configs, most promising first, in one
call. The harness then runs them in that order through tools.run_experiment,
like random search: same guardrails, duplicate check, accept rule, budget and
log. Agent vs. this control measures what the loop adds beyond the model's
prior knowledge (NOTES.md, research questions).

Usage: python main.py --searcher no_feedback --seed 0
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import anthropic
import yaml
from anthropic.types import ToolParam
from dotenv import load_dotenv

from src import tools
from src.agent import account
from src.search_space import START_PARAMS
from src.tasks.gefcom import REPO_ROOT

SYSTEM = (
    "You are an ML engineer improving a month-ahead probabilistic load forecasting model. You will not see the "
    "result of any experiment: plan the whole search now, in one call to plan_experiments. The experiments run "
    "in your order; each becomes the new best only if the accept rule takes it, and the search stops when the "
    "budget is used up, so put the most promising configs first. Unlike in the task rules below, each experiment "
    "is a complete config, given as changes to the START config (experiment 0): it does not inherit earlier "
    "experiments, so to build on an idea, repeat its changes."
)
PLAN_TOOL: ToolParam = {
    "name": "plan_experiments",
    "description": "Submit the planned experiments, most promising first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "description": "One entry per experiment, in the order to run them.",
                "items": {
                    "type": "object",
                    "properties": {
                        "changes": {"type": "object",
                                    "description": "Parameter name -> value, relative to the START config "
                                                   "(experiment 0). Unlisted parameters keep their start value."},
                        "hypothesis": {"type": "string", "description": "Why this config should help."},
                        "expected_change_pct": {"type": "number",
                                                "description": "Predicted dev loss change vs. the start config, "
                                                               "in percent (negative = better)."},
                    },
                    "required": ["changes", "hypothesis", "expected_change_pct"],
                },
            },
        },
        "required": ["experiments"],
    },
}


def run(seed: int, experiments: int | None = None, model: str | None = None) -> tools.SearchState:
    """seed only labels the run, as for the agent."""
    load_dotenv(REPO_ROOT / ".env")
    settings = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    model = model or settings["model"]
    state = tools.start_search(f"no_feedback_s{seed}_{datetime.now():%Y%m%d-%H%M%S}", "no_feedback", seed, model,
                               experiments)
    prompt = (f"Plan up to {state.experiments_left()} experiments.\n\nTask:\n{json.dumps(tools.describe_task(state))}"
              f"\n\nSearch space:\n{json.dumps(tools.get_search_space())}")
    client = anthropic.Anthropic(max_retries=5)
    try:
        with client.messages.stream(
            model=model, max_tokens=64000, system=SYSTEM, tools=[PLAN_TOOL],
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": settings["effort"]},
        ) as stream:
            response = stream.get_final_message()
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
        tools.finish(state, state.best_id, "ended by the harness", stop_reason=f"API error after retries: {e}")
        return state
    account(state, response, settings["price_per_mtok"][model])
    # A plan cut off at max_tokens may be incomplete JSON, so it doesn't run.
    plan = (next((b.input for b in response.content if b.type == "tool_use"), None)
            if response.stop_reason != "max_tokens" else None)
    planned: list[Any] = plan.get("experiments", []) if isinstance(plan, dict) else []
    state.log.write("plan", experiments=planned)

    for item in planned:
        if state.budget.exhausted():
            break
        item = item if isinstance(item, dict) else {}
        changes = item.get("changes", {})
        if isinstance(changes, dict):  # planned against the start config; run_experiment wants changes vs. the best
            params = {**START_PARAMS, **changes}
            # Unknown names stay in, so the guardrails reject them (and count "actual" as leakage).
            changes = {k: v for k, v in params.items() if k not in state.best_params or v != state.best_params[k]}
        tools.run_experiment(state, changes, str(item.get("hypothesis", "")), item.get("expected_change_pct"))
    stop_reason = state.budget.exhausted() or (f"plan used up ({len(planned)} planned)" if plan is not None
                                               else f"model stopped without a plan: {response.stop_reason}")
    tools.finish(state, state.best_id, "no-feedback control: best by the accept rule", stop_reason=stop_reason)
    return state
