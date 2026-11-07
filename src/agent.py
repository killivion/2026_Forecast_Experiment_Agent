"""The agent: a manual Claude tool-use loop over the five tools in src/tools.py.

The LLM proposes, code decides: every proposal goes through the guardrails,
the evaluation and the accept rule inside tools.run_experiment. The loop ends
when the agent calls finish, ends a turn without a tool call, or a budget runs
out (then it gets one more turn to call finish). Either way the accept rule's
best is the result.

Usage: python main.py --searcher agent --seed 0   (ANTHROPIC_API_KEY in .env)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

import anthropic
import yaml
from anthropic.types import Message, MessageParam, Usage
from dotenv import load_dotenv

from src import tools
from src.tasks.gefcom import REPO_ROOT

SYSTEM = (
    "You are an ML engineer improving a month-ahead probabilistic load forecasting model by running "
    "experiments. Work in a loop: state a hypothesis, test it with run_experiment, read the result "
    "(dev loss, loss per month, interval coverage, runtime) and decide what to try next. Changes are "
    "relative to the current best config, and a change only counts if the accept rule takes it. "
    "Experiments use budget; rejected proposals do not. Call finish when further experiments are "
    "unlikely to beat the best, or when the budget is used up."
)
KICKOFF = "Start the search. Read the task and the search space first."
LONG_PROMPT_TOKENS = 100_000  # above this, Haiku 5.5 switches to its higher rate card


def run(seed: int, experiments: int | None = None, model: str | None = None) -> tools.SearchState:
    """seed only labels the run: the API has no seed and Sonnet 5.5 takes no temperature (NOTES.md).
    experiments and model override agent.yaml, e.g. for a cheap test run with Haiku."""
    load_dotenv(REPO_ROOT / ".env")
    settings = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    model = model or settings["model"]
    prices: dict[str, float] = settings["price_per_mtok"][model]
    state = tools.start_search(f"agent_s{seed}_{datetime.now():%Y%m%d-%H%M%S}", "agent", seed, model, experiments)
    client = anthropic.Anthropic(max_retries=5)  # the SDK retries 429/5xx and connection errors
    messages: list[MessageParam] = [{"role": "user", "content": KICKOFF}]
    stop_reason, final_turn = "", False

    while not state.finished:
        try:
            # Streaming, so a long thinking turn can use a large max_tokens without an HTTP timeout.
            with client.messages.stream(
                model=model, max_tokens=64000, system=SYSTEM, tools=tools.TOOLS, messages=messages,
                thinking={"type": "adaptive", "display": "summarized"},  # summaries go to the log
                output_config={"effort": settings["effort"]},
                # Caches the growing conversation. 1 h, not the default 5 min: a GEFCom run between two
                # calls can take longer than 5 min, and a miss rewrites the whole prefix at 12x a cache read.
                cache_control={"type": "ephemeral", "ttl": "1h"},
            ) as stream:
                response = stream.get_final_message()
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
            stop_reason = f"API error after retries: {e}"  # transient; a 400 or 401 should crash loudly
            break
        _account(state, response, prices)
        messages.append({"role": "assistant", "content": response.content})  # append-only history

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason in ("refusal", "max_tokens"):
            stop_reason = f"model stopped: {response.stop_reason}"
            break
        results: list[Any] = [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": json.dumps(tools.call_tool(state, b.name, cast("dict[str, Any]", b.input)))}
            for b in tool_uses
        ]  # all results of one turn go back in one user message
        if state.finished:
            break
        if final_turn:
            stop_reason = "no finish after the budget ran out"
            break
        exhausted = state.budget.exhausted()
        if exhausted:
            results.append({"type": "text", "text": f"The budget is used up ({exhausted}). Call finish now."})
            final_turn = True
        messages.append({"role": "user", "content": results})

    if not state.finished:
        tools.finish(state, state.best_id, "ended by the harness", stop_reason=stop_reason)
    return state


def call_cost(tokens: dict[str, int], prices: dict[str, float]) -> float:
    """USD for one call. A model with a long_prompt_factor (Haiku 5.5) bills every token at that
    multiple when the prompt (input + cache reads + cache writes) exceeds 100K tokens."""
    prompt = (tokens["input_tokens"] + tokens["cache_read_input_tokens"]
              + tokens["cache_write_5m_tokens"] + tokens["cache_write_1h_tokens"])
    factor = prices.get("long_prompt_factor", 1.0) if prompt > LONG_PROMPT_TOKENS else 1.0
    return factor * (tokens["input_tokens"] * prices["input"] + tokens["output_tokens"] * prices["output"]
                     + tokens["cache_read_input_tokens"] * prices["cache_read"]
                     + tokens["cache_write_5m_tokens"] * prices["cache_write_5m"]
                     + tokens["cache_write_1h_tokens"] * prices["cache_write_1h"]) / 1e6


def _tokens(u: Usage) -> dict[str, int]:
    """input_tokens excludes cached tokens; cache writes are split by lifetime because they are priced differently."""
    split = u.cache_creation
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            "cache_write_5m_tokens": split.ephemeral_5m_input_tokens if split else 0,
            # no split reported: every write is from our 1 h cache_control
            "cache_write_1h_tokens": split.ephemeral_1h_input_tokens if split else u.cache_creation_input_tokens or 0}


def _account(state: tools.SearchState, response: Message, prices: dict[str, float]) -> None:
    """Add the call's cost to the budget and log its tokens, text and thinking summaries."""
    tokens = _tokens(response.usage)
    cost = call_cost(tokens, prices)
    state.budget.usd += cost
    state.log.write("llm_call", **tokens, cost_usd=cost, stop_reason=response.stop_reason,
                    text=[b.text for b in response.content if b.type == "text"],
                    thinking=[b.thinking for b in response.content if b.type == "thinking"])
