"""Append-only run log at runs/<run_id>/log.jsonl, shared by the agent, random
search, the final evaluation and the notebook. One JSON object per line, each
with "time" and "event":

  start       searcher, seed, model, tier, budget, accept_rule, start_params
  experiment  id (0 = start config), hypothesis, expected_change_pct, changes,
              params, config_hash, dev_loss, coverage_90, fold_losses, runtime_s,
              cached, accepted, decision, best_id, best_dev_loss
  rejected    changes, hypothesis, category ("leakage" | "invalid" | "duplicate"), reason
  error       id, changes, hypothesis, message, runtime_s   (GEFCom run failed; uses budget)
  llm_call    input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens, cost_usd, stop_reason, text,
              thinking (summaries)                           (agent only)
  finish      best_id, claimed_best_id, rationale, stop_reason, best_params,
              best_dev_loss, experiments, leakage_hits
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


class RunLog:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.path = RUNS_DIR / run_id / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_log(path: Path) -> list[dict[str, Any]]:
    """All records of one log.jsonl, in order. JSON turns int dict keys
    (fold_losses) into strings."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
