"""Phase 5: score the best config of every finished search once, on folds no searcher saw.

For each search named with --runs (its runs/<run_id>/log.jsonl must have a finish
event), the accept rule's best config (finish event) goes through gefcom.evaluate_final:
GEFCom's default.yaml (99 quantiles) on the holdout or the extended folds. A
"human" row adds GEFCom's final model exactly as configured. Rows are merged
into results/summary.csv by run_id, so --folds extended later fills the
extended columns and keeps the holdout ones.

dev_loss is the fast-tier loss the search saw (9 quantiles; human row:
evaluate({}, "fast")), the holdout / extended columns use 99 quantiles, so
compare losses within a column, e.g. as improvement over the human row.
compute_minutes = sum of the experiments' GEFCom runtimes (a cached result
counts with its original runtime, so the order of the searches doesn't
matter); wall_minutes = finish - start (includes LLM latency, and cache hits
from earlier searches shorten it). cost_usd = sum of the agent's logged
per-call costs (src/agent.py prices each call with its model's prices).

Usage: python -m scripts.final_eval --folds holdout|extended --runs RUN_ID [RUN_ID ...]
(per config up to ~30 min on holdout, ~1.5-2.5 h on extended [estimates]; run nothing else in parallel)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

import pandas as pd

from src.logging_utils import RUNS_DIR, read_log
from src.search_space import to_overrides
from src.tasks.gefcom import REPO_ROOT, FinalFolds, evaluate, evaluate_final

SUMMARY_PATH = REPO_ROOT / "results" / "summary.csv"
COLUMNS = ["searcher", "seed", "run_id", "best_id", "dev_loss", "holdout_loss", "holdout_coverage_90",
           "extended_loss", "extended_coverage_90", "experiments", "stop_reason", "compute_minutes", "wall_minutes",
           "cost_usd", "leakage_hits"]


def finished_runs(run_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """run_id -> log records of the given runs; each must have finished."""
    logs = {run_id: read_log(RUNS_DIR / run_id / "log.jsonl") for run_id in run_ids}
    unfinished = [run_id for run_id, records in logs.items() if not any(r["event"] == "finish" for r in records)]
    if unfinished:
        raise ValueError(f"no finish event in the log of {unfinished}")
    return logs


def log_row(run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """The summary columns that come from a search's log."""
    start, finish = _event(records, "start"), _event(records, "finish")
    wall = (datetime.fromisoformat(finish["time"]) - datetime.fromisoformat(start["time"])).total_seconds() / 60
    compute = sum(r["runtime_s"] for r in records if r["event"] in ("experiment", "error")) / 60
    cost = sum(r["cost_usd"] for r in records if r["event"] == "llm_call")
    return {"searcher": start["searcher"], "seed": start["seed"], "run_id": run_id, "best_id": finish["best_id"],
            "dev_loss": finish["best_dev_loss"], "experiments": finish["experiments"],
            "stop_reason": finish["stop_reason"],
            "compute_minutes": round(compute, 1), "wall_minutes": round(wall, 1), "cost_usd": round(cost, 2),
            "leakage_hits": finish["leakage_hits"]}


def final_columns(name: str, overrides: dict[str, Any], folds: FinalFolds) -> dict[str, float]:
    """evaluate_final, printed as progress (a run can take hours), as summary columns."""
    result = evaluate_final(overrides, folds)
    took = "cached" if result.cached else f"{result.runtime_s / 60:.0f} min"
    print(f"{name}: {folds} loss {result.dev_loss:.3f}, coverage_90 {result.coverage_90:.3f} ({took})", flush=True)
    return {f"{folds}_loss": result.dev_loss, f"{folds}_coverage_90": result.coverage_90}


def merge_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """One row per run_id. A new row overwrites the columns it has and keeps the others (the other folds)."""
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            merged = {row["run_id"]: row for row in csv.DictReader(f)}
    for row in rows:
        merged[row["run_id"]] = {**merged.get(row["run_id"], {}), **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score each finished search's best config once on unseen folds.")
    parser.add_argument("--folds", choices=get_args(FinalFolds), required=True)
    # Required: runs/ also holds test runs (--budget 3) and early-stopped ones, and each final run takes 0.5-2.5 h.
    parser.add_argument("--runs", nargs="+", metavar="RUN_ID", required=True, help="the searches to score")
    args = parser.parse_args(argv)
    runs = finished_runs(args.runs)
    log_rows = {run_id: log_row(run_id, records) for run_id, records in runs.items()}  # fail before hours of runs

    # Human first: on the holdout it must reproduce GEFCom's 6.79 before hours go into the searchers' configs.
    rows = [{"searcher": "human", "run_id": "human", "dev_loss": evaluate({}, "fast").dev_loss,
             **final_columns("human", {}, args.folds)}]
    for run_id, records in runs.items():
        best_params = _event(records, "finish")["best_params"]
        rows.append({**log_rows[run_id], **final_columns(run_id, to_overrides(best_params), args.folds)})
    merge_summary(SUMMARY_PATH, rows)
    print(pd.read_csv(SUMMARY_PATH).to_string(index=False))


def _event(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(r for r in records if r["event"] == name)


if __name__ == "__main__":
    main()
