"""Phase 5: score the best config of every finished search once, on folds no searcher saw.

For each search named with --runs (its runs/<run_id>/log.jsonl must have a finish
event), the accept rule's best config (finish event) goes through gefcom.evaluate_final
on the holdout or the extended folds, at 20 or 99 quantiles (Design v2: extended at
20 is the primary test, holdout at 99 the secondary). A "human" row adds GEFCom's
final model exactly as configured. Rows are merged into results/summary.csv by
run_id, so a later call fills its own columns (e.g. extended20_loss) and keeps the
others. Every scored config's per-month losses go to results/months.csv, for the
paired month-level test against the human row.

dev_loss is the fast-tier loss the search saw (9 quantiles; human row:
evaluate({}, "fast")); the final columns use their own grid, so compare losses
within a column, e.g. as improvement over the human row.
compute_minutes = sum of the experiments' GEFCom runtimes (a cached result
counts with its original runtime, so the order of the searches doesn't
matter); wall_minutes = finish - start (includes LLM latency, and cache hits
from earlier searches shorten it). cost_usd = sum of the logged per-call costs
of the LLM searchers.

--cut SEARCH_RUN=AGENT_RUN adds the equal-compute comparison: SEARCH_RUN cut at
AGENT_RUN's compute (cut_search), as a row of its own ("<searcher>_cut").
--experiment RUN_ID:ID adds one experiment's config as a row ("<searcher>_experiment"),
e.g. the lowest dev loss the accept rule rejected.

Usage: python -m scripts.final_eval --folds holdout|extended --quantiles 20|99 --runs RUN_ID [RUN_ID ...]
       [--cut SEARCH_RUN=AGENT_RUN ...] [--experiment RUN_ID:ID ...]
(per config ~75 min on the holdout at 99 quantiles [measured: human row]; extended at 20 quantiles
roughly the same [estimate: 5x the months, a fifth of the quantiles]; run nothing else in parallel)
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
from src.tasks.gefcom import REPO_ROOT, FinalFolds, FinalQuantiles, evaluate, evaluate_final

SUMMARY_PATH = REPO_ROOT / "results" / "summary.csv"
MONTHS_PATH = REPO_ROOT / "results" / "months.csv"
FINAL_COLUMNS = [f"{folds}{q}_{metric}" for folds in get_args(FinalFolds) for q in get_args(FinalQuantiles)
                 for metric in ("loss", "coverage_90")]
COLUMNS = ["searcher", "seed", "run_id", "best_id", "dev_loss", *FINAL_COLUMNS, "experiments", "stop_reason",
           "compute_minutes", "wall_minutes", "cost_usd", "leakage_hits"]


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
    compute = compute_minutes(records)
    cost = sum(r["cost_usd"] for r in records if r["event"] == "llm_call")
    return {"searcher": start["searcher"], "seed": start["seed"], "run_id": run_id, "best_id": finish["best_id"],
            "dev_loss": finish["best_dev_loss"], "experiments": finish["experiments"],
            "stop_reason": finish["stop_reason"],
            "compute_minutes": round(compute, 1), "wall_minutes": round(wall, 1), "cost_usd": round(cost, 2),
            "leakage_hits": finish["leakage_hits"]}


def cut_search(run_id: str, records: list[dict[str, Any]], limit_minutes: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """A search as if it had stopped at limit_minutes of compute (NOTES.md, cut-off rule): its summary
    columns and the accept rule's best params among the experiments that fit, in order."""
    spent, done = 0.0, []
    for record in (r for r in records if r["event"] in ("experiment", "error")):
        if spent + record["runtime_s"] / 60 > limit_minutes:
            break
        spent += record["runtime_s"] / 60
        done.append(record)
    experiments = [r for r in done if r["event"] == "experiment"]
    last = experiments[-1]
    params = next(r["params"] for r in experiments if r["id"] == last["best_id"])
    cost = sum(r["cost_usd"] for r in records if r["event"] == "llm_call")  # an LLM plan is paid before any cut
    row = {"searcher": f"{_event(records, 'start')['searcher']}_cut", "seed": _event(records, "start")["seed"],
           "run_id": f"{run_id}_cut", "best_id": last["best_id"], "dev_loss": last["best_dev_loss"],
           "experiments": len(done) - 1,  # the start config is outside the budget
           "stop_reason": f"cut at {limit_minutes:.1f} min of compute", "compute_minutes": round(spent, 1),
           "cost_usd": round(cost, 2), "leakage_hits": 0}
    return row, params


def experiment_row(run_id: str, records: list[dict[str, Any]], experiment_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One experiment of a search as a row of its own, with its params."""
    start = _event(records, "start")
    record = next(r for r in records if r["event"] == "experiment" and r["id"] == experiment_id)
    row = {"searcher": f"{start['searcher']}_experiment", "seed": start["seed"], "run_id": f"{run_id}_e{experiment_id}",
           "best_id": experiment_id, "dev_loss": record["dev_loss"],
           "stop_reason": f"experiment {experiment_id}, accepted: {record['accepted']}"}
    return row, record["params"]


def compute_minutes(records: list[dict[str, Any]]) -> float:
    return sum(r["runtime_s"] for r in records if r["event"] in ("experiment", "error")) / 60


def final_columns(name: str, overrides: dict[str, Any], folds: FinalFolds, quantiles: FinalQuantiles,
                  months: list[dict[str, Any]]) -> dict[str, float]:
    """evaluate_final, printed as progress (a run can take hours), as summary columns; appends the
    per-month losses to months."""
    result = evaluate_final(overrides, folds, quantiles)
    took = "cached" if result.cached else f"{result.runtime_s / 60:.0f} min"
    print(f"{name}: {folds} ({quantiles} quantiles) loss {result.dev_loss:.3f}, coverage_90 {result.coverage_90:.3f}"
          f" ({took})", flush=True)
    months += [{"run_id": name, "folds": folds, "quantiles": quantiles, "test_task": task, "loss": loss}
               for task, loss in sorted(result.fold_losses.items())]
    return {f"{folds}{quantiles}_loss": result.dev_loss, f"{folds}{quantiles}_coverage_90": result.coverage_90}


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


def merge_months(path: Path, months: list[dict[str, Any]]) -> None:
    """Long format, one row per (run_id, folds, quantiles, test_task); new rows replace old ones with that key."""
    new = pd.DataFrame(months)
    key = ["run_id", "folds", "quantiles", "test_task"]
    old = pd.read_csv(path) if path.exists() else new.iloc[:0]
    merged = pd.concat([old, new]).drop_duplicates(subset=key, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score each finished search's best config once on unseen folds.")
    parser.add_argument("--folds", choices=get_args(FinalFolds), required=True)
    parser.add_argument("--quantiles", type=int, choices=get_args(FinalQuantiles), required=True)
    # Required: runs/ also holds test runs (--budget 3) and early-stopped ones, and each final run takes hours.
    parser.add_argument("--runs", nargs="+", metavar="RUN_ID", required=True, help="the searches to score")
    parser.add_argument("--cut", nargs="+", metavar="SEARCH_RUN=AGENT_RUN", default=[],
                        help="also score SEARCH_RUN cut at AGENT_RUN's compute (the equal-compute comparison)")
    parser.add_argument("--experiment", nargs="+", metavar="RUN_ID:ID", default=[],
                        help="also score one experiment's config of a search")
    args = parser.parse_args(argv)
    # Everything that reads logs runs first, so a typo fails before hours of runs.
    runs = finished_runs(args.runs)
    log_rows = {run_id: log_row(run_id, records) for run_id, records in runs.items()}
    pairs = [pair.split("=") for pair in args.cut]
    cut_logs = finished_runs([run_id for pair in pairs for run_id in pair])
    extras = [cut_search(search_run, cut_logs[search_run], compute_minutes(cut_logs[agent_run]))
              for search_run, agent_run in pairs]
    for spec in args.experiment:
        run_id, experiment_id = spec.rsplit(":", 1)
        extras.append(experiment_row(run_id, finished_runs([run_id])[run_id], int(experiment_id)))

    folds, quantiles, months = args.folds, args.quantiles, []
    # Human first: on the holdout at 99 quantiles it must reproduce GEFCom's 6.79 before hours go into the rest.
    rows = [{"searcher": "human", "run_id": "human", "dev_loss": evaluate({}, "fast").dev_loss,
             **final_columns("human", {}, folds, quantiles, months)}]
    for run_id, records in runs.items():
        best_params = _event(records, "finish")["best_params"]
        rows.append({**log_rows[run_id], **final_columns(run_id, to_overrides(best_params), folds, quantiles, months)})
    for row, params in extras:
        rows.append({**row, **final_columns(row["run_id"], to_overrides(params), folds, quantiles, months)})
    merge_summary(SUMMARY_PATH, rows)
    merge_months(MONTHS_PATH, months)
    print(pd.read_csv(SUMMARY_PATH).dropna(axis=1, how="all").to_string(index=False))


def _event(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(r for r in records if r["event"] == name)


if __name__ == "__main__":
    main()
