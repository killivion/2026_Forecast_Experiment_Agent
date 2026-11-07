"""Tests for gefcom.evaluate_final and scripts/final_eval.py. The GEFCom subprocess is always mocked;
runs/, results/ and the cache live in tmp_path."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from scripts import final_eval
from src.search_space import START_PARAMS
from src.tasks import gefcom
from src.tasks.gefcom import (FINAL_TIMEOUT_S, REPO_ROOT, TIMEOUT_S, EvalResult, FinalFolds, config_hash,
                              deep_merge, evaluate, evaluate_final)
from tests.test_gefcom import FakeGefcom

# dev.yaml and default.yaml in miniature: identical except for the quantile grid.
DEV_BASE: dict[str, Any] = {
    "quantiles": {"levels": [0.01, 0.05, 0.5, 0.95, 0.99]},
    "leakage": {"temperature_source": "climatology"},
    "model": {
        "params": {"n_estimators": 400, "max_depth": 4, "num_leaves": 16, "learning_rate": 0.05, "random_state": 0},
        "temperature_scenarios": {"enabled": True, "shift_days": list(range(-7, 8)), "combine": "mixture",
                                  "max_years": None},
        "recency_half_life_years": 2,
    },
}
FINAL_BASE: dict[str, Any] = {**DEV_BASE, "quantiles": {"levels": [round(0.01 * i, 2) for i in range(1, 100)]}}

TASKS = {"development": range(2, 9), "holdout": range(9, 16), "extended": range(-31, 2)}
OFFSET = {"development": 0.0, "holdout": 1.0, "extended": 2.0}
COVERAGE = {"development": 0.88, "holdout": 0.87, "extended": 0.86}


class FoldsFake(FakeGefcom):
    """Answers --folds development / holdout / extended with rows of that fold group, next to decoy
    rows (another fold group, another method). Loss = n_estimators / 100 + OFFSET[folds], so every
    result can be traced to its config and folds."""

    def _write_reports(self, output_dir: str) -> None:
        cmd = self.comparison_calls[-1][0]  # FakeGefcom records the call before writing the reports
        folds = cmd[cmd.index("--folds") + 1]
        config = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
        loss = config["model"]["params"]["n_estimators"] / 100 + OFFSET[folds]
        out = self.gefcom_path / "reports" / output_dir
        out.mkdir(parents=True)
        pd.DataFrame({
            "method": ["lightgbm_full", "lightgbm_full", "other"],
            "fold_group": [folds, "all", folds],
            "mean_pinball_loss": [loss, 99.0, 98.0],
            "std_pinball_loss": [0.5, 9.0, 9.0],
            "n_folds": [len(TASKS[folds])] * 3,
            "mean_coverage_90": [COVERAGE[folds], 0.1, 0.2],
        }).to_csv(out / "summary.csv", index=False)
        pd.DataFrame({"test_task": TASKS[folds], "lightgbm_full": loss, "other": 100.0}).to_csv(
            out / "pinball_by_fold.csv", index=False)

    def folds_run(self) -> list[str]:
        return [cmd[cmd.index("--folds") + 1] for cmd, _ in self.comparison_calls]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FoldsFake:
    gefcom_path = tmp_path / "gefcom"
    gefcom_path.mkdir()
    (gefcom_path / "dev.yaml").write_text(yaml.safe_dump(DEV_BASE))
    (gefcom_path / "default.yaml").write_text(yaml.safe_dump(FINAL_BASE))
    settings = {"gefcom_path": gefcom_path, "gefcom_python": tmp_path / "py.exe", "base_config": "dev.yaml",
                "final_config": "default.yaml"}
    monkeypatch.setattr(gefcom, "_settings", lambda: settings)
    monkeypatch.setattr(gefcom, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gefcom, "SCRATCH_DIR", tmp_path / "scratch")
    monkeypatch.setattr(gefcom, "_keep_awake", contextlib.nullcontext)
    monkeypatch.setattr(final_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(final_eval, "SUMMARY_PATH", tmp_path / "results" / "summary.csv")
    runner = FoldsFake(gefcom_path)
    monkeypatch.setattr(gefcom.subprocess, "run", runner)
    return runner


# ---- evaluate_final ---------------------------------------------------------------------------

@pytest.mark.parametrize("folds", ["holdout", "extended"])
def test_evaluate_final_flags_base_grid_and_timeout(fake: FoldsFake, monkeypatch: pytest.MonkeyPatch,
                                                    folds: FinalFolds) -> None:
    keep_awake = MagicMock(side_effect=contextlib.nullcontext)
    monkeypatch.setattr(gefcom, "_keep_awake", keep_awake)
    overrides = {"quantiles": {"levels": [0.2, 0.8]},
                 "model": {"params": {"n_estimators": 300}, "temperature_scenarios": {"shift_days": [-3, -1, 0, 1, 3]}}}
    result = evaluate_final(overrides, folds)

    expected = deep_merge(FINAL_BASE, overrides)
    expected["quantiles"] = FINAL_BASE["quantiles"]  # always default.yaml's grid
    digest = config_hash(expected, f"final-{folds}", "deadbeef")
    cmd, kwargs = fake.comparison_calls[0]
    assert cmd[cmd.index("--folds") + 1] == folds
    assert cmd[cmd.index("--baseline") + 1] == "none" and "--skip-stats" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == f"agent_final-{folds}_{digest}"
    assert kwargs["timeout"] == FINAL_TIMEOUT_S > TIMEOUT_S
    assert keep_awake.call_count == 1
    assert yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text()) == expected  # odd shift days kept

    assert result.tier == "final" and result.config_hash == digest and result.cached is False
    assert (result.dev_loss, result.coverage_90) == (3.0 + OFFSET[folds], COVERAGE[folds])  # not the decoy rows
    assert result.fold_losses == {t: 3.0 + OFFSET[folds] for t in TASKS[folds]}


def test_final_cache_is_separate_from_search_tiers_and_per_folds(fake: FoldsFake) -> None:
    def run_all() -> list[EvalResult]:
        return [evaluate({}, "fast"), evaluate({}, "confirm"), evaluate_final({}, "holdout"),
                evaluate_final({}, "extended")]

    first = run_all()
    assert [r.tier for r in first] == ["fast", "confirm", "final", "final"]
    assert fake.folds_run() == ["development", "development", "holdout", "extended"]
    assert [kwargs["timeout"] for _, kwargs in fake.comparison_calls] == [TIMEOUT_S, TIMEOUT_S, FINAL_TIMEOUT_S,
                                                                          FINAL_TIMEOUT_S]
    labels = sorted(path.stem.split("_")[0] for path in gefcom.CACHE_DIR.glob("*.json"))
    assert labels == ["confirm", "fast", "final-extended", "final-holdout"]

    second = run_all()
    assert len(fake.comparison_calls) == 4 and all(r.cached for r in second)
    # fold_losses come back with int keys, negative ones for the extended folds
    assert [(r.dev_loss, r.fold_losses) for r in second] == [(r.dev_loss, r.fold_losses) for r in first]


# ---- scripts/final_eval.py --------------------------------------------------------------------

# llm_call events as src/agent.py writes them: tokens plus the cost it priced the call at.
CALLS = [
    {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 2_000_000,
     "cache_creation_input_tokens": 400_000, "cost_usd": 4.4},
    {"input_tokens": 500_000, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
     "cost_usd": 1.0},
]


def _search_log(searcher: str, seed: int, n_estimators: int, llm_calls: list[dict[str, Any]],
                finished: bool = True) -> list[dict[str, Any]]:
    """A synthetic log.jsonl, 90.5 min from start to finish; the best config differs only in n_estimators."""
    records: list[dict[str, Any]] = [
        {"time": "2026-11-08T10:00:00", "event": "start", "searcher": searcher, "seed": seed, "model": None,
         "tier": "fast", "start_params": START_PARAMS},
        {"time": "2026-11-08T10:01:00", "event": "experiment", "id": 0, "params": START_PARAMS, "dev_loss": 8.9,
         "runtime_s": 54.0, "accepted": True},
        {"time": "2026-11-08T10:05:00", "event": "experiment", "id": 1, "params": START_PARAMS, "dev_loss": 8.8,
         "runtime_s": 186.0, "accepted": False},
        *({"time": "2026-11-08T10:02:00", "event": "llm_call", **call} for call in llm_calls),
        {"time": "2026-11-08T10:03:00", "event": "rejected", "category": "leakage", "reason": "oracle"},
    ]
    if finished:
        records.append({"time": "2026-11-08T11:30:30", "event": "finish", "best_id": 3, "claimed_best_id": 3,
                        "rationale": "done", "stop_reason": "finish called",
                        "best_params": {**START_PARAMS, "n_estimators": n_estimators},
                        "best_dev_loss": n_estimators / 100 + 0.5, "experiments": 20, "leakage_hits": 2})
    return records


def _write_log(run_id: str, records: list[dict[str, Any]]) -> None:
    path = final_eval.RUNS_DIR / run_id / "log.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_log_row_sums_logged_costs_and_runtimes() -> None:
    row = final_eval.log_row("agent_0", _search_log("agent", 0, 300, CALLS))
    # cost 4.4 + 1.0; compute (54 s + 186 s) / 60 = 4.0 min; wall 10:00:00 -> 11:30:30
    assert row == {"searcher": "agent", "seed": 0, "run_id": "agent_0", "best_id": 3, "dev_loss": 3.5,
                   "experiments": 20, "stop_reason": "finish called", "compute_minutes": 4.0, "wall_minutes": 90.5,
                   "cost_usd": 5.4, "leakage_hits": 2}


def test_compute_minutes_include_failed_runs() -> None:
    records = _search_log("random", 1, 200, [])
    records.insert(-1, {"time": "2026-11-08T11:00:00", "event": "error", "id": 2, "message": "timed out",
                        "runtime_s": 1800.0})
    assert final_eval.log_row("random_1", records)["compute_minutes"] == 34.0  # (54 + 186 + 1800) / 60
    assert final_eval.log_row("random_1", _search_log("random", 1, 200, []))["cost_usd"] == 0


def test_main_merges_holdout_then_extended_into_one_row_per_run(fake: FoldsFake) -> None:
    logs = {"agent_0": _search_log("agent", 0, 300, CALLS), "random_1": _search_log("random", 1, 200, [])}
    for run_id, records in logs.items():
        _write_log(run_id, records)
    runs = ["--runs", "agent_0", "random_1"]

    final_eval.main(["--folds", "holdout", *runs])
    after_holdout = pd.read_csv(final_eval.SUMMARY_PATH)
    assert list(after_holdout.columns) == final_eval.COLUMNS
    assert list(after_holdout["run_id"]) == ["human", "agent_0", "random_1"]
    assert after_holdout["extended_loss"].isna().all()

    final_eval.main(["--folds", "extended", *runs])
    summary = pd.read_csv(final_eval.SUMMARY_PATH).set_index("run_id")
    assert list(summary.index) == ["human", "agent_0", "random_1"]
    # loss = n_estimators / 100 + folds offset: the human row is default.yaml's 400 trees, a run its best_params
    assert summary[["holdout_loss", "extended_loss"]].to_dict("index") == {
        "human": {"holdout_loss": 5.0, "extended_loss": 6.0},
        "agent_0": {"holdout_loss": 4.0, "extended_loss": 5.0},
        "random_1": {"holdout_loss": 3.0, "extended_loss": 4.0}}
    assert (summary["holdout_coverage_90"] == COVERAGE["holdout"]).all()
    assert (summary["extended_coverage_90"] == COVERAGE["extended"]).all()
    # human dev loss = dev.yaml as configured in the fast tier (400 trees), run once and then cached
    assert summary.loc["human", "searcher"] == "human" and summary.loc["human", "dev_loss"] == 4.0
    assert summary.loc["human", ["seed", "best_id", "experiments", "stop_reason", "compute_minutes", "wall_minutes",
                                 "cost_usd", "leakage_hits"]].isna().all()
    assert fake.folds_run() == ["development"] + ["holdout"] * 3 + ["extended"] * 3
    assert {path.stem.split("_")[0] for path in gefcom.CACHE_DIR.glob("*.json")} == {
        "fast", "final-holdout", "final-extended"}

    for run_id, records in logs.items():
        for column, value in final_eval.log_row(run_id, records).items():
            if column != "run_id":
                assert summary.loc[run_id, column] == value, (run_id, column)


def test_runs_option_is_required(fake: FoldsFake) -> None:
    with pytest.raises(SystemExit):
        final_eval.main(["--folds", "holdout"])
    assert fake.comparison_calls == []


def test_runs_option_selects_runs_and_requires_a_finish_event(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 300, CALLS))
    _write_log("random_1", _search_log("random", 1, 200, []))
    _write_log("unfinished", _search_log("agent", 2, 500, CALLS, finished=False))

    final_eval.main(["--folds", "holdout", "--runs", "random_1"])
    assert list(pd.read_csv(final_eval.SUMMARY_PATH)["run_id"]) == ["human", "random_1"]
    with pytest.raises(ValueError, match="unfinished"):
        final_eval.main(["--folds", "holdout", "--runs", "random_1", "unfinished"])
    assert len(fake.comparison_calls) == 3  # human dev + human + random_1; the failing call ran nothing
