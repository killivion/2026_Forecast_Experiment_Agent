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
    monkeypatch.setattr(final_eval, "MONTHS_PATH", tmp_path / "results" / "months.csv")
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


def test_evaluate_final_20_uses_dev_yaml_grid_and_its_own_label(fake: FoldsFake) -> None:
    overrides = {"quantiles": {"levels": [0.2, 0.8]}, "model": {"params": {"n_estimators": 300}}}
    result = evaluate_final(overrides, "extended", quantiles=20)
    expected = deep_merge(DEV_BASE, overrides)
    expected["quantiles"] = DEV_BASE["quantiles"]
    digest = config_hash(expected, "final20-extended", "deadbeef")
    cmd, kwargs = fake.comparison_calls[0]
    assert cmd[cmd.index("--output-dir") + 1] == f"agent_final20-extended_{digest}"
    assert yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text()) == expected
    assert kwargs["timeout"] == FINAL_TIMEOUT_S
    assert (result.tier, result.config_hash, result.dev_loss) == ("final", digest, 5.0)


def test_final_20_and_99_caches_are_separate(fake: FoldsFake) -> None:
    r99, r20 = evaluate_final({}, "holdout"), evaluate_final({}, "holdout", quantiles=20)
    assert len(fake.comparison_calls) == 2 and r99.config_hash != r20.config_hash
    assert sorted(p.stem.split("_")[0] for p in gefcom.CACHE_DIR.glob("*.json")) == ["final-holdout", "final20-holdout"]
    assert evaluate_final({}, "holdout", quantiles=20).cached and evaluate_final({}, "holdout").cached
    assert len(fake.comparison_calls) == 2


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


def test_main_merges_holdout99_then_extended20_into_one_row_per_run(fake: FoldsFake) -> None:
    logs = {"agent_0": _search_log("agent", 0, 300, CALLS), "random_1": _search_log("random", 1, 200, [])}
    for run_id, records in logs.items():
        _write_log(run_id, records)
    runs = ["--runs", "agent_0", "random_1"]

    final_eval.main(["--folds", "holdout", "--quantiles", "99", *runs])
    after_holdout = pd.read_csv(final_eval.SUMMARY_PATH)
    assert list(after_holdout.columns) == final_eval.COLUMNS
    assert list(after_holdout["run_id"]) == ["human", "agent_0", "random_1"]
    assert after_holdout["holdout99_loss"].notna().all() and after_holdout["extended20_loss"].isna().all()

    final_eval.main(["--folds", "extended", "--quantiles", "20", *runs])
    summary = pd.read_csv(final_eval.SUMMARY_PATH).set_index("run_id")
    assert list(summary.index) == ["human", "agent_0", "random_1"]
    # loss = n_estimators / 100 + folds offset: the human row is the base config's 400 trees, a run its best_params
    assert summary[["holdout99_loss", "extended20_loss"]].to_dict("index") == {
        "human": {"holdout99_loss": 5.0, "extended20_loss": 6.0},
        "agent_0": {"holdout99_loss": 4.0, "extended20_loss": 5.0},
        "random_1": {"holdout99_loss": 3.0, "extended20_loss": 4.0}}
    assert (summary["holdout99_coverage_90"] == COVERAGE["holdout"]).all()
    assert (summary["extended20_coverage_90"] == COVERAGE["extended"]).all()
    unrun = [c for c in final_eval.FINAL_COLUMNS if not c.startswith(("holdout99", "extended20"))]
    assert summary[unrun].isna().all(axis=None)  # the other fold/quantile combinations stay empty
    # human dev loss = dev.yaml as configured in the fast tier (400 trees), run once and then cached
    assert summary.loc["human", "searcher"] == "human" and summary.loc["human", "dev_loss"] == 4.0
    assert summary.loc["human", ["seed", "best_id", "experiments", "stop_reason", "compute_minutes", "wall_minutes",
                                 "cost_usd", "leakage_hits"]].isna().all()
    assert fake.folds_run() == ["development"] + ["holdout"] * 3 + ["extended"] * 3
    assert {path.stem.split("_")[0] for path in gefcom.CACHE_DIR.glob("*.json")} == {
        "fast", "final-holdout", "final20-extended"}

    for run_id, records in logs.items():
        for column, value in final_eval.log_row(run_id, records).items():
            if column != "run_id":
                assert summary.loc[run_id, column] == value, (run_id, column)


def test_months_csv_is_long_format_and_a_rerun_replaces_its_rows(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 300, CALLS))
    final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "agent_0"])
    months = pd.read_csv(final_eval.MONTHS_PATH)
    assert list(months.columns) == ["run_id", "folds", "quantiles", "test_task", "loss"]
    assert len(months) == 2 * len(TASKS["holdout"])  # human + agent_0
    agent = months[months["run_id"] == "agent_0"]
    assert set(agent["folds"]) == {"holdout"} and set(agent["quantiles"]) == {99}
    assert list(agent["test_task"]) == list(TASKS["holdout"]) and (agent["loss"] == 4.0).all()

    # another fold/quantile adds rows; a rerun of the first with a changed best config replaces, not duplicates
    final_eval.main(["--folds", "extended", "--quantiles", "20", "--runs", "agent_0"])
    total = 2 * (len(TASKS["holdout"]) + len(TASKS["extended"]))
    assert len(pd.read_csv(final_eval.MONTHS_PATH)) == total
    (final_eval.RUNS_DIR / "agent_0" / "log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in _search_log("agent", 0, 500, CALLS)), encoding="utf-8")
    final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "agent_0"])
    months = pd.read_csv(final_eval.MONTHS_PATH)
    assert len(months) == total
    holdout = months[months["folds"] == "holdout"].set_index("run_id")["loss"]
    assert (holdout["agent_0"] == 6.0).all() and (holdout["human"] == 5.0).all()  # new replaced old
    kept = months[(months["run_id"] == "agent_0") & (months["folds"] == "extended")]["loss"]
    assert (kept == 5.0).all()  # other keys untouched


def test_main_experiment_adds_a_scored_experiment_row(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 500, CALLS))
    final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "agent_0", "--experiment", "agent_0:1"])
    summary = pd.read_csv(final_eval.SUMMARY_PATH).set_index("run_id")
    assert list(summary.index) == ["human", "agent_0", "agent_0_e1"]
    row = summary.loc["agent_0_e1"]
    assert (row["searcher"], row["seed"], row["best_id"], row["dev_loss"]) == ("agent_experiment", 0, 1, 8.8)
    assert row["stop_reason"] == "experiment 1, accepted: False"
    assert row["holdout99_loss"] == 4.0  # experiment 1's params (300 trees), not the best config's 500
    assert summary.loc["agent_0", "holdout99_loss"] == 6.0
    assert "agent_0_e1" in set(pd.read_csv(final_eval.MONTHS_PATH)["run_id"])


def test_quantiles_option_is_required_and_limited_to_20_or_99(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 300, CALLS))
    for argv in (["--folds", "holdout", "--runs", "agent_0"],
                 ["--folds", "holdout", "--quantiles", "50", "--runs", "agent_0"]):
        with pytest.raises(SystemExit):
            final_eval.main(argv)
    assert fake.comparison_calls == []


def test_runs_option_is_required(fake: FoldsFake) -> None:
    with pytest.raises(SystemExit):
        final_eval.main(["--folds", "holdout", "--quantiles", "99"])
    assert fake.comparison_calls == []


def test_runs_option_selects_runs_and_requires_a_finish_event(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 300, CALLS))
    _write_log("random_1", _search_log("random", 1, 200, []))
    _write_log("unfinished", _search_log("agent", 2, 500, CALLS, finished=False))

    final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "random_1"])
    assert list(pd.read_csv(final_eval.SUMMARY_PATH)["run_id"]) == ["human", "random_1"]
    with pytest.raises(ValueError, match="unfinished"):
        final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "random_1", "unfinished"])
    assert len(fake.comparison_calls) == 3  # human dev + human + random_1; the failing call ran nothing


# ---- cut_search / --cut -----------------------------------------------------------------------

def _cut_log(searcher: str = "random", finish_n_estimators: int = 100) -> list[dict[str, Any]]:
    """Runtimes 600, 600, 600 (error), 600, 100 s; n_estimators = 100 * id. Best by the accept rule: 0, 1, 1, 1, 4
    (the last, cheap experiment is accepted but sits after the expensive one)."""
    def experiment(run_id: int, runtime_s: float, best_id: int) -> dict[str, Any]:
        return {"time": "2026-11-08T10:00:00", "event": "experiment", "id": run_id, "runtime_s": runtime_s,
                "params": {**START_PARAMS, "n_estimators": 100 * run_id}, "dev_loss": 9.0 - run_id,
                "best_id": best_id, "best_dev_loss": 9.0 - best_id}
    return [
        {"time": "2026-11-08T10:00:00", "event": "start", "searcher": searcher, "seed": 7, "start_params": START_PARAMS},
        experiment(0, 600, 0), experiment(1, 600, 1),
        {"time": "2026-11-08T10:00:00", "event": "error", "id": 2, "message": "timed out", "runtime_s": 600.0},
        experiment(3, 600, 1), experiment(4, 100, 4),
        {"time": "2026-11-08T11:00:00", "event": "finish", "best_id": 4, "claimed_best_id": 4, "rationale": "",
         "stop_reason": "budget", "best_params": {**START_PARAMS, "n_estimators": finish_n_estimators},
         "best_dev_loss": 5.0, "experiments": 4, "leakage_hits": 0},
    ]


def test_cut_search_counts_start_config_and_errors_and_stops_on_exact_limit() -> None:
    # 30 min = start 10 + experiment 1 10 + error 10: lands exactly on the sum, so the error still fits
    row, params = final_eval.cut_search("random_7", _cut_log(), 30.0)
    assert row == {"searcher": "random_cut", "seed": 7, "run_id": "random_7_cut", "best_id": 1, "dev_loss": 8.0,
                   "experiments": 2, "stop_reason": "cut at 30.0 min of compute", "compute_minutes": 30.0,
                   "cost_usd": 0, "leakage_hits": 0}  # start config excluded from "experiments", error included
    assert params["n_estimators"] == 100

    # just under: the error no longer fits
    row, _ = final_eval.cut_search("random_7", _cut_log(), 29.9)
    assert (row["experiments"], row["compute_minutes"]) == (1, 20.0)


def test_cut_search_does_not_pick_up_cheaper_experiment_after_the_cut() -> None:
    # experiment 3 (10 min) would exceed 35 min; experiment 4 (1.7 min) would fit but comes after the cut
    row, params = final_eval.cut_search("random_7", _cut_log(), 35.0)
    assert (row["best_id"], row["experiments"], row["compute_minutes"]) == (1, 2, 30.0)
    assert params["n_estimators"] == 100


def test_cut_search_returns_the_best_at_the_cut_not_the_last_experiment() -> None:
    # experiment 3 fits (40 min) but was rejected: the best stays experiment 1
    row, params = final_eval.cut_search("random_7", _cut_log(), 40.0)
    assert (row["best_id"], row["dev_loss"], row["experiments"]) == (1, 8.0, 3)
    assert params["n_estimators"] == 100
    # with room for everything the last experiment is accepted
    row, params = final_eval.cut_search("random_7", _cut_log(), 100.0)
    assert (row["best_id"], row["experiments"], params["n_estimators"]) == (4, 4, 400)


def test_cut_search_with_room_only_for_the_start_config() -> None:
    row, params = final_eval.cut_search("random_7", _cut_log(), 10.0)
    assert (row["best_id"], row["experiments"], params["n_estimators"]) == (0, 0, 0)


def test_main_cut_adds_a_scored_cut_row_next_to_the_normal_rows(fake: FoldsFake) -> None:
    _write_log("agent_0", _search_log("agent", 0, 500, CALLS))  # compute 4.0 min = 240 s
    records = _cut_log(finish_n_estimators=150)
    for record in records:  # random runtimes 100, 140, 60 s: the cut at 240 s keeps the first two
        if record["event"] in ("experiment", "error"):
            record["runtime_s"] = {0: 100.0, 1: 140.0, 2: 60.0, 3: 60.0, 4: 60.0}[record["id"]]
    _write_log("random_7", records)

    final_eval.main(["--folds", "holdout", "--quantiles", "99", "--runs", "agent_0", "random_7",
                     "--cut", "random_7=agent_0"])
    summary = pd.read_csv(final_eval.SUMMARY_PATH).set_index("run_id")
    assert list(summary.index) == ["human", "agent_0", "random_7", "random_7_cut"]
    cut = summary.loc["random_7_cut"]
    assert (cut["searcher"], cut["seed"], cut["best_id"], cut["experiments"]) == ("random_cut", 7, 1, 1)
    assert (cut["compute_minutes"], cut["dev_loss"], cut["cost_usd"]) == (4.0, 8.0, 0)
    # scored with the cut's best params (100 trees: 1.0 + holdout offset), not the finished search's (150)
    assert cut["holdout99_loss"] == 2.0 and cut["holdout99_coverage_90"] == COVERAGE["holdout"]
    assert summary.loc["random_7", "holdout99_loss"] == 2.5 and summary.loc["random_7", "experiments"] == 4
    assert summary.loc["agent_0", "holdout99_loss"] == 6.0


def test_cut_search_carries_the_llm_cost_for_any_searcher() -> None:
    records = _cut_log(searcher="agent")
    records[1:1] = [{"time": "2026-11-08T10:00:00", "event": "llm_call", **call} for call in CALLS]
    row, _ = final_eval.cut_search("agent_7", records, 30.0)
    assert (row["searcher"], row["run_id"], row["cost_usd"]) == ("agent_cut", "agent_7_cut", 5.4)
