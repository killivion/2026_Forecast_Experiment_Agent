from pathlib import Path

import pytest

import src.logging_utils as logging_utils
from src.logging_utils import RunLog, read_log


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(logging_utils, "RUNS_DIR", tmp_path)
    return tmp_path


def test_write_one_json_object_per_line_in_order(runs_dir: Path) -> None:
    log = RunLog("r1")
    log.write("start", seed=1)
    log.write("experiment", id=1, dev_loss=2.5)
    log.write("finish", best_id=1)
    assert log.path == runs_dir / "r1" / "log.jsonl"
    assert len(log.path.read_text(encoding="utf-8").splitlines()) == 3
    records = read_log(log.path)
    assert [r["event"] for r in records] == ["start", "experiment", "finish"]
    assert all("time" in r for r in records)
    assert records[1]["dev_loss"] == 2.5


def test_int_fold_keys_become_strings(runs_dir: Path) -> None:
    log = RunLog("r2")
    log.write("experiment", fold_losses={2: 1.0, 3: 2.0})
    assert read_log(log.path)[0]["fold_losses"] == {"2": 1.0, "3": 2.0}


def test_reopening_appends_and_blank_lines_are_skipped(runs_dir: Path) -> None:
    RunLog("r3").write("start")
    log = RunLog("r3")
    log.write("finish")
    with log.path.open("a", encoding="utf-8") as f:
        f.write("\n")
    assert [r["event"] for r in read_log(log.path)] == ["start", "finish"]
