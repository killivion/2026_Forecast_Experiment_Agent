"""Tests for src/tasks/gefcom.py. The GEFCom subprocess is always mocked."""
from __future__ import annotations

import contextlib
import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
import yaml

from src.search_space import month_wins
from src.tasks import gefcom
from src.tasks.gefcom import (FAST_QUANTILES, EvalResult, GefcomRunError, build_config,
                              config_hash, deep_merge, evaluate)

BASE: dict[str, Any] = {
    "quantiles": {"levels": [0.01, 0.5, 0.99]},
    "leakage": {"guard": True},
    "model": {
        "params": {"num_leaves": 31, "learning_rate": 0.05},
        "temperature_scenarios": {"enabled": True, "shift_days": [-10, 0, 10], "combine": "mean", "max_years": 5},
        "recency_half_life_years": 3.0,
    },
}
SCENARIOS_OFF = {"model": {"temperature_scenarios": {"enabled": False}}}


# ---- deep_merge -------------------------------------------------------------------------------

def test_deep_merge_nested_list_none_new_keys_and_no_mutation() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 5, "c": 1}
    overrides = {"a": {"y": [9], "z": 3}, "b": None, "d": {"k": 1}}
    base_copy, overrides_copy = copy.deepcopy(base), copy.deepcopy(overrides)
    merged = deep_merge(base, overrides)
    assert merged == {"a": {"x": 1, "y": [9], "z": 3}, "b": None, "c": 1, "d": {"k": 1}}
    assert base == base_copy and overrides == overrides_copy
    merged["d"]["k"] = 2  # result must not alias the inputs
    assert overrides == overrides_copy


# ---- build_config -----------------------------------------------------------------------------

def _shifts(config: dict[str, Any]) -> list[int]:
    return config["model"]["temperature_scenarios"]["shift_days"]


def test_fast_sets_grid_and_keeps_even_shifts() -> None:
    base = copy.deepcopy(BASE)
    base["model"]["temperature_scenarios"]["shift_days"] = list(range(-7, 8))
    config = build_config(base, {}, "fast")
    assert config["quantiles"]["levels"] == FAST_QUANTILES
    assert _shifts(config) == [-6, -4, -2, 0, 2, 4, 6]
    assert _shifts(base) == list(range(-7, 8))  # base untouched


def test_fast_thins_user_shift_days() -> None:
    def thinned(days: list[int]) -> list[int]:
        overrides = {"model": {"temperature_scenarios": {"shift_days": days}}}
        return _shifts(build_config(BASE, overrides, "fast"))

    assert thinned(list(range(-7, 8))) == [-6, -4, -2, 0, 2, 4, 6]
    assert thinned([-3, -2, -1, 0, 1, 2, 3]) == [-2, 0, 2]


def test_fast_leaves_shifts_when_scenarios_disabled() -> None:
    overrides = {"model": {"temperature_scenarios": {"enabled": False, "shift_days": [-3, -1, 0, 1]}}}
    assert _shifts(build_config(BASE, overrides, "fast")) == [-3, -1, 0, 1]


def test_fast_replaces_user_quantiles() -> None:
    config = build_config(BASE, {"quantiles": {"levels": [0.2, 0.8]}}, "fast")
    assert config["quantiles"]["levels"] == FAST_QUANTILES


def test_confirm_is_merge_with_base_quantiles() -> None:
    overrides = {"quantiles": {"levels": [0.2, 0.8]}, "model": {"params": {"num_leaves": 7},
                 "temperature_scenarios": {"shift_days": [-3, -1, 0, 1]}}}
    expected = deep_merge(BASE, overrides)
    expected["quantiles"] = BASE["quantiles"]
    config = build_config(BASE, overrides, "confirm")
    assert config == expected
    assert _shifts(config) == [-3, -1, 0, 1]  # odd days not thinned in confirm


# ---- config_hash ------------------------------------------------------------------------------

def test_config_hash_sensitivity() -> None:
    config = {"a": 1, "b": {"x": 1, "y": 2}}
    reordered = {"b": {"y": 2, "x": 1}, "a": 1}
    h = config_hash(config, "fast", "abc")
    assert h == config_hash(reordered, "fast", "abc")
    assert h != config_hash(config, "confirm", "abc")
    assert h != config_hash({**config, "a": 2}, "fast", "abc")
    assert h != config_hash(config, "fast", "def")


# ---- evaluate (subprocess mocked) -------------------------------------------------------------

class FakeGefcom:
    """Stands in for subprocess.run: answers git rev-parse and fakes run_comparison."""

    def __init__(self, gefcom_path: Path) -> None:
        self.gefcom_path = gefcom_path
        self.commit = "deadbeef"
        self.returncode = 0
        self.stderr = ""
        self.raise_timeout = False
        self.comparison_calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        if cmd[0] == "git":
            return SimpleNamespace(stdout=self.commit + "\n", stderr="", returncode=0)
        self.comparison_calls.append((cmd, kwargs))
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        if self.returncode == 0:
            self._write_reports(cmd[cmd.index("--output-dir") + 1])
        return SimpleNamespace(stdout="ok", stderr=self.stderr, returncode=self.returncode)

    def _write_reports(self, output_dir: str) -> None:
        out = self.gefcom_path / "reports" / output_dir
        out.mkdir(parents=True)
        pd.DataFrame({
            "method": ["lightgbm_full", "other"],
            "fold_group": ["development", "development"],
            "mean_pinball_loss": [1.5, 9.0],
            "std_pinball_loss": [0.25, 9.0],
            "n_folds": [7, 7],
            "mean_coverage_90": [0.88, 0.5],
        }).to_csv(out / "summary.csv", index=False)
        pd.DataFrame({
            "test_task": range(2, 9),
            "lightgbm_full": [float(i) for i in range(2, 9)],
            "other": [100.0] * 7,
        }).to_csv(out / "pinball_by_fold.csv", index=False)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGefcom:
    gefcom_path = tmp_path / "gefcom"
    gefcom_path.mkdir()
    (gefcom_path / "dev.yaml").write_text(yaml.safe_dump(BASE))
    settings = {"gefcom_path": gefcom_path, "gefcom_python": tmp_path / "py.exe", "base_config": "dev.yaml"}
    monkeypatch.setattr(gefcom, "_settings", lambda: settings)
    monkeypatch.setattr(gefcom, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gefcom, "SCRATCH_DIR", tmp_path / "scratch")
    runner = FakeGefcom(gefcom_path)
    monkeypatch.setattr(gefcom.subprocess, "run", runner)
    monkeypatch.setattr(gefcom, "_keep_awake", contextlib.nullcontext)
    return runner


def test_evaluate_happy_path(fake: FakeGefcom, tmp_path: Path) -> None:
    overrides = {"model": {"params": {"num_leaves": 7}}}
    result = evaluate(overrides, "fast")

    digest = config_hash(build_config(BASE, overrides, "fast"), "fast", "deadbeef")
    cmd, kwargs = fake.comparison_calls[0]
    assert cmd[:3] == [str(tmp_path / "py.exe"), "-m", "src.evaluation.run_comparison"]
    assert cmd[cmd.index("--folds") + 1] == "development"
    assert cmd[cmd.index("--baseline") + 1] == "none"
    assert "--skip-stats" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == f"agent_fast_{digest}"
    assert kwargs["cwd"] == tmp_path / "gefcom"

    scratch_yaml = Path(cmd[cmd.index("--config") + 1])
    assert scratch_yaml.parent == tmp_path / "scratch"
    assert yaml.safe_load(scratch_yaml.read_text()) == build_config(BASE, overrides, "fast")

    assert result.tier == "fast" and result.config_hash == digest
    assert (result.dev_loss, result.dev_loss_std, result.coverage_90) == (1.5, 0.25, 0.88)
    assert result.fold_losses == {i: float(i) for i in range(2, 9)}
    assert all(isinstance(k, int) for k in result.fold_losses)
    assert result.run_dir == str(tmp_path / "gefcom" / "reports" / f"agent_fast_{digest}")
    assert result.runtime_s >= 0 and result.cached is False


def test_cache_hit_and_other_tier_reruns(fake: FakeGefcom) -> None:
    first = evaluate({}, "fast")
    second = evaluate({}, "fast")
    assert len(fake.comparison_calls) == 1
    assert second.cached is True
    assert (second.dev_loss, second.fold_losses) == (first.dev_loss, first.fold_losses)
    assert all(isinstance(k, int) for k in second.fold_losses)
    evaluate({}, "confirm")
    assert len(fake.comparison_calls) == 2


def test_nonzero_exit_raises_and_is_not_cached(fake: FakeGefcom) -> None:
    fake.returncode, fake.stderr = 1, "noise\n" + "\n".join(f"line{i}" for i in range(30)) + "\nBoom: bad param"
    with pytest.raises(GefcomRunError, match="Boom: bad param"):
        evaluate({}, "fast")
    assert not gefcom.CACHE_DIR.exists() or not list(gefcom.CACHE_DIR.glob("*.json"))
    with pytest.raises(GefcomRunError):
        evaluate({}, "fast")
    assert len(fake.comparison_calls) == 2


def test_timeout_raises_and_writes_log(fake: FakeGefcom) -> None:
    fake.raise_timeout = True
    with pytest.raises(GefcomRunError, match="timed out"):
        evaluate({}, "fast")
    logs = list(gefcom.SCRATCH_DIR.glob("*.log"))
    assert len(logs) == 1 and "run_comparison" in logs[0].read_text(encoding="utf-8")


# ---- _keep_awake ------------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
@pytest.mark.parametrize("fails", [False, True])
def test_keep_awake_sets_and_resets_execution_state(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    kernel32 = MagicMock()
    monkeypatch.setattr(gefcom.ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    try:
        with gefcom._keep_awake():
            kernel32.SetThreadExecutionState.assert_called_once_with(
                gefcom.ES_CONTINUOUS | gefcom.ES_SYSTEM_REQUIRED | gefcom.ES_DISPLAY_REQUIRED)
            if fails:
                raise RuntimeError("boom")
    except RuntimeError:
        assert fails
    assert kernel32.SetThreadExecutionState.call_args_list[-1] == call(gefcom.ES_CONTINUOUS)
    assert kernel32.SetThreadExecutionState.call_count == 2


# ---- month_wins -------------------------------------------------------------------------------

def _result(losses: dict[int, float]) -> EvalResult:
    return EvalResult("fast", "h", 0.0, 0.0, 0.9, losses, 0.0, "d", False)


def test_month_wins_counts_strictly_lower() -> None:
    a = _result({2: 1.0, 3: 2.0, 4: 3.0, 5: 5.0})
    b = _result({2: 2.0, 3: 2.0, 4: 1.0, 5: 6.0})  # a wins 2 and 5, ties 3, loses 4
    assert month_wins(a, b) == 2
    assert month_wins(b, a) == 1
