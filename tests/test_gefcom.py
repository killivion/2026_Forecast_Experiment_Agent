"""Tests for src/tasks/gefcom.py and scripts/proxy_check.month_wins. The GEFCom subprocess is always mocked."""
from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from scripts.proxy_check import month_wins
from src.tasks import gefcom
from src.tasks.gefcom import (FAST_QUANTILES, FAST_SHIFT_DAYS, EvalResult, GefcomRunError, build_config,
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

def test_fast_sets_grid_and_thins_shifts() -> None:
    config = build_config(BASE, {}, "fast")
    assert config["quantiles"]["levels"] == FAST_QUANTILES
    assert config["model"]["temperature_scenarios"]["shift_days"] == FAST_SHIFT_DAYS
    assert BASE["model"]["temperature_scenarios"]["shift_days"] == [-10, 0, 10]  # base untouched


def test_fast_leaves_shifts_when_scenarios_disabled() -> None:
    config = build_config(BASE, SCENARIOS_OFF, "fast")
    assert config["model"]["temperature_scenarios"]["shift_days"] == [-10, 0, 10]


def test_fast_keeps_user_shift_days() -> None:
    overrides = {"model": {"temperature_scenarios": {"shift_days": [-1, 1]}}}
    assert build_config(BASE, overrides, "fast")["model"]["temperature_scenarios"]["shift_days"] == [-1, 1]


def test_fast_replaces_user_quantiles() -> None:
    config = build_config(BASE, {"quantiles": {"levels": [0.2, 0.8]}}, "fast")
    assert config["quantiles"]["levels"] == FAST_QUANTILES


def test_confirm_is_merge_with_base_quantiles() -> None:
    overrides = {"quantiles": {"levels": [0.2, 0.8]}, "model": {"params": {"num_leaves": 7}}}
    expected = deep_merge(BASE, overrides)
    expected["quantiles"] = BASE["quantiles"]
    config = build_config(BASE, overrides, "confirm")
    assert config == expected
    assert config["model"]["temperature_scenarios"]["shift_days"] == [-10, 0, 10]  # not thinned


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


def test_timeout_raises(fake: FakeGefcom) -> None:
    fake.raise_timeout = True
    with pytest.raises(GefcomRunError, match="timed out"):
        evaluate({}, "fast")


# ---- month_wins -------------------------------------------------------------------------------

def _result(losses: dict[int, float]) -> EvalResult:
    return EvalResult("fast", "h", 0.0, 0.0, 0.9, losses, 0.0, "d", False)


def test_month_wins_counts_strictly_lower() -> None:
    a = _result({2: 1.0, 3: 2.0, 4: 3.0, 5: 5.0})
    b = _result({2: 2.0, 3: 2.0, 4: 1.0, 5: 6.0})  # a wins 2 and 5, ties 3, loses 4
    assert month_wins(a, b) == 2
    assert month_wins(b, a) == 1
