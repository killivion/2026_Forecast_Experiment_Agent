"""GEFCom2014 adapter: evaluate(overrides, tier) -> EvalResult.

GEFCom is a black box: the overrides are deep-merged into its dev.yaml, the
tier sets the quantile grid (and thins the temperature-scenario shifts in the
fast tier), the YAML is written to runs/scratch/, and run_comparison is called
via subprocess on the development folds only. Results are cached per
(GEFCom commit, tier, merged config), so the same config never runs twice.

Fast and confirm losses use different quantile grids: never compare them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

Tier = Literal["fast", "confirm"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runs" / "cache"
SCRATCH_DIR = REPO_ROOT / "runs" / "scratch"
MODEL_METHOD = "lightgbm_full"
TIMEOUT_S = 30 * 60

# The fast tier must keep 0.05 and 0.95: GEFCom's coverage metric looks them up by value.
FAST_QUANTILES = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
FAST_SHIFT_DAYS = [-6, -4, -2, 0, 2, 4, 6]


@dataclass(frozen=True)
class EvalResult:
    tier: Tier
    config_hash: str
    dev_loss: float
    dev_loss_std: float
    coverage_90: float
    fold_losses: dict[int, float]  # test_task -> pinball loss
    runtime_s: float
    run_dir: str
    cached: bool


class GefcomRunError(RuntimeError):
    """run_comparison failed or timed out; the message is meant for the agent."""


def _settings() -> dict[str, Any]:
    with open(REPO_ROOT / "configs" / "agent.yaml") as f:
        settings = yaml.safe_load(f)
    for key in ("gefcom_path", "gefcom_python"):
        settings[key] = (REPO_ROOT / settings[key]).resolve()
    return settings


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """New dict: nested dicts merge, anything else (lists, None) replaces."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_config(base: dict[str, Any], overrides: dict[str, Any], tier: Tier) -> dict[str, Any]:
    """base + overrides, then the tier rules. The tier owns the quantile grid;
    in the fast tier, scenario shifts are thinned unless the overrides set them."""
    config = deep_merge(base, overrides)
    if tier == "confirm":
        config["quantiles"] = copy.deepcopy(base["quantiles"])
        return config
    config["quantiles"]["levels"] = list(FAST_QUANTILES)
    scenarios = config["model"].get("temperature_scenarios") or {}
    user_set_shifts = "shift_days" in (overrides.get("model", {}).get("temperature_scenarios") or {})
    if scenarios.get("enabled") and not user_set_shifts:
        scenarios["shift_days"] = list(FAST_SHIFT_DAYS)
    return config


def config_hash(config: dict[str, Any], tier: Tier, gefcom_commit: str) -> str:
    payload = json.dumps({"commit": gefcom_commit, "tier": tier, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _gefcom_commit(gefcom_path: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=gefcom_path, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def _run_gefcom(python: Path, gefcom_path: Path, config_path: Path, output_dir: str, log_path: Path) -> float:
    cmd = [str(python), "-m", "src.evaluation.run_comparison", "--config", str(config_path),
           "--folds", "development", "--baseline", "none", "--skip-stats", "--output-dir", output_dir]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=gefcom_path, capture_output=True, encoding="utf-8",
                              errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise GefcomRunError(f"run_comparison timed out after {TIMEOUT_S} s (log: {log_path})")
    log_path.write_text(f"$ {' '.join(cmd)}\n\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}",
                        encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise GefcomRunError(f"run_comparison exited with code {proc.returncode} (log: {log_path}):\n{tail}")
    return time.perf_counter() - start


def _parse_reports(run_dir: Path) -> tuple[float, float, float, dict[int, float]]:
    summary = pd.read_csv(run_dir / "summary.csv")
    row = summary[(summary["method"] == MODEL_METHOD) & (summary["fold_group"] == "development")].iloc[0]
    by_fold = pd.read_csv(run_dir / "pinball_by_fold.csv", index_col="test_task")[MODEL_METHOD]
    fold_losses = {int(task): float(loss) for task, loss in by_fold.items()}
    return float(row["mean_pinball_loss"]), float(row["std_pinball_loss"]), float(row["mean_coverage_90"]), fold_losses


def evaluate(overrides: dict[str, Any], tier: Tier) -> EvalResult:
    settings = _settings()
    gefcom_path: Path = settings["gefcom_path"]
    with open(gefcom_path / settings["base_config"]) as f:
        base = yaml.safe_load(f)
    config = build_config(base, overrides, tier)
    digest = config_hash(config, tier, _gefcom_commit(gefcom_path))
    name = f"{tier}_{digest}"

    cache_path = CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        stored = json.loads(cache_path.read_text())["result"]
        stored["fold_losses"] = {int(k): v for k, v in stored["fold_losses"].items()}
        return EvalResult(**{**stored, "cached": True})

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    config_path = SCRATCH_DIR / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output_dir = f"agent_{name}"
    runtime_s = _run_gefcom(settings["gefcom_python"], gefcom_path, config_path, output_dir,
                            SCRATCH_DIR / f"{name}.log")

    run_dir = gefcom_path / "reports" / output_dir
    dev_loss, dev_loss_std, coverage_90, fold_losses = _parse_reports(run_dir)
    result = EvalResult(tier=tier, config_hash=digest, dev_loss=dev_loss, dev_loss_std=dev_loss_std,
                        coverage_90=coverage_90, fold_losses=fold_losses, runtime_s=runtime_s,
                        run_dir=str(run_dir), cached=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"result": asdict(result), "config": config}, indent=2))
    return result
