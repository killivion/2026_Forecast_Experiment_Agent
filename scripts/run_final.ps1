# Design v2 final evaluation (NOTES.md, "Final evaluation"): extended folds at 20 quantiles first (primary),
# then the holdout at 99 quantiles (secondary). Configs fixed 2026-11-11 from dev losses only.
# Run from the repo root in your own terminal, lid open, charger in, other apps closed:
#   powershell -File scripts\run_final.ps1
# Estimate: extended ~7 configs x 30-70 min, holdout 3 new configs x ~75 min (human is cached).
# Resumable: every finished config is cached in runs/cache, so a rerun skips it; a step logged with exit 0 is skipped.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$python = ".venv\Scripts\python.exe"
$steps = @(
    ("--folds extended --quantiles 20 --runs agent_s1_20261110-184739 agent_full_s0_20261110-192041 " +
     "no_feedback_s2_20261110-205527 optuna_s0_20261110-210920 random_s1_20261108-184952 " +
     "--cut no_feedback_s2_20261110-205527=agent_s2_20261110-190642 optuna_s0_20261110-210920=agent_s0_20261110-125827 " +
     "random_s1_20261108-184952=agent_s1_20261110-184739 --experiment random_s0_20261108-181049:9"),
    ("--folds holdout --quantiles 99 --runs agent_s1_20261110-184739 optuna_s0_20261110-210920 " +
     "random_s1_20261108-184952")
)
$log = if (Test-Path runs\run_final.log) { @(Get-Content runs\run_final.log) } else { @() }
$started = $null
$done = foreach ($line in $log) {  # progress lines sit between a step's start and exit line
    if ($line -match " start  (.+)$") { $started = $Matches[1] }
    elseif ($line -match " exit 0 " -and $started) { $started }
}
foreach ($step in $steps) {
    if ($done -contains $step) { continue }
    "$(Get-Date -Format s) start  $step" | Tee-Object -Append runs\run_final.log
    # Progress lines (one per config) go to the console and the log as they come.
    & $python -m scripts.final_eval $step.Split(" ") 2>&1 | Tee-Object -Append runs\run_final.log
    "$(Get-Date -Format s) exit $LASTEXITCODE " | Tee-Object -Append runs\run_final.log
}
"$(Get-Date -Format s) all done" | Tee-Object -Append runs\run_final.log
