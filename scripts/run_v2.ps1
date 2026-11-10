# Design v2 searches (NOTES.md), one after another; GEFCom runs must not overlap (timing, memory).
# Run from the repo root in your own terminal, lid open, charger in:  powershell -File scripts\run_v2.ps1
# Each search writes runs/<run_id>/log.jsonl; this script's progress goes to runs/run_v2.log.
# A failed search is logged and the next one starts.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$python = ".venv\Scripts\python.exe"
$searches = @(
    "--searcher agent --seed 0", "--searcher agent --seed 1", "--searcher agent --seed 2",
    "--searcher agent --variant full_budget --seed 0",
    "--searcher agent --variant actual_offered --seed 0",
    "--searcher no_feedback --seed 0", "--searcher no_feedback --seed 1", "--searcher no_feedback --seed 2",
    "--searcher optuna --seed 0", "--searcher optuna --seed 1", "--searcher optuna --seed 2"
)
# Resumable: a search whose "start" line is directly followed by "exit 0" in the log is skipped.
$log = if (Test-Path runs\run_v2.log) { @(Get-Content runs\run_v2.log) } else { @() }
$done = for ($i = 0; $i -lt $log.Count - 1; $i++) {
    if ($log[$i + 1] -match " exit 0 " -and $log[$i] -match "start  (.+)$") { $Matches[1] }
}
foreach ($search in $searches) {
    if ($done -contains $search) { continue }
    "$(Get-Date -Format s) start  $search" | Tee-Object -Append runs\run_v2.log
    $output = & $python main.py $search.Split(" ") 2>&1 | Out-String
    "$(Get-Date -Format s) exit $LASTEXITCODE  $($output.Trim())" | Tee-Object -Append runs\run_v2.log
}
"$(Get-Date -Format s) all done" | Tee-Object -Append runs\run_v2.log
