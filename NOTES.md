# Notes

## Architecture
- Separate repo. GEFCom2014 is treated as a black box: the adapter writes a YAML config, calls `python -m src.evaluation.run_comparison` in `../GEFCOM2014` through a subprocess, and parses `reports/<dir>/summary.csv` and `pinball_by_fold.csv`.
- The LLM proposes changes, and Python code decides: the accept rule, the budget and the leakage guards are all in code, not in the prompt.
- The agent loop uses the plain Anthropic SDK with tool use, no framework.

## Decisions
| Decision | Why |
|---|---|
| GEFCom as a subprocess, not a package | The agent can't change the evaluation code, and other tasks only need an `evaluate(config)` adapter |
| Phase 1 = config-space only | Safe and fast. Code-writing comes later (Phase 8) |
| No agent framework | A ~150-line loop I can explain line by line |
| Sonnet as the agent model | Cheap and good at tool use |
| Two tiers: **fast** (9 quantiles, 7 scenario shifts `[-6,-4,-2,0,2,4,6]`) for search, **confirm** (dev.yaml's 20 quantiles) for checking | A full dev run takes ~6.5 min, too slow for 20 experiments × 6 searches. The fast grid keeps 0.05/0.95 because GEFCom's coverage metric looks those levels up by value. The shift grid keeps 0 (the unshifted scenario) |
| The tier owns the quantile grid, and an explicit `shift_days` override wins over thinning | Otherwise an override could break the coverage metric, or the fast tier would silently undo an experiment on the shift window |
| Fast and confirm losses are never compared | Different quantile grids give different loss scales. Same reason the human 8.78 (99 quantiles) is not comparable to dev.yaml's 8.43 (20 quantiles) |
| Cache key = GEFCom git commit + tier + merged config, stored as JSON in `runs/cache/` | The same config never runs twice; a code change in GEFCom invalidates old results automatically. Uncommitted GEFCom edits are **not** detected |
| Failed runs raise `GefcomRunError` and are never cached; 30 min timeout | The agent sees the error text as the tool result; a hung run can't block a search |
| No key validation in the adapter | That is the guardrails' job (Phase 2): the adapter would run `temperature_source: actual` as-is |

## Measurements
- **Dev run (Phase 0):** `configs/dev.yaml` with 20 quantiles and 7 development folds takes about **6–7 min** (370 s and 463 s in two runs; the slower one overlapped with a test run).
- **Where the time goes:** LightGBM training and scenario prediction take almost all of it. The baseline (~1 s), significance tests (~0.03 s) and calibration plot (~0.2 s) add only 1–2 s together. GEFCom got `--baseline none` and `--skip-stats` flags anyway. They don't change any reported number: `lightgbm_full`'s pinball loss is identical with and without them.
- **Consequence:** 20 experiments × ~6.5 min ≈ 2.2 h per search run, and ≈ 13 h for the 6 runs in Phase 5. Speeding things up means changing the model work (fewer quantiles, fewer scenario shifts), not the reporting. That is the fast tier's job in Phase 1. If it is still over 5 min, the fallback is 4 dev months (risk list in PLAN.md).
- **Timing must run alone:** parallel jobs (e.g. the test suite) distorted one measurement by ~25%, so the searchers run one experiment at a time.

## Fast-tier proxy check (Phase 1)
`python -m scripts.proxy_check` runs plain / recency only / scenarios only / final in both tiers. It compares the ranking by dev loss and, for each pair, whether the 4-of-7 month verdict is the same in both tiers (the accept rule depends on it). Sanity value: confirm "final" should reproduce 8.43.
_Result pending._

## Results
_None yet._
