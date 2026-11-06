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
| Human reference numbers: **8.43** on the dev grid (confirm tier), **8.78** dev / **6.79** holdout on 99 quantiles | From GEFCOM2014/README.md (experiment log and ablation table). Search results are compared with 8.43; the final once-only evaluation uses 99 quantiles and is compared with 8.78 / 6.79 |
| Failing `git rev-parse` in GEFCom is a hard error | A fallback key would let results from different GEFCom versions share a cache entry |
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
**Result (2026-11-06): the fast tier is a valid proxy.**

| config | fast loss | rank | fast min | confirm loss | rank | confirm min |
|---|---|---|---|---|---|---|
| plain | 8.863 | 4 | 0.9 | 9.165 | 4 | 2.4 |
| recency only | 8.490 | 3 | 3.1 | 8.770 | 3 | 9.2 |
| scenarios only | 8.281 | 2 | 1.2 | 8.609 | 2 | 3.1 |
| final | 8.109 | 1 | 3.8 | 8.432 | 1 | 167.6* |

- **Ranking:** identical in both tiers, and the same as the human 99-quantile ablation. All 6 pairwise 4-of-7 verdicts agree across tiers. The month-win counts differ by at most 1 (recency vs scenarios: 2 vs 3, both below 4).
- **Sanity check:** confirm "final" reproduces 8.432, the human dev-grid result (8.43).
- **Fast tier is 2.6–3× faster** than confirm. Recency weighting is the expensive part: it makes runs ~3× slower in both tiers.
- \* **Wall-time anomaly:** the confirm "final" run took 168 min, against 6.2 min for the same config in Phase 0. No GEFCom error was logged, so it was probably the machine (sleep or contention), not the config. **Consequence:** the 30 min timeout and the wall-time budget protect against this, but it is why runtimes in the logs should be read with care.
- **Budget:** a fast run takes 1–4 min, so 20 experiments ≈ 40–80 min per search. `wall_minutes` is set to 120.

## Results
_None yet._
