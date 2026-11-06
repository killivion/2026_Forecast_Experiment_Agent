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

## Measurements
- Dev run runtime (Phase 0): _pending_

## Results
_None yet._
