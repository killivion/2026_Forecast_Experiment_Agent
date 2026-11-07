# Notes

## Architecture
- Separate repo. GEFCom2014 is treated as a black box: the adapter writes a YAML config, calls `python -m src.evaluation.run_comparison` in `../GEFCOM2014` through a subprocess, and parses `reports/<dir>/summary.csv` and `pinball_by_fold.csv`.
- The LLM proposes changes, and Python code decides: the accept rule, the budget and the leakage guards are all in code, not in the prompt.
- Searchers speak in flat, named parameters (`src/search_space.py`). `to_overrides` turns them into GEFCom's nested config, so things outside the space (folds, quantile grid, data) can't even be expressed.
- The agent loop uses the plain Anthropic SDK with tool use, no framework.
- **One experiment core for both searchers** (`src/tools.py`): guardrails → duplicate check → fast-tier evaluation → accept rule → log. The agent reaches it through `tool_use` blocks; random search calls the same functions directly. So the only difference between the searchers is how the next proposal is chosen.
- **Run log** `runs/<run_id>/log.jsonl` (format in `src/logging_utils.py`): start, experiment (id 0 = start config), rejected, error, llm_call, finish. The notebook and the final evaluation read only this.

## Decisions
| Decision | Why |
|---|---|
| GEFCom as a subprocess, not a package | The agent can't change the evaluation code, and other tasks only need an `evaluate(config)` adapter |
| Phase 1 = config-space only | Safe and fast. Code-writing comes later (Phase 8) |
| No agent framework | A ~110-line loop I can explain line by line |
| Sonnet as the agent model | Cheap and good at tool use |
| Two tiers: **fast** (9 quantiles, every second scenario shift day) for search, **confirm** (dev.yaml's 20 quantiles) for checking | A full dev run takes ~6 min; the fast tier is ~1.6–2.7× faster. The fast grid keeps 0.05/0.95 because GEFCom's coverage metric looks those levels up by value. Even shift days keep 0 (the unshifted scenario) and the window width |
| The tier owns the quantile grid; the fast tier keeps even shift days of any shift list | An override can't break the coverage metric, and a window-width experiment keeps its width in both tiers (dev.yaml's `[-7..7]` becomes `[-6,-4,...,6]`, exactly what the proxy check validated) |
| Fast and confirm losses are never compared | Different quantile grids give different loss scales. Same reason the human 8.78 (99 quantiles) is not comparable to dev.yaml's 8.43 (20 quantiles) |
| Human reference numbers: **8.43** on the dev grid (confirm tier), **8.78** dev / **6.79** holdout on 99 quantiles | From GEFCOM2014/README.md (experiment log and ablation table). Search results are compared with 8.43; the final once-only evaluation uses 99 quantiles and is compared with 8.78 / 6.79 |
| Failing `git rev-parse` in GEFCom is a hard error | A fallback key would let results from different GEFCom versions share a cache entry |
| Cache key = GEFCom git commit + tier + merged config, stored as JSON in `runs/cache/` | The same config never runs twice; a code change in GEFCom invalidates old results automatically. Uncommitted GEFCom edits are **not** detected |
| Failed runs raise `GefcomRunError` and are never cached; 30 min timeout | The agent sees the error text as the tool result; a hung run can't block a search |
| The adapter keeps Windows awake while GEFCom runs (`SetThreadExecutionState`, system + display required) | Windows doesn't count CPU load as activity: an unattended laptop goes into Modern Standby and suspends the run (see Runtime findings) |
| Final evaluation (`gefcom.evaluate_final`, `scripts/final_eval.py`): default.yaml + the best config's overrides, 99 quantiles, holdout or extended folds, cached per folds, 12 h timeout | The once-only score on folds no searcher saw; the searchers can't reach it (`tools.py` only calls the fast tier). The human row (GEFCom's final model, no overrides) runs first and must reproduce 6.79 on the holdout |
| Dev vs. holdout compared as improvement over the human row, within each column | The search's dev loss is fast tier (9 quantiles), holdout and extended are 99 quantiles; a relative comparison is scale-free and needs no extra 99-quantile dev runs |
| `compute_minutes` (sum of experiment runtimes) next to `wall_minutes` | Searches share the cache, so later ones are faster in wall time; summed runtimes don't depend on the order of the searches |
| Searchers' window ±6 vs. the human ±7 | The window is even only (fast-tier rule), so the space contains the human recipe up to one shift day |

## Measurements
- **Dev run (Phase 0):** `configs/dev.yaml` with 20 quantiles and 7 development folds takes about **6–7 min** (370 s and 463 s in two runs; the slower one overlapped with a test run).
- **Where the time goes:** LightGBM training and scenario prediction take almost all of it. The baseline (~1 s), significance tests (~0.03 s) and calibration plot (~0.2 s) add only 1–2 s together. GEFCom got `--baseline none` and `--skip-stats` flags anyway. They don't change any reported number: `lightgbm_full`'s pinball loss is identical with and without them.
- **Timing must run alone, plugged in, lid open:** parallel jobs (e.g. GEFCom's test suite) distorted one measurement by ~25%, and standby suspends runs (below). The searchers run one experiment at a time.

## Fast-tier proxy check (Phase 1)
`python -m scripts.proxy_check` runs plain / recency only / scenarios only / final in both tiers. It compares the ranking by dev loss and, for each pair, whether the 4-of-7 month verdict is the same in both tiers (the accept rule depends on it). Sanity value: confirm "final" should reproduce 8.43.
**Result (2026-11-06): the fast tier is a valid proxy.**

| config | fast loss | rank | fast min | confirm loss | rank | confirm min |
|---|---|---|---|---|---|---|
| plain | 8.863 | 4 | 0.9 | 9.165 | 4 | 2.4 |
| recency only | 8.490 | 3 | 3.1 | 8.770 | 3 | 9.2* |
| scenarios only | 8.281 | 2 | 1.2 | 8.609 | 2 | 3.1* |
| final | 8.109 | 1 | 3.8 | 8.432 | 1 | 167.6** |

- **Ranking:** identical in both tiers, and the same as the human 99-quantile ablation. All 6 pairwise 4-of-7 verdicts agree across tiers. The month-win counts differ by at most 1 (recency vs scenarios: 2 vs 3, both below 4).
- **Sanity check:** confirm "final" reproduces 8.432, the human dev-grid result (8.43).
- **Speed:** the fast minutes are clean. Of the confirm minutes only plain (2.4) is: \* ran while the laptop was in connected standby (screen off, on battery), \*\* was suspended (next section). Clean pairs: plain 0.9 vs 2.4 min, final 3.8 vs 6.2 min (Phase 0) → the fast tier is **~1.6–2.7× faster**. An earlier "2.6–3×" claim used the distorted numbers.
- **Budget:** a fast run takes 1–4 min with the start-like configs, so 20 experiments ≈ 40–80 min per search.

## Runtime findings
- **The 168-min run was the laptop in standby, not the code.** Windows event log: Modern Standby at 15:04 (idle timeout), network-disconnected standby from 15:13, lid closed until 17:51. The run computed for ~1 min before and ~9 min after; the loss (8.432) is unaffected. The 30-min timeout did not fire because Windows timeouts pause in standby. This machine has no S3 sleep, only Modern Standby (`powercfg /a`).
- **Fix:** the adapter now holds a system- and display-required request while GEFCom runs, so the idle timeout can't start standby. Closing the lid still suspends a run, so long runs need the lid open and the charger plugged in.
- **Recency weighting makes runs ~3× slower** (fast: plain 0.9 → 3.1 min, scenarios 1.2 → 3.8 min). Isolated LightGBM 4.4 benchmark (50k rows, 200 trees, timed interleaved because another job was running): quantile loss 0.70 s without weights, 0.92 s with all-ones weights, 2.85 s with recency weights (4.1×); l2 loss shows no slowdown. So the cost sits inside LightGBM's quantile training with uneven weights, not in our code or GEFCom's. It can't be removed without changing the model, so it stays. The agent sees it in the parameter description.

## Search space (Phase 2)
Starting point = GEFCom's plain ablation (`START_PARAMS`), so a searcher has to rediscover scenarios and recency itself. `min_child_samples=20` is LightGBM's default, made explicit (same model).

| parameter | range | start | why this range |
|---|---|---|---|
| `n_estimators` | 100–600, log | 300 | Half to double the human value. Runtime is linear; at 1000 one run with recency takes ~13 min, and the wall-time cap, not the search, would decide |
| `learning_rate` | 0.02–0.2, log | 0.05 | 0.01 was dropped: with ≤ 600 trees it can only underfit (trees × rate ≤ 6 vs 15 now) |
| `max_depth` | 3–7, `num_leaves` = 2^depth | 4 | Depth 2 can't model hour × temperature × day-type interactions; depth 8 (256 leaves on ~50k rows) doubles the runtime and mostly overfits the 0.05/0.95 quantiles. Leaves are tied to depth: one complexity knob, no inconsistent pairs |
| `min_child_samples` | 10–300, log | 20 | Never tried. Below 10 rows a leaf's 5% quantile is noise |
| `recency_half_life_years` | null or 1–8, log | null | Human tried 2 (kept) and 4. 0.5 would mean last year's month weighs 0.25, i.e. mostly one year of data; 8 is close to no weighting (~6 years of load) |
| `scenarios_enabled` | true/false | false | The main human finding (~0.5 loss) |
| `scenario_window_days` | 0–14, even | 6 | Human tried 0 / ±4 / ±7. ±14 is two weeks of seasonal drift, about where a "same date" scenario stops being plausible. One number instead of a raw shift list. Even only, because the fast tier keeps even shift days and an odd window would rerun the even one below it; dev.yaml's ±7 equals ±6 in the fast tier |
| `scenario_combine` | mixture / quantile_average | mixture | The known bad option stays on purpose: quantile averaging collapsed coverage to 0.52. Tests whether a searcher reads coverage |
| `scenario_max_years` | null or 1–9 | null | Never tried. Load starts in 2005 but temperature in 2001, and the training folds include those temperature rows, so the dev months have 9–10 earlier years: more samples vs. recent climate |
| `temperature_source` | climatology / last_year | climatology | Only used with scenarios off. `actual` is excluded (oracle) |
| `temperature_dynamics` | subset of 5 features | none | Human found no gain (+0.01–0.03): a known dead end, tests budget discipline |

**Canonical configs:** `to_overrides` writes settings the model doesn't use with their start values (`scenario_*` while scenarios are off, `temperature_source` while they are on), floats as floats and features sorted. So two parameter sets share a cache entry exactly when they give the same model, and a proposal that only changes unused settings is rejected as "no effective change" instead of spending budget on a duplicate run.

Left out on purpose: `random_state` (no bagging or feature subsampling, so LightGBM is deterministic and a seed changes nothing; this also means re-running a config can't measure noise), the quantile grid, folds and data (fixed evaluation).

## Accept rule
New best only if the fast-tier dev loss is **more than 1% lower** than the current best's **and** lower in **at least 4 of the 7** dev months (`search_space.accept`).
- **Why 1%:** in the human log, ideas that did nothing moved the dev loss by ≤ 0.6% (temperature dynamics +0.01–0.03, recalibration +0.05 on 8.43). The smallest real gain that was kept is 2.1% (recency on top of scenarios, 8.61 → 8.43). 1% sits between them.
- **Why 4 of 7 as well:** Dec 2010 (record cold, loss ~19 vs 3–11 in other months) can carry the mean on its own; the month rule stops a "win" that comes from one month. On its own the month rule is weak: a change with no effect wins ≥ 4 of 7 months half the time, so the 1% threshold does the noise filtering.
- **Known trade-off:** each accepted best raises the bar on the same 7 months, so later accepts are partly selection on noise. The untouched holdout measures how much (dev-vs-holdout gap per searcher).
- Both numbers are in `configs/agent.yaml`. Comparing results from different tiers raises an error.

## Guardrails
`guardrails.check_proposal(changes, current)` runs before every experiment and returns a readable reason the searcher sees:
- **leakage:** any `"actual"` value or "oracle" anywhere in the proposal (oracle temperature), or anything mentioning folds / holdout / extended / backtest / test tasks. The whole proposal is scanned, including nested and unknown keys, so a leak can't hide behind a wrong key.
- **invalid:** unknown parameter (the reason lists the allowed ones), wrong type (`5.0` or `true` for an integer), out of range or off the step grid, null where not allowed, or no effective change against the current best.
- "Leakage" means only the oracle temperature and the evaluation folds. Everything else outside the space (quantile grid, data paths, look-alike spellings such as a Cyrillic "а") is still blocked, but counted as invalid, so the reported leakage count is a lower bound. The allow-list in `_check_value` is the real defence; the regexes only label the rejection.
- A rejected proposal is not run and uses no experiment budget. The loop logs it, and the report counts the leakage ones.
- Second line of defence: the flat space can't express folds or quantiles, and the adapter always passes `--folds development`.

**Budget** (`guardrails.Budget`, values in `configs/agent.yaml`): 20 experiments (proposals that passed the guardrails; the start config isn't one), 180 min wall time as a safety cap that should not bind, $5 of API spend for the agent (~$1–2 expected per search). The spend cap replaced a token cap: with prompt caching most input tokens are cache reads at a tenth of the price, so 1.5M tokens would have bound at about $1. Limits are checked between experiments, so a search can overrun the wall-time cap by at most one run (30 min timeout). The heaviest config (600 trees, depth 7, recency, ±14 days, all dynamics) is estimated at ~15 min in the fast tier, under the timeout.

## Agent tools (Phase 3, step 1)
| tool | input | design choice |
|---|---|---|
| `describe_task` | none | States the problem, metric, what is known at forecast time and the rules, but not which settings helped before, so the agent has to find out |
| `get_search_space` | none | Generated from `SEARCH_SPACE`, so the agent's view can't drift from what the guardrails enforce |
| `run_experiment` | `changes`, `hypothesis`, `expected_change_pct` | `changes` is a free-form object on purpose: the guardrails, not the schema, enforce the space, so leak attempts stay visible and countable. The required numeric prediction makes self-correction measurable (prediction vs. outcome) |
| `get_history` | none | Compact rows (changes, loss, coverage, accepted, hypothesis), so the agent can re-read its own reasoning |
| `finish` | `best_run_id`, `rationale` | The agent names a best run, but the accept rule's best is what gets evaluated; a mismatch is logged as a finding |

- **Tool results** show the change vs. the best in %, the months it beat the best, month losses by name ("Dec 2010"), coverage and runtime, but not whether a result came from the cache: the cache is shared across searches, and exposing it would tell a later run what an earlier one tried.
- **Duplicates are rejected without budget:** re-running a known config teaches nothing.
- **No temperature or seed for the agent:** Claude Sonnet 5.5 rejects non-default `temperature` (from API version 2026-10-21 every model does), so the 3 agent runs differ only by sampling variation. Random-search seeds are real seeds.
- **Anthropic SDK 0.125.0 on Python 3.9:** the newest SDK for Python 3.9 (1.x needs 3.10+). It already has the request parameters the loop needs.

## Agent loop (Phase 3, step 2: `src/agent.py`, not run yet)
- **Manual loop** (`client.messages.create` until `finish`), not the SDK's tool runner: the harness has to check the budgets between turns and price every call.
- **Append-only history:** every response is appended unchanged (current Claude models tie thinking blocks to the unedited conversation). All tool results of a turn go back in one user message.
- **Settings:** Claude Sonnet 5.5, effort `high` (the decisions are what is measured), adaptive thinking with `display: summarized` so the reasoning summaries land in the log for the notebook trace, prompt caching on the growing conversation (`cache_control` on the request). Requests are streamed with `max_tokens` 64000, so a long thinking turn can't hit the cap or an HTTP timeout. Tool inputs are tiny, so no eager input streaming.
- **Stopping:** the agent calls `finish`; or a budget runs out, then it gets one more turn with "call finish now"; or it ends a turn without a tool call, refuses or hits `max_tokens`. If it never calls `finish`, the harness does, with the reason in the log. The accept rule's best is the result either way.
- **API errors:** only transient ones (connection, 429, 5xx, after the SDK's 5 retries) end a search normally. A 400 or 401 crashes loudly, so a broken run can't look like a finished one.
- **Failed GEFCom runs** use budget, show up in `get_history` and count as duplicates afterwards, so a config that timed out (30 min) can't be retried.
- **No refusal fallback to another model:** the API's server-side fallback would switch models mid-search and blur "Sonnet vs. random". A refusal ends the search and is logged.
- **`--seed` only labels the run** (no API seed, no temperature), `--budget N` overrides the experiment budget for test runs.

## Random search (Phase 4: `src/random_search.py`)
- Every sample is a full config drawn independently from `SEARCH_SPACE`: ints uniform on their step grid, floats uniform, log-uniform for the four log-scale parameters, bool and choice uniform, each temperature-dynamics feature in or out at 50%, nullable parameters null at 50%.
- It goes through the same `tools.run_experiment` as the agent, as changes against the current best, so same guardrails, duplicate check, accept rule, budget and log. Its best is the accept rule's best, not its lowest-loss sample.
- The seed alone fixes the samples; with the shared cache a rerun of a seed costs no GEFCom time. Rejected samples use no budget; 100 rejections in a row end the search.

**Fairness agent vs. random search.** Same space, start config, guardrails, duplicate check, accept rule, experiment budget and log. Asymmetries, stated rather than hidden: the $5 cap and the LLM's latency in the wall clock apply to the agent only (neither should bind), and the agent sees per-month losses and the accept reason, which random search ignores by design. That information is what the LLM is supposed to exploit.

**Final evaluation runs only named searches** (`--runs` is required): `runs/` also holds `--budget 3` test runs and early-stopped searches, and `stop_reason` is a column of `results/summary.csv`, so an early-stopped search can't pass as a normal result.

## Is the LLM doing real work? (purpose check, 2026-11-07)
Goal: prove I can build agentic AI where the LLM does real work in a loop (decide, act, see the result, correct itself), unlike the BESS analyst, where the forecast and the MILP do the hard work and the LLM orchestrates and explains.
- **The loop:** the LLM decides what to change and why (hypothesis), runs it, sees dev loss, per-month losses, coverage, runtime and the accept/reject reason, and picks the next step or stops (`finish`). Its decisions are the only thing that differs from random search, which gets the same space, budget and accept rule. So agent vs. random search measures the value of the LLM's decisions directly.
- **Kept honest by design:** it starts from plain, the task description won't say which features helped, the space has knobs nobody tried and known traps, and the accept rule is code, so the LLM can't declare victory.
- **Planned for Phase 3:** every experiment carries a falsifiable prediction (expected direction and rough size), and the log records whether it came true. That makes "correcting itself" measurable as a prediction hit rate over the run, instead of only visible in a trace.
- **Limits:** it is still parameter search, not modelling code (Phase 8 is that step); 3 seeds per searcher can't separate small differences; the model may know GEFCom2014 from training data.
- **Phase 5 cost risk:** a 99-quantile run on the 33 extended folds is roughly 25× a fast run (~1.5–2.5 h per config). Plan: holdout for every searcher's best config, extended only for the best agent, best random and the human config.

## Results
_None yet._
