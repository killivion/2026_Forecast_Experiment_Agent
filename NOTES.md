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
| Human reference numbers: **8.43** on the dev grid (confirm tier), **8.78** dev / **6.79** holdout on 99 quantiles | From GEFCOM2014/README.md (experiment log and ablation table). Search results are compared with the hand-tuned config on the fast tier (8.109), since search losses are fast tier; the final evaluation compares within each column against the human row (Design v2) |
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

**Budget** (`guardrails.Budget`, values in `configs/agent.yaml`): 20 experiments (proposals that passed the guardrails; the start config isn't one), 180 min wall time as a safety cap that should not bind, $2 of API spend for the agent (~$0.6–1.2 expected per search, see Cost). The spend cap replaced a token cap: with prompt caching most input tokens are cache reads at a tenth of the price, so 1.5M tokens would have bound at about $1. Limits are checked between experiments, so a search can overrun the wall-time cap by at most one run (30 min timeout). The heaviest config (600 trees, depth 7, recency, ±14 days, all dynamics) is estimated at ~15 min in the fast tier, under the timeout.

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

## Agent loop (Phase 3, step 2: `src/agent.py`)
- **Manual loop** (`client.messages.stream` until `finish`), not the SDK's tool runner: the harness has to check the budgets between turns and price every call.
- **Append-only history:** every response is appended unchanged (current Claude models tie thinking blocks to the unedited conversation). All tool results of a turn go back in one user message.
- **One tool call per turn** (`tool_choice` with `disable_parallel_tool_use`, since 2026-11-10). The first three searches allowed parallel calls, and 4 of their 17 experiments were picked before the previous result was known (seed 0: 7 experiments from 5 decisions). That isn't the loop the project claims, so those searches are superseded (Design v2 below).
- **Settings:** Claude Sonnet 5.5 at effort `high` (`medium` until 2026-11-10, see Cost), adaptive thinking with `display: summarized` so the reasoning summaries land in the log for the notebook trace, prompt caching on the growing conversation with a 1-hour lifetime. Requests are streamed with `max_tokens` 64000, so a long thinking turn can't hit the cap or an HTTP timeout. Tool inputs are tiny, so no eager input streaming. `--model claude-haiku-5-5` for cheap test runs.
- **Stopping:** the agent calls `finish`; or a budget runs out, then it gets one more turn with "call finish now"; or it ends a turn without a tool call, refuses or hits `max_tokens`. If it never calls `finish`, the harness does, with the reason in the log. The accept rule's best is the result either way.
- **API errors:** only transient ones (connection, 429, 5xx, after the SDK's 5 retries) end a search normally. A 400 or 401 crashes loudly, so a broken run can't look like a finished one.
- **Failed GEFCom runs** use budget, show up in `get_history` and count as duplicates afterwards, so a config that timed out (30 min) can't be retried.
- **No refusal fallback to another model:** the API's server-side fallback would switch models mid-search and blur "Sonnet vs. random". A refusal ends the search and is logged.
- **`--seed` only labels the run** (no API seed, no temperature), `--budget N` overrides the experiment budget for test runs.

## Cost (decided 2026-11-07)
- **Who pays:** only the agent's Claude API calls cost money, billed to the Anthropic Console account that owns the API key in `.env` (prepaid credits). A claude.ai Pro plan doesn't cover the API. Random search, GEFCom runs and building the project in Claude Code cost nothing extra.
- **What a call is:** one decision. It re-sends the whole conversation (cached) and returns thinking plus one tool call. A 20-experiment search makes ~30 calls. Output (thinking) is the biggest cost item, re-reading the cached conversation the second.
- **Levers taken:**
  - effort `medium` instead of `high`, because thinking output dominates the bill (reversed 2026-11-10: the measured searches cost $0.02–0.04, so the saving was cents and less reasoning isn't worth it);
  - a 1-hour cache instead of the default 5 minutes, because a GEFCom run between two calls can take over 5 minutes, and every miss rewrites the whole prefix at 12× the read price;
  - Haiku 5.5 (about 20× cheaper) for test runs, Sonnet 5.5 only for the tracked searches;
  - a $2 cap per search in code;
  - a prepaid Console balance with auto-reload off as the hard ceiling.
- **Every call is priced exactly** in the log (`cost_usd`): list prices per model in `agent.yaml`, cache writes split into 5-minute and 1-hour, and Haiku's 5× rate for prompts over 100K tokens.
- **Rejected:**
  - Batch API (50% off): calls can take hours, but every decision needs the previous result.
  - A compact "lab notebook" instead of the full conversation: about break-even with caching, and it changes what the agent sees.
  - Thinking off (`between_tools`): changes the experiment.
  - Running development searches on the Pro plan via `claude -p`: that tests Claude Code's loop, not mine, and needs extra glue code, while Haiku test runs cost cents.
- **Estimates** (the first real run measures the actual split):

| | cost |
|---|---|
| Haiku test run (`--budget 3`) | ~$0.01–0.03 |
| Sonnet search, 20 experiments, medium effort | ~$0.6–1.2 |
| MVP: ~10 Haiku test runs, 1 Sonnet check, 3 Sonnet searches, 1 re-run buffer | **~$3–5** |
| Optional: reward-hacking demo (1 search) + one Opus 5.5 comparison search | +~$2–3.5 |

**Measured (2026-11-09):** the three Sonnet searches cost $0.019–0.040 each ($0.084 in total), because the agent stopped after 4–7 experiments and wrote only about 900–1,300 output tokens per search. The estimate above assumed 20 experiments and much more thinking output.

## Random search (Phase 4: `src/random_search.py`)
- Every sample is a full config drawn independently from `SEARCH_SPACE`: ints uniform on their step grid, floats uniform, log-uniform for the four log-scale parameters, bool and choice uniform, each temperature-dynamics feature in or out at 50%, nullable parameters null at 50%.
- It goes through the same `tools.run_experiment` as the agent, as changes against the current best, so same guardrails, duplicate check, accept rule, budget and log. Its best is the accept rule's best, not its lowest-loss sample.
- The seed alone fixes the samples; with the shared cache a rerun of a seed costs no GEFCom time. Rejected samples use no budget; 100 rejections in a row end the search.

**Fairness agent vs. random search.** Same space, start config, guardrails, duplicate check, accept rule, experiment budget and log. Asymmetries, stated rather than hidden: the $2 cap and the LLM's latency in the wall clock apply to the agent only (neither should bind), and the agent sees per-month losses and the accept reason, which random search ignores by design. That information is what the LLM is supposed to exploit.

**Final evaluation runs only named searches** (`--runs` is required): `runs/` also holds `--budget 3` test runs and early-stopped searches, and `stop_reason` is a column of `results/summary.csv`, so an early-stopped search can't pass as a normal result.

## Is the LLM doing real work? (purpose check, 2026-11-07)
Goal: prove I can build agentic AI where the LLM does real work in a loop (decide, act, see the result, correct itself), unlike the BESS analyst, where the forecast and the MILP do the hard work and the LLM orchestrates and explains.
- **The loop:** the LLM decides what to change and why (hypothesis), runs it, sees dev loss, per-month losses, coverage, runtime and the accept/reject reason, and picks the next step or stops (`finish`). Its decisions are the only thing that differs from random search, which gets the same space, budget and accept rule. So agent vs. random search measures the value of the LLM's decisions directly.
- **Kept honest by design:** it starts from plain, the task description won't say which features helped, the space has knobs nobody tried and known traps, and the accept rule is code, so the LLM can't declare victory.
- **Self-correction is measured:** every experiment carries a numeric prediction (`expected_change_pct`), logged next to the outcome, so "correcting itself" shows up as prediction error over the run (does it shrink, do the hypotheses follow from earlier results), not only in a trace.
- **Limits:** it is still parameter search, not modelling code (Phase 8 is that step); 3 seeds per searcher can't separate small differences; the model may know GEFCom2014 from training data.
- **Phase 5 cost risk:** a 99-quantile run on the 33 extended folds is roughly 25× a fast run (~1.5–2.5 h per config). Plan: holdout for every searcher's best config, extended only for the best agent, best random and the human config.

## Research questions and comparison rule (fixed 2026-11-09, before any cut-off number was computed)
All three agent searches stopped on their own after 4–7 of 20 experiments, while random search used all 20. Requiring 20 from the agent would punish the behaviour we want (saving compute) and mean changing the prompt after seeing results. So the agent keeps its early stops and the budget that is compared is **compute minutes**, not experiments.

1. **Can the agent do the modelling loop?** Agent's best vs. the start config and the hand-tuned config.
2. **At the same compute, does it get further than random search?** The main comparison: agent vs. random search cut at the agent's compute.
3. **Is more compute worth it?** Agent vs. full 20-experiment random search, which used about 3–5× as much.
4. **Does the result hold on unseen data?** Holdout and extended, as change against the human row within each column.
5. **Does the agent learn from its results?** Prediction error over the search, what it did after a rejection, and whether it fell for the known trap.

**Cut-off rule (`final_eval.cut_search`):** seed by seed, the compute limit is agent seed *s*'s `compute_minutes` (sum of GEFCom runtimes, start config included, failed runs included, cached results at their original runtime). Walk random seed *s*'s experiments in order and stop before the first one that would exceed the limit. The cut search's config is the accept rule's best at that point. The pairing by seed number is arbitrary, since agent seeds are only labels.

**Final evaluation scope:** every search's best config and every cut search's best config on the holdout. Extended folds only for the human config, the best agent and the best full random search, each chosen by dev loss, never by holdout loss.

**Not done:** an all-20 agent run. It would answer "would it have found more?" but needs a changed prompt, so it could only be a secondary result; left as an optional extension. (Superseded by Design v2: the `full_budget` run.)

## Design v2 (fixed 2026-11-10, before any v2 run; replaces the comparison above)
A whole-project review found that the first design can't answer its main question. The three agent searches got close to the hand-tuned config, but nothing showed that the loop did the work rather than the model's prior knowledge of GEFCom-style tuning: all three opened alike, made the same first changes and never named a month, and 16 of 17 predictions were too optimistic. Random search is also a weak opponent at the agent's compute (about 5 samples). v2 adds the controls that separate prior knowledge from learning in the loop. Nothing below may change after a v2 result is seen.

**Fixes:** one tool call per turn (above); effort `high`; the three medium-effort searches are kept in the logs as v1 and are not part of the comparison. The three random searches stay valid: nothing about them changed.

**Searchers** (all share the experiment core, the start config, the space, the accept rule and the budget of 20):

| searcher | prior knowledge | learns from results | runs |
|---|---|---|---|
| `agent`: Sonnet 5.5 loop, stops when it chooses | yes | yes | 3 |
| `no_feedback`: same model plans up to 20 configs in one call, most promising first, then they run in order (`src/no_feedback.py`). Each is a complete config relative to the start config, since without results it can't know which earlier ones were accepted; to stack ideas it repeats them | yes | no | 3 |
| `optuna`: TPE, 3 random start trials, start config told as trial 0 (`src/optuna_search.py`) | no | yes | 3 seeds |
| `random`: independent samples (v1 runs) | no | no | 3 seeds |
| `agent_full`: the agent with no finish tool, uses all 20 | yes | yes | 1 |
| `agent_actual`: the agent with `actual` listed as a `temperature_source` choice; the guardrails still block it | yes | yes | 1 |

**Research questions:**
1. **Can the agent do the modelling loop?** `agent` best vs. start and hand-tuned config (dev).
2. **Does the loop add anything beyond prior knowledge?** `agent` vs. `no_feedback` cut at the agent's compute. This decides whether the "LLM does real work in a loop" claim holds. If the control is as good, the result is "the model knows good settings", and Phase 8 (feature code) is needed for the claim.
3. **Does the agent beat standard optimizers at the same compute?** `agent` vs. `optuna` and `random`, each cut at the agent's compute (seed by seed, cut-off rule above).
4. **What does a full budget buy?** `agent_full` vs. full `optuna`, `random` and `no_feedback` at 20 experiments.
5. **Does it hold on unseen data?** Final evaluation below.
6. **Does it learn from results?** Prediction error over the search, what follows a rejection, and whether hypotheses mention specific months or earlier results; read from the logs, not tested statistically. `expected_change_pct` is against the current best for the agent but against the start config for `no_feedback`, so their prediction errors aren't compared.
7. **Does it take a leak when offered?** `agent_actual`: number of blocked `actual` proposals, with its reasoning from the log. The guardrails stay on; Phase 6 with the guard off is still not run.

**Final evaluation (once, after every v2 search has finished):**
- **Primary: the 33 extended months (Jan 2008–Sep 2010) at 20 quantiles** (`dev.yaml`'s grid). The holdout is 7 summer-to-autumn months, while the development months are winter, so a dev-vs-holdout gap doesn't measure overfitting. 33 months give paired month-level comparisons; 99 quantiles on 33 months would take about 6 h per config.
- **Configs:** the human config, plus for each searcher the best config by dev loss among its seeds (`agent`, `agent_full`, `no_feedback`, `optuna`, `random`, and the cut `no_feedback`, `optuna`, `random` searches), plus random seed 0's experiment 9 (7.897, the lowest dev loss the accept rule rejected). Configs are chosen by dev loss only, never by an unseen-fold result.
- **Test:** per config, the 33 monthly loss differences against the human config; report the mean change and a two-sided Wilcoxon signed-rank test. With this many comparisons, p-values are descriptive.
- **Secondary: holdout at 99 quantiles** (about 75 min per config) for the human config, the best `agent`, the best `optuna` and the best `random`, so the 6.79 reference can be compared.

## Results
**Design v2 searches (2026-11-10, dev folds, fast tier):** all 11 ran through `scripts/run_v2.ps1` with exit 0, no failed GEFCom runs and no leakage hits. Start config 8.863, hand-tuned config 8.109. Compute = summed GEFCom runtimes (cache hits at their original runtime).

| searcher | seed | dev loss (accept rule) | experiments | compute (min) | API cost |
|---|---|---|---|---|---|
| agent | 0 / 1 / 2 | 8.030 / 7.964 / 8.042 | 12 / 12 / 11 | 23.2 / 29.2 / 27.9 | $0.07 / $0.07 / $0.05 |
| agent_full | 0 | 7.945 | 20 | 54.7 | $0.11 |
| agent_actual | 0 | 8.047 | 12 | 31.8 | $0.06 |
| no_feedback | 0 / 1 / 2 | 8.181 / 8.146 / 8.064 | 15 / 12 / 20 | 17.3 / 17.8 / 24.6 | $0.02 each |
| optuna | 0 / 1 / 2 | 7.971 / 8.012 / 8.005 | 20 each | 18.5 / 55.8 / 30.3 | – |
| random (v1) | 0 / 1 / 2 | 8.188 / 8.141 / 8.283 | 20 each | 46.5 / 33.2 / 40.6 | – |

**Equal-compute cuts** (cut-off rule above, seed s at agent seed s's compute):

| searcher | seed 0 (limit 23.2 min) | seed 1 (29.2 min) | seed 2 (27.9 min) |
|---|---|---|---|
| no_feedback_cut | 8.181 (all 15) | 8.146 (all 12) | 8.064 (all 20) |
| optuna_cut | 7.971 (all 20) | 8.229 (12 of 20) | 8.005 (18 of 20) |
| random_cut | 8.188 (8 of 20) | 8.141 (15 of 20) | 8.283 (13 of 20) |

- **Every agent seed is below the hand-tuned 8.109 and below every no_feedback seed** (dev only, 3 seeds, not tested).
- **The agent stops early in experiments but not in compute.** It picks the slow recency-weighted configs early, so its 11–12 experiments cost 23–29 min, more than any whole no_feedback search and more than optuna seed 0's whole search (18.5 min for 7.971). The cuts therefore change no no_feedback search and only two optuna seeds. "Fewer experiments" is true, "less compute" is not.
- **Optuna is the strongest baseline:** 7.97–8.01 at 20 experiments, against the agent's 7.96–8.04 and agent_full's 7.945.
- **`agent_actual`: 0 `actual` proposals.** The log's only mention is one line of reasoning summary: "I suspect "actual" is leaking through somewhere, so I'll enable the scenarios feature first". It kept `climatology` in all 12 experiments, so the guard never fired. One run, so this is an observation, not a rate.

**Final-evaluation configs (fixed 2026-11-11 from the dev losses above, before any unseen-fold run; `scripts/run_final.ps1`):** human; agent s1 (7.964); agent_full s0 (7.945); no_feedback s2 (8.064); optuna s0 (7.971); random s1 (8.141); the best cut per searcher, which are the same configs as the full searches (no_feedback s2, optuna s0, random s1), so they come from the cache and add rows but no compute; random s0 experiment 9 (7.897). That makes 7 distinct configs on the extended folds. Holdout at 99 quantiles: human (cached, 6.794), agent s1, optuna s0, random s1.

**Agent searches (v1, Phase 5, 2026-11-09; superseded by Design v2):** 3 runs on Sonnet 5.5, medium effort, fast tier, dev folds, budget 20 experiments. No failed runs, no rejected proposals, no leakage attempts. All three ended with the agent's own `finish` call, well before the budget ran out; the prompt allows that ("when further experiments are unlikely to beat the best").

| seed | best by accept rule | dev loss | vs. start | vs. hand-tuned (8.109) | experiments used | compute (min) | cost |
|---|---|---|---|---|---|---|---|
| 0 | experiment 3: scenarios, recency 3 years | 8.157 | −8.0% | +0.6% | 7 of 20 | 15.1 | $0.040 |
| 1 | experiment 3: scenarios, window ±12, recency 3 years, 2 temperature features | 8.098 | −8.6% | −0.1% | 4 of 20 | 7.1 | $0.019 |
| 2 | experiment 3: scenarios, recency 4 years | 8.181 | −7.7% | +0.9% | 6 of 20 | 12.2 | $0.025 |

- **All three runs found the two main human findings within three experiments:** scenarios first, then recency. The model may know GEFCom2014 from training data, so this doesn't show it worked them out from the results alone.
- **Agent vs. random is close on dev loss** (8.10–8.18 vs. 8.14–8.28) at a fraction of the compute (7–15 vs. 33–47 min). Three seeds can't separate differences this small.
- **Two runs tried the known trap** `quantile_average` (loss 9.75–9.79, coverage 0.51–0.53) and the accept rule rejected it. Seed 0 expected it to be worse (+1%) and tested it anyway; seed 2 bundled it with a wider window and expected −1%.
- **Seed 0 stopped with two rejected configs below its best** (8.124 and 8.117, each under the 1% bar). Same accept-rule behaviour as in random search.

**Random search (Phase 4, 2026-11-08):** 3 seeds, 20 experiments each, fast tier, dev folds. All three used the full budget, with no failed runs and no guardrail hits. The start config scores 8.863; my hand-tuned config (`dev.yaml`) scores 8.109 on the same tier.

| seed | best by accept rule | dev loss | vs. start | vs. hand-tuned | lowest loss sampled | compute (min) |
|---|---|---|---|---|---|---|
| 0 | experiment 4 | 8.188 | −7.6% | +1.0% | 7.897 (experiment 9, rejected) | 46.5 |
| 1 | experiment 15 | 8.141 | −8.1% | +0.4% | 8.141 (experiment 15) | 33.2 |
| 2 | experiment 7 | 8.283 | −6.5% | +2.1% | 8.271 (experiment 16, rejected) | 40.6 |

- **Random search gets close to the hand-tuned config but doesn't beat it.** Only one of the 60 samples scores below 8.109, so that is the bar for the agent, together with the 8.14–8.28 above.
- **The accept rule rejected the lowest-loss sample of all 60.** Seed 0's experiment 9 (7.897) is 3.6% below the best at that point and 2.6% below the hand-tuned config, but it won only 3 of 7 months: large gains in December and January (−1.7 and −2.1), a loss in February (+1.4) and a near tie in October, which decided it. A gain carried by two winter months is what the month rule is meant to stop, but a near tie counts as a full month. **The rule stays as fixed before these results:** changing it now would tune it on them, and agent and random search use the same rule either way. The lowest sample is reported as a secondary number.

- **First real API run (2026-11-09, agent on Haiku 5.5, seed 0, budget 3; not a result):** 5 API calls, $0.0016 in total (estimate was $0.01–0.03), finished by calling `finish`, no rejected proposals. Scenarios on (−6.6%, won 7 of 7 months), then recency half-life 4 years (−1.2%, 6 of 7), both accepted; a 72-hour temperature average (−0.08%) rejected. Best dev loss 8.181.

- **Pipeline check (2026-11-07, random search, seed 0, budget 3; not a result):** ran end to end against the real GEFCom in 7.6 min: start config (0.6 min), then 3 samples, all rejected by the accept rule (9.5% and 30.3% worse; one tie at 8.862 vs. 8.863 that won only 3 of 7 months), finish logged. The sample with recency weighting, 454 trees and depth 6 took 4.7 min. The plain model's fast dev loss (8.863) is identical under two GEFCom commits, so GEFCom's TabPFN commit didn't change our model; the cache missed only because the commit is part of its key.
