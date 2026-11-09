# Forecast experiment agent: plan

## Goal and headline
An LLM agent runs the modelling loop on the existing GEFCom2014 harness: it states a hypothesis, picks a config change, runs the backtest on the **development folds only**, reads the result, keeps or drops the change, and stops when the budget is spent. The best config is then scored **once** on the holdout and the extended folds.

Target headline for README / CV:
> "Under a budget of N experiments, the agent reached a development loss of X (human: 8.78). On the untouched folds it scored Y vs. 6.79 for my hand-tuned model and Z for random search with the same budget. Guardrails blocked K leakage attempts."

Open: the "human: 8.78" comparison doesn't hold as written. Search losses are fast tier (9 quantiles, hand-tuned config: 8.11) and 8.78 is on 99 quantiles, so the two can't be compared (NOTES.md, Decisions).

The comparison against **random search** and **your own experiment log** (GEFCOM2014/README.md) is what makes this an ML project, not a demo.

## Key design decisions (decide once, write into NOTES.md)
- **Separate repo, GEFCom used as a black box.** The agent calls `python -m src.evaluation.run_comparison` in the GEFCOM2014 folder via subprocess with a generated YAML and `--output-dir`, then reads `reports/<dir>/summary.csv` and `pinball_by_fold.csv`. No packaging needed, the agent can't touch GEFCom code, and any other task with a `evaluate(config) -> metrics` adapter can be plugged in later.
- **Phase 1 = config-space agent** (safe, fast, enough for the MVP). Phase 2 (optional) = agent may write feature code in a sandbox.
- **The LLM proposes; code decides.** Accept/reject rule, budget, and leakage checks live in Python, not in the prompt.
- **Plain Anthropic SDK tool-use loop, no framework** (LangGraph etc.). ~150 lines you can explain line by line in an interview.
- **Agent model: Sonnet** (cheap, good at tool use). Use Opus only to compare in one run, if at all.

## Order of work

### Phase 0: Setup and measurements (0.5 day)
- Create repo `forecast-experiment-agent` (in this folder), venv, `ANTHROPIC_API_KEY` in `.env` (gitignored), set a spend limit in the Anthropic Console.
- **Measure** the runtime of one dev run in GEFCOM2014: `python -m src.evaluation.run_comparison --config configs/dev.yaml --folds development --output-dir scratch_timing`. Expected ~5-10 min [estimate]. This decides the fast tier in Phase 1.
- AI use: Claude Code in VSCode for scaffolding. Read the tool-use docs yourself once: https://docs.claude.com (Build with Claude, Tool use).

### Phase 1: Evaluation tool / task adapter (1 day)
- `src/tasks/gefcom.py`: `evaluate(overrides: dict, tier: "fast" | "confirm") -> EvalResult` (dev loss, per-fold losses, 90% coverage, runtime, run dir).
- Merges overrides into `dev.yaml`, writes the YAML to a scratch dir, runs the subprocess, parses CSVs (`summary.csv` columns: method, fold_group, mean_pinball_loss, std_pinball_loss, n_folds, mean_coverage_90; the model row is `lightgbm_full`). Cache results by config hash (same config never runs twice).
- Pass `--baseline` with a single baseline so each run doesn't recompute all baselines (check whether `run_comparison` can skip them entirely; if not, add a small flag in GEFCom and note it). The 20-quantile dev grid has no 0.90 level, so check how coverage is computed before showing it to the agent.
- **Fast tier:** fewer quantiles (e.g. 9) and/or coarser scenario shifts, target 1-3 min per run. **Proxy check:** run plain / recency only / scenarios only / final in both tiers and check that the ranking matches. If it doesn't, the fast tier is useless; write the result into NOTES.md either way.
- AI use: **planner subagent** for the interface first, then Claude Code implements; **test-writer** for config merging and CSV parsing (mock the subprocess).

### Phase 2: Search space and guardrails (0.5 day)
- `src/search_space.py`: allowed overrides with types and ranges, e.g. LightGBM `n_estimators`, `max_depth`, `learning_rate`; `recency_half_life_years` (null or 0.5-6); `temperature_scenarios.enabled`, `shift_days` width, `combine`; `leakage.temperature_source` in {climatology, last_year}.
- `src/guardrails.py`: reject `temperature_source: actual` (oracle leak), any holdout/extended fold, unknown keys; hard budget on experiments, wall time and tokens. Every rejection is logged with its reason.
- **Accept rule** (in code): new best only if the dev loss improves by more than a noise threshold AND wins at least 4 of 7 dev months vs. the current best. You choose the threshold; justify it in NOTES.md.
- AI use: write the rule and the search space **yourself** (it's your judgment the interview will probe), let Claude Code implement and test.

### Phase 3: Agent loop (1 day)
- `src/agent.py`: manual loop with the Anthropic SDK. Tools:
  - `describe_task()`: problem, metric, what is known at forecast time, current best.
  - `get_search_space()`
  - `run_experiment(overrides, hypothesis)`: hypothesis is required; goes through guardrails, then `evaluate`.
  - `get_history()`: compact table of all runs so far.
  - `finish(best_run_id, rationale)`
- Memory = `runs/<run_id>/log.jsonl` (hypothesis, overrides, result, accept/reject, tokens). Same file feeds the notebook.
- Start the agent from the **plain** config (your ablation_plain) so it can, in principle, rediscover scenarios and recency.
- AI use: write the loop **together** with Claude Code, step by step, not in one prompt. You must be able to explain every line. Use the **reviewer** subagent afterwards.

### Phase 4: Baselines for the agent (0.5 day)
- `src/random_search.py`: same search space, same budget, same accept rule, 3 seeds.
- Reference: your own experiment log (human run) as a fixed row.
- Optional: Optuna TPE as a stronger non-LLM searcher.
- AI use: Claude Code, almost fully delegated.

### Phase 5: Runs and final evaluation (0.5 day + run time)
- 3 agent runs + 3 random-search runs, same budget (20 experiments). The agent runs differ only by sampling variation: the seed is a label, and Claude Sonnet 5.5 rejects a non-default `temperature`.
- Best config of each run: one full 99-quantile run on the holdout folds (`--folds holdout`), never seen by the searchers. The extended folds (`--folds extended`, ~1.5-2.5 h per config) only for the best agent, the best random and the human config.
- Report: dev loss vs. holdout loss per searcher (gap = validation overfitting), cost in $ and minutes, number of guardrail hits.
- AI use: run from the terminal yourself; Claude Code for the aggregation script.

### Phase 6: Reward-hacking demo (0.5 day, optional but very featureable)
- One run with the `actual` temperature guard **off** (clearly labelled). Expected: the agent finds the oracle input (loss ~3.3 instead of ~8.8) and "wins". Shows why the guard exists. Never mixed into the honest results.

### Phase 7: Presentation and docs (1 day)
- `main.py`: `python main.py --searcher agent|random --budget 20 --seed 0`.
- `notebooks/presentation.ipynb`: problem, architecture diagram, one agent trace (hypotheses in its own words), search curves (best-so-far loss vs. experiment number, agent vs. random), dev vs. holdout table, cost, failure cases, limitations.
- README (results first), NOTES.md (decisions, trade-offs, interview Q&A).
- AI use: Claude Code for the notebook; **reviewer** subagent with "full review" for consistency; claude.ai chat for interview questions on the finished repo.

### Phase 8 (later): Code-writing agent
- Agent may add one feature function to a sandboxed `agent_features.py`; GEFCom's leakage tests must pass (pytest as a tool) before any evaluation. Only after the MVP is published.

**Total MVP: ~5-6 days** (Phases 0-5 + 7), plus run time for Phase 5.

## Repo layout
```
forecast-experiment-agent/
  main.py
  src/  agent.py  tools.py  guardrails.py  search_space.py  random_search.py
        tasks/gefcom.py  logging_utils.py
  configs/  agent.yaml          # model, budget, threshold, GEFCom path
  runs/                         # gitignored except summaries
  notebooks/presentation.ipynb
  tests/
  README.md  NOTES.md  .env.example  requirements.txt
```

## Risks
- **Slow evaluations** limit the budget. Mitigation: fast tier + cache; if a fast run is still > 5 min, reduce folds to 4 dev months and say so.
- **Agent beats random only by luck** with 3 seeds. Report it honestly; "LLM was no better than random search here" is a legitimate, interesting result.
- **The agent knows GEFCom2014 from its training data.** Mention it in the limitations; the leakage guard and the dev/holdout split still hold.

## Carry-over prompts for Claude Code (VSCode)
Open `2026_Forecast_Experiment_Agent` in VSCode, start Claude Code there. CLAUDE.md from the parent folder applies.

**Prompt 0 (setup)**
```
Read PLAN.md. Do Phase 0: create the repo layout from "Repo layout" (empty modules with docstrings only), requirements.txt (anthropic, pyyaml, pandas, python-dotenv, pytest), .env.example, .gitignore (runs/, .env, .venv). Do not write logic yet. Then give me the exact command to time one dev run in ../GEFCOM2014.
```

**Prompt 1 (adapter)**
```
Use the planner subagent first. Implement Phase 1 of PLAN.md: src/tasks/gefcom.py with evaluate(overrides, tier) as described. GEFCom path comes from configs/agent.yaml. Base config is GEFCOM2014/configs/dev.yaml; read its structure, don't guess keys. Parse reports/<dir>/summary.csv and pinball_by_fold.csv (read one existing file first to get the columns). Add a config-hash cache. Then use the test-writer subagent with a mocked subprocess. Finally, write scripts/proxy_check.py that runs the four ablation configs in both tiers and prints the rankings.
```

**Prompt 2 (search space + guardrails)**
```
Implement Phase 2 of PLAN.md. Search space and accept rule are below (my decisions): <paste yours>. Guardrails must reject temperature_source=actual, any fold group other than development, and unknown keys, and return a readable reason the agent will see. Tests for every rejection path.
```

**Prompt 3 (agent loop, do it in steps)**
```
Phase 3 of PLAN.md, step 1 only: define the five tools as JSON schemas in src/tools.py and their Python handlers. No agent loop yet. Explain each schema choice in 1 line.
```
```
Step 2: write the minimal tool-use loop in src/agent.py with the Anthropic SDK (check the current SDK docs, don't guess the API). Loop until finish() or budget. Log every step to runs/<run_id>/log.jsonl. Keep it under ~150 lines. Then run it with budget=3 on the fast tier and show me the log.
```

**Prompt 4-7**
```
Phase 4 of PLAN.md: random search with the same search space, budget, accept rule and log format as the agent, so the notebook can read both.
```
```
Phase 5: write scripts/final_eval.py that takes the best run of each searcher, runs the confirm tier on holdout and extended folds once, and writes results/summary.csv (searcher, seed, dev loss, holdout loss, extended loss, cost USD, minutes, guardrail hits).
```
```
Phase 7: build notebooks/presentation.ipynb and main.py as in PLAN.md, from runs/ and results/ only (no re-running). Then run the reviewer subagent with "full review".
```

## Where to use which AI
- **claude.ai chat:** design questions before each phase, reading docs, interview prep on the finished repo.
- **Claude Code in VSCode:** all implementation; planner for Phase 1, test-writer after Phases 1-2, reviewer after Phases 3 and 7, debugger on failures.
- **You, without AI:** search space, accept rule, which results go in the headline, the NOTES.md "why" lines. These are what interviewers test.
