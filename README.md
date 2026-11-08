# Forecast Experiment Agent

An LLM agent that runs the modelling loop on the GEFCom2014 load-forecasting harness: it states a hypothesis, changes the config, runs a backtest on the development folds, keeps or drops the change, and stops when the budget runs out. The best config is then scored once on the holdout and extended folds, and compared against random search and my own hand-tuned experiment log.

**Status:** code for Phases 2–5 is in place and tested with GEFCom mocked: a search space of 11 parameters starting from GEFCom's plain model, guardrails that block oracle and fold leaks, an accept rule (more than 1% lower dev loss and at least 4 of 7 months better), one experiment core shared by the agent and random search, and the once-only final evaluation. The three random searches are done: best dev loss by the accept rule 8.14–8.28, against 8.86 for the start config and 8.11 for my hand-tuned config (fast tier, [details](NOTES.md#results)). Next: the agent's first API run (budget 3, Haiku), then the Sonnet searches, then the once-only final evaluation. See [PLAN.md](PLAN.md); decisions and their reasons are in [NOTES.md](NOTES.md).

Runs keep Windows awake while GEFCom computes, but a closed lid still suspends them: keep the lid open and the charger plugged in during long searches.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then add ANTHROPIC_API_KEY
```
GEFCom2014 is expected at `../GEFCOM2014` with its own venv (see `configs/agent.yaml`).

## Cost
Only the agent's Claude API calls cost money, billed to the Console account of the key in `.env` (a claude.ai Pro plan doesn't cover the API). Random search and GEFCom run locally for free.
- **Hard ceiling:** buy a small prepaid balance in the Anthropic Console, keep auto-reload off, and set a workspace spend limit.
- **Test runs cost cents:** `python main.py --searcher agent --seed 0 --budget 3 --model claude-haiku-5-5`.
- **Tracked searches:** Sonnet 5.5 at medium effort, about $0.6–1.2 each, capped at $2 per search in `configs/agent.yaml`. The whole MVP comes to about $3–5 (estimates; every call's exact cost is in the run log). Details in [NOTES.md](NOTES.md#cost-decided-2026-11-07).

## Layout
- `src/tasks/gefcom.py`: black-box adapter, `evaluate(overrides, tier)` (tier `fast` = 9 quantiles and every second scenario shift day, for search; `confirm` = dev.yaml's 20 quantiles), cached in `runs/cache/`
- `scripts/proxy_check.py`: checks the fast tier ranks configs like the confirm tier (`python -m scripts.proxy_check`, ~40 min)
- `src/search_space.py`: the 11 searchable parameters with ranges, `START_PARAMS` (plain model), `to_overrides`, and the accept rule
- `src/guardrails.py`: `check_proposal` (leakage and invalid proposals, with a reason the searcher sees) and the hard `Budget`
- `src/tools.py`: the five agent tools (schemas + handlers) and the experiment core both searchers share: guardrails, duplicate check, evaluation, accept rule, log
- `src/logging_utils.py`: the run log `runs/<run_id>/log.jsonl` (event format in its docstring)
- `main.py`: runs one search, `python main.py --searcher agent|random --seed 0 [--budget 3] [--model claude-haiku-5-5]` (`--model` is agent only)
- `src/agent.py`: the manual Claude tool-use loop (Phase 3 step 2: not run against the API yet)
- `src/random_search.py`: baseline searcher with the same space, budget, accept rule and log (Phase 4)
- `scripts/final_eval.py`: scores the named searches' best configs once on unseen folds, `python -m scripts.final_eval --folds holdout --runs <id> ...` (then `--folds extended` for the best ones), into `results/summary.csv` with a human reference row
- `runs/`: per-run `log.jsonl` (gitignored)
- `tests/`: `python -m pytest tests -q` (subprocess mocked, no GEFCom run)
