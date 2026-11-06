# Forecast Experiment Agent

An LLM agent that runs the modelling loop on the GEFCom2014 load-forecasting harness: it states a hypothesis, changes the config, runs a backtest on the development folds, keeps or drops the change, and stops when the budget runs out. The best config is then scored once on the holdout and extended folds, and compared against random search and my own hand-tuned experiment log.

**Status:** Phase 1 in progress (GEFCom adapter with fast/confirm tiers; proxy check running). There are no results yet. See [PLAN.md](PLAN.md).

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then add ANTHROPIC_API_KEY
```
GEFCom2014 is expected at `../GEFCOM2014` with its own venv (see `configs/agent.yaml`).

## Layout
- `src/tasks/gefcom.py`: black-box adapter, `evaluate(overrides, tier)` (tier `fast` = 9 quantiles for search, `confirm` = 20), cached in `runs/cache/`
- `scripts/proxy_check.py`: checks the fast tier ranks configs like the confirm tier (`python -m scripts.proxy_check`, ~40 min)
- `src/search_space.py`, `src/guardrails.py`: allowed overrides, accept rule, leakage checks
- `src/agent.py`, `src/tools.py`: tool-use loop
- `src/random_search.py`: baseline searcher
- `runs/`: per-run `log.jsonl` (gitignored)
- `tests/`: `python -m pytest tests -q` (subprocess mocked, no GEFCom run)
