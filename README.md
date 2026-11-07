# Forecast Experiment Agent

An LLM agent that runs the modelling loop on the GEFCom2014 load-forecasting harness: it states a hypothesis, changes the config, runs a backtest on the development folds, keeps or drops the change, and stops when the budget runs out. The best config is then scored once on the holdout and extended folds, and compared against random search and my own hand-tuned experiment log.

**Status:** Phase 2 done: a search space of 11 parameters starting from GEFCom's plain model, guardrails that block oracle and fold leaks, and an accept rule (more than 1% lower dev loss and at least 4 of 7 months better). Next: Phase 3 (agent loop). See [PLAN.md](PLAN.md); decisions and their reasons are in [NOTES.md](NOTES.md).

Runs keep Windows awake while GEFCom computes, but a closed lid still suspends them: keep the lid open and the charger plugged in during long searches.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then add ANTHROPIC_API_KEY
```
GEFCom2014 is expected at `../GEFCOM2014` with its own venv (see `configs/agent.yaml`).

## Layout
- `src/tasks/gefcom.py`: black-box adapter, `evaluate(overrides, tier)` (tier `fast` = 9 quantiles and every second scenario shift day, for search; `confirm` = dev.yaml's 20 quantiles), cached in `runs/cache/`
- `scripts/proxy_check.py`: checks the fast tier ranks configs like the confirm tier (`python -m scripts.proxy_check`, ~40 min)
- `src/search_space.py`: the 11 searchable parameters with ranges, `START_PARAMS` (plain model), `to_overrides`, and the accept rule
- `src/guardrails.py`: `check_proposal` (leakage and invalid proposals, with a reason the searcher sees) and the hard `Budget`
- `src/agent.py`, `src/tools.py`: tool-use loop (planned, Phase 3)
- `src/random_search.py`: baseline searcher (planned, Phase 4)
- `runs/`: per-run `log.jsonl` (gitignored)
- `tests/`: `python -m pytest tests -q` (subprocess mocked, no GEFCom run)
