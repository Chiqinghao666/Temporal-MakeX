# Repository Guidelines

## Project Structure & Module Organization
Makex combines Python orchestration with a C++17 extension. `Makex.py` and `structure/` implement REP mining primitives, `gnn_models/` stores recommender backbones, and `pattern_generator/`, `global_explanations/`, plus `local_explanations/` provide the pipelines for discovering and ranking rules. Data lives under `DataSets/` (raw CSVs) with preprocessing helpers in `data_preprocess/` and shared utilities in `utils/`. Baseline adapters are collected in `baselines/`, while `pyMakex/` contains the native module (`setup.py`, headers, and build scripts). Keep large artifacts in `structure/`-scoped cache folders referenced via relative paths rather than committing them.

## Build, Test, and Development Commands
- `export CFLAGS='-std=c++17' CC=g++-9 CXX=g++-9`: ensure macOS builds use the correct toolchain.
- `cd pyMakex && python setup.py build_ext --inplace`: compiles the `pyMakex` extension required by `structure/`.
- `cd global_explanations && ./rep_discovery.sh`: runs REP discovery over the configured dataset.
- `cd local_explanations && ./run_local_explanation.sh`: produces the top-k local explanation rules.
- `python test_simple.py`: lightweight sanity test for data loading, pattern generation, and scoring without compiling C++.

## Coding Style & Naming Conventions
Target Python 3.9, Torch 1.8.1, and PEP 8 (4-space indents, snake_case names, explicit docstrings for algorithms). Keep configuration constants in UPPER_SNAKE_CASE and align module names with their directories (e.g., `structure/matchers/star_rule.py`). C++ code must compile under C++17, place shared headers in `pyMakex/include/`, and guard platform-specific code with `#ifdef`.

## Testing Guidelines
Add `pytest` cases beside the code they cover (e.g., `structure/tests/test_iterator.py`) and use deterministic fixtures derived from the small CSV samples in `DataSets/`. Maintain `python test_simple.py` as a zero-dependency smoke test and add end-to-end checks for every new REP mining strategy (load a toy graph, run the pipeline, assert rule counts). Agree on at least basic coverage: critical dataflow and score computations must have assertions before merging.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `build:`) and mention the touched dataset or module in the subject. Every PR should include: purpose, runnable commands, expected outputs or metrics, linked issues, and screenshots of explanation diffs when relevant. Request reviewers for each directory you modify and highlight any migrations (new CSV schema, encoding tables) in the checklist.

## Security & Configuration Tips
Never commit downloaded datasets; instead document their expected location (`DataSets/movielens`, `DataSets/yelp`). Keep Drive links and credentials in environment variables or private secrets, not in code. Scrub logs for user/item identifiers before uploading artifacts, and store schema converters in `data_preprocess/` so provenance remains auditable.
