# Repository Guidelines

## Project Structure & Module Organization
- `src/` — core libraries: `core/`, `inference/`, `plugins/`, `frontends/`, language `bindings/`.
- `tests/` — C++ (gtest/ctest), Python (pytest) suites: e2e, stress, memory, model hub.
- `samples/` — runnable C/C++ and JS examples (enable with CMake).
- `docs/`, `tools/`, `scripts/`, `thirdparty/`, `bin/`, `build/` — docs, utilities, vendored deps, outputs.

## Build, Test, and Development Commands
- Init and deps (Linux): `git submodule update --init --recursive` then `sudo ./install_build_dependencies.sh`.
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON -DENABLE_SAMPLES=ON`.
- Build: `cmake --build build --parallel`.
- C++ tests: `ctest --test-dir build --output-on-failure [-R <pattern>]`.
- Samples (example): `./bin/intel64/Release/hello_classification <model.xml> <image.jpg> CPU`.
- Python wheel (optional): add `-DENABLE_PYTHON=ON -DENABLE_WHEEL=ON` and `pip install build/wheels/<wheel>.whl`.

## Coding Style & Naming Conventions
- C++: `.clang-format` (Google-based), 4-space indent, 120-column limit. Prefer `.hpp/.cpp`. Types `PascalCase`, functions/variables `snake_case`, macros `UPPER_SNAKE`.
- Python: Black configured (line length 160). Run `black src/bindings/python` and type-check with `pyright` when applicable. Modules/functions `snake_case`, classes `PascalCase`.
- CMake/Docs: follow existing style in touched files; keep targets and options consistent with current naming (e.g., `ENABLE_TESTS`).

## Testing Guidelines
- C++: enable with `-DENABLE_TESTS=ON`; run via `ctest`. Add unit tests alongside components; keep fast tests filterable with `-R`.
- Python: install deps `pip install -r tests/constraints.txt` (plus `tests/requirements_*` as needed). Run: `pytest -q -m precommit` for quick checks; mark long runs `@pytest.mark.nightly`.
- Cover new code paths and edge cases (failures, device fallbacks, shapes/precision).

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject; scope tags where relevant (e.g., `[CORE]`, `[NPU]`), reference issues/PRs: `Fixes #12345`.
- PRs: clear description, rationale, linked issues, test plan/results, perf notes if applicable. Include docs updates where user-facing. Ensure CI green and request reviewers per `.github/CODEOWNERS`.

## Security & Configuration Tips
- Do not commit secrets or local paths; respect `.gitignore`. See `SECURITY.md` for reporting.
- In restricted networks, use `./scripts/submodule_update_with_gitee.sh` for submodules.

