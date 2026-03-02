# Coverage Scripts

This directory contains the Bash implementation used by `.github/workflows/coverage.yml`.

## Goals
- Keep workflow YAML small and maintainable.
- Reuse the same scripts in CI and local reproduction.
- Preserve existing CI behavior (including continue-through-failures for tests).

## Config-Driven Test Lists
Coverage test sets are defined as YAML files:
- `config/tests_cpp.yml`
- `config/tests_python.yml`
- `config/tests_js.yml`

All three use the same top-level structure (`schema_version`, `suite`, `tests`) and are read by `lib/coverage_config.py`.

For C++, profile-aware filters are configured directly in YAML via `args` maps, for example:

```yaml
args:
  cpu: "-*IE_GPU*"
  cpu_npu: "-*IE_GPU*"
  default: ""
```

## Main Entry Points
- `install_deps.sh`: apt + pip prerequisites used in coverage workflow.
- `configure_ov.sh`: CMake configure for coverage build.
- `build_install.sh`: build OpenVINO, install runtime/tests/wheels, build JS addon runtime.
- `run_cpp_tests.sh`: C++ test suite execution and per-suite summary counters.
- `run_python_tests.sh`: Python API/frontend/layer/OVC coverage tests.
- `run_js_tests.sh`: Node.js lint/tsc/unit/e2e coverage tests.
- `collect_cpp_coverage.sh`: lcov/genhtml collection for native code.
- `write_summary.sh`: writes final Markdown summary table.
- `package_artifacts.sh`: creates `coverage-report.tgz`.
- `run_full.sh`: local orchestration wrapper.

## Local Usage
Run full flow in CPU profile:

```bash
bash scripts/coverage/run_full.sh --profile cpu
```

Run with dependency installation:

```bash
bash scripts/coverage/run_full.sh --profile cpu --install-deps
```

Run a subset:

```bash
bash scripts/coverage/run_full.sh --from run_cpp_tests --to package_artifacts
```

## Notes
- Profiles supported: `cpu`, `cpu_gpu`, `cpu_npu`, `cpu_npu_gpu`.
- Local mode stores summary/env emulation under `.tmp/coverage-local/`.
