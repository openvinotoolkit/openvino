# Coverage Scripts

Coverage workflow orchestration is implemented in Python and is used by `.github/workflows/coverage.yml`.
GitHub matrix helpers live in `tools/coverage/ci_matrix.py`.
GitHub artifact aggregation helpers live in `tools/coverage/ci_reports.py`.

## Entrypoint
```bash
python3 tools/coverage/coverage.py <command>
```

## Commands
- `run-all`: run the full local flow (or a selected range).
- `step <name>`: run a single step.
- `list-tests --suite <cpp|python|js> --profile <...>`: show resolved tests for a profile.
- `validate-config`: validate YAML suite configs.

Step modules are in `tools/coverage/steps/`.

## Profiles
Supported profiles:
- `cpu`
- `gpu` (GPU-only; runs only tests explicitly marked for GPU execution)
- `npu` (NPU-only; runs only tests explicitly marked for NPU execution)

Profile-specific test selection and args are defined in:
- `tools/coverage/config/tests_cpp.yml`
- `tools/coverage/config/tests_python.yml`
- `tools/coverage/config/tests_js.yml`

## Local Prerequisites
- Python 3.10+
- C/C++ build toolchain and system deps (installed by `install-deps`)
- Node.js + npm for JS coverage steps

To install dependencies locally (including Node.js):
```bash
python3 tools/coverage/coverage.py step install-deps --install-nodejs --nodejs-version 22
```

`--install-nodejs` is optional. If omitted and Node.js is missing, JS coverage steps will fail.

## Typical Local Flows
Run full flow with dependency installation:
```bash
python3 tools/coverage/coverage.py run-all --profile cpu --install-deps --install-nodejs --nodejs-version 22
```

Run full flow without reinstalling deps:
```bash
python3 tools/coverage/coverage.py run-all --profile cpu
```

Run only test + coverage collection phase:
```bash
python3 tools/coverage/coverage.py run-all --profile cpu --from run-cpp-tests --to package-artifacts
```

Validate configs:
```bash
python3 tools/coverage/coverage.py validate-config
```

Inspect resolved tests:
```bash
python3 tools/coverage/coverage.py list-tests --suite python --profile gpu
```

## Outputs
Main artifacts in workspace root:
- `coverage.info` (native C/C++ lcov)
- `coverage-artifact-metadata.json` (CI shard-artifact identity file used for summary aggregation)
- `cpp-coverage-stats.env` (per-run C++ shard stats)
- `cpp-test-durations.csv` (per-run C++ test durations in seconds)
- `python-coverage-stats.env` (per-run Python shard stats)
- `python-test-durations.csv` (per-run Python test durations in seconds)
- `python-coverage.xml` (Python coverage XML)
- `js-coverage-stats.env` (per-run JS shard stats)
- `js-test-durations.csv` (per-run JS test durations in seconds)
- `js-lcov.info` (Node.js lcov)
- `coverage-report/index.html` (HTML report)
- `.tmp/coverage-local/step_summary.md` (local summary when not in GitHub Actions)

## Key Runtime Options
- `--parallel-jobs <N>`
- `--cpp-test-concurrency <N>`: run configured C++ coverage test binaries in parallel. Values above `1` isolate gcov output per run and merge it during `collect-cpp-coverage`.
- `--pytest-workers <N>`
- `--js-test-concurrency <N>`
- `--profile <name>`
- `--install-deps`
- `--install-nodejs`
- `--nodejs-version <major>`

## Step-specific Environment
- `CXX_TEST_NAMES=name1,name2,...`: limit `run-cpp-tests` to the named config entries. This is used by the GitHub Actions workflow to shard C++ coverage jobs.
- `PY_TEST_NAMES=name1,name2,...`: limit `run-python-tests` to the named config entries. This is used by the GitHub Actions workflow to shard Python coverage jobs.
- `JS_TEST_NAMES=name1,name2,...`: limit `run-js-tests` to the named config entries. This is used by the GitHub Actions workflow to shard JS coverage jobs.
- `COVERAGE_WRITE_STEP_SUMMARY=false`: suppress per-step summary output so only the final workflow summary is published in GitHub Actions.
