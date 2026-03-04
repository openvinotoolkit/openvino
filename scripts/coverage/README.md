# Coverage Scripts

Coverage workflow orchestration is implemented in Python and is used by `.github/workflows/coverage.yml`.

## Entrypoint
```bash
python3 scripts/coverage/coverage.py <command>
```

## Commands
- `run-all`: run the full local flow (or a selected range).
- `step <name>`: run a single step.
- `list-tests --suite <cpp|python|js> --profile <...>`: show resolved tests for a profile.
- `validate-config`: validate YAML suite configs.

Step modules are in `scripts/coverage/steps/`.

## Profiles
Supported profiles:
- `cpu`
- `gpu` (GPU-only; runs only tests explicitly marked for GPU-only execution)
- `cpu_gpu`
- `cpu_npu`
- `cpu_npu_gpu`

Profile-specific test selection and args are defined in:
- `scripts/coverage/config/tests_cpp.yml`
- `scripts/coverage/config/tests_python.yml`
- `scripts/coverage/config/tests_js.yml`

## Local Prerequisites
- Python 3.10+
- C/C++ build toolchain and system deps (installed by `install-deps`)
- Node.js + npm for JS coverage steps

To install dependencies locally (including Node.js):
```bash
python3 scripts/coverage/coverage.py step install-deps --install-nodejs --nodejs-version 22
```

`--install-nodejs` is optional. If omitted and Node.js is missing, JS coverage steps will fail.

## Typical Local Flows
Run full flow with dependency installation:
```bash
python3 scripts/coverage/coverage.py run-all --profile cpu --install-deps --install-nodejs --nodejs-version 22
```

Run full flow without reinstalling deps:
```bash
python3 scripts/coverage/coverage.py run-all --profile cpu
```

Run only test + coverage collection phase:
```bash
python3 scripts/coverage/coverage.py run-all --profile cpu --from run-cpp-tests --to package-artifacts
```

Validate configs:
```bash
python3 scripts/coverage/coverage.py validate-config
```

Inspect resolved tests:
```bash
python3 scripts/coverage/coverage.py list-tests --suite python --profile cpu_gpu
```

## Outputs
Main artifacts in workspace root:
- `coverage.info` (native C/C++ lcov)
- `python-coverage.xml` (Python coverage XML)
- `js-lcov.info` (Node.js lcov)
- `coverage-report/index.html` (HTML report)
- `.tmp/coverage-local/step_summary.md` (local summary when not in GitHub Actions)

## Key Runtime Options
- `--parallel-jobs <N>`
- `--pytest-workers <N>`
- `--js-test-concurrency <N>`
- `--profile <name>`
- `--install-deps`
- `--install-nodejs`
- `--nodejs-version <major>`
