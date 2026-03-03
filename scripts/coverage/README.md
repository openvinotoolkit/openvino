# Coverage Scripts

Coverage workflow orchestration is implemented in Python and used by `.github/workflows/coverage.yml`.

## Main Entrypoint

```bash
python3 scripts/coverage/coverage.py <command>
```

## Commands
- `step <name>`: run one workflow step.
- `run-all`: run all steps locally with optional range selection.
- `list-tests --suite <cpp|python|js> --profile <...>`: print resolved tests.
- `validate-config`: validate YAML test configs.

Step names:
- `install-deps`
- `configure`
- `build-install`
- `run-cpp-tests`
- `run-python-tests`
- `run-js-tests`
- `collect-cpp-coverage`
- `write-summary`
- `package-artifacts`

## Config-Driven Test Lists
- `config/tests_cpp.yml`
- `config/tests_python.yml`
- `config/tests_js.yml`

Each file uses the same top-level schema:
- `schema_version`
- `suite`
- `tests`

Profile-aware fields are supported (for example C++ `args` maps for CPU vs GPU profiles).

## Local Usage
Run full flow in CPU profile:

```bash
python3 scripts/coverage/coverage.py run-all --profile cpu
```

Run with dependency installation:

```bash
python3 scripts/coverage/coverage.py run-all --profile cpu --install-deps
```

Run subset:

```bash
python3 scripts/coverage/coverage.py run-all --from run-cpp-tests --to package-artifacts
```

Validate config files:

```bash
python3 scripts/coverage/coverage.py validate-config
```

## Notes
- Supported profiles: `cpu`, `cpu_gpu`, `cpu_npu`, `cpu_npu_gpu`.
- Local mode writes emulated GitHub outputs to `.tmp/coverage-local/`.
