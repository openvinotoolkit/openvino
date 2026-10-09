# Batch runner contract

Use this optional helper for long GGUF validation batches. It launches existing test
harnesses in manifest order; the report kind selects the checks applied afterward.
There is no built-in test list: the manifest selects all commands, models and backends.
The runner handles execution and checkpoints; [matrix_reports.py](../scripts/matrix_reports.py)
handles GTest and GGUF report validation.

Run with Python 3.9+ on Linux:

```bash
python3 /path/to/openvino/.claude/skills/ov-gguf/scripts/run_matrix.py \
  /absolute/path/matrix.json --output /absolute/path/results
```

Add `--retry-failed` for a deliberate retry of completed failures. Only one runner
may use an output directory at a time. Exit codes: 0 means all commands and configured
report checks passed, 1 means a case failed/timed out/could not start, 2 means invalid setup or
changed inputs, and 130 means interrupted. Logs stay on disk; console output is
one summary per completed case. `summary.json` can be read while the batch runs.
`summary.json.cases` maps case IDs to records; it is not an array. Inspect its
provenance before interpreting counts in an existing directory.

## Manifest

The following is illustrative; replace paths and the harness with actual ones.
All filesystem paths must be absolute. Commands are argument arrays, executed
without a shell. Use a wrapper file as an input if shell setup is necessary.

```json
{
  "version": 1,
  "context": {
    "build": "Release; built from the recorded OpenVINO sources",
    "device": "CPU; record CPU model and driver/runtime versions here"
  },
  "repositories": ["/work/openvino"],
  "inputs": [
    "/work/build/CMakeCache.txt",
    "/work/install/lib/libopenvino.so",
    "/work/install/lib/libopenvino_intel_cpu_plugin.so"
  ],
  "cases": [
    {
      "id": "frontend-integration-cpu",
      "cwd": "/work/openvino",
      "command": ["/work/venv/bin/python", "/work/check_suite.py", "--suite", "frontend-integration", "--device", "CPU"],
      "env": {"OMP_NUM_THREADS": "8"},
      "timeout_seconds": 1800,
      "inputs": ["/work/check_suite.py", "/fixtures/test_cases.json"],
      "metadata": {
        "suite": "frontend integration; record exact test selection",
        "configuration": "CPU; record relevant dependency versions and runtime settings",
        "acceptance": "all selected tests pass; check_suite.py rejects empty selection and unexpected skips"
      },
      "reports": [{
        "path": "/work/results/frontend-tests.xml",
        "kind": "gtest",
        "expected_tests": 96,
        "max_skips": 0
      }]
    }
  ]
}
```

`context` is required descriptive provenance. `repositories` must contain at
least one checkout; include relevant submodules as separate entries if validating
local changes inside them. `inputs` lists files whose contents affect every case;
per-case `inputs` restrict invalidation to affected cases. Include test wrappers,
untracked source dependencies, fixtures/datasets, baselines,
configuration, and all runtime binaries relevant to the claim. Files are hashed
once per invocation; hashing large inputs has an initial I/O cost.

Each case requires a unique filename-safe `id`, existing `cwd`, nonempty command,
positive timeout, and nonempty `metadata` recording the acceptance contract.
`env` is optional and augments the inherited environment. The environment is
hashed for invalidation but not copied into artifacts; do not put secrets into
the manifest. Environment changes conservatively invalidate reuse.

## Test types

Each case supplies its executable, arguments, environment and timeout. There are three
paths; tests are not regrouped or reordered by type.

### Command-only checks

For builds, Python tests, loading checks or other self-checking harnesses, omit `reports`.
The command's exit status determines success; the harness must enforce its own coverage
and acceptance criteria. The runner has no separate pytest or performance-report parser.

### GTest unit and integration tests

Run the test executable with `--gtest_output=xml:<report-path>` and configure a `gtest`
report. `validate_gtest_contract()` checks the manifest settings;
`audit_gtest_report()` checks the fresh XML for expected counts, skips and failures.
Both functions are in `matrix_reports.py`.

### GGUF GenAI accuracy and API tests

Run GenAI's `validate_gguf_mmproj.py` with `--report <report-path>` and configure a
`gguf_mmproj` report. `validate_gguf_mmproj_contract()` checks the settings;
`audit_gguf_mmproj_report()` verifies modalities, replay choices, API checks and provenance
fields. Choose the reference and numerical thresholds explicitly.

## Report contracts

`reports` are case outputs, not stable `inputs`. Each report needs a unique absolute
path across the matrix, and the command must write it. Missing or unchanged reports
fail the case even if the command returns zero. Each fresh report is copied beside the
attempt's log and hashed; later runs overwriting the original path do not destroy it.

`gtest` counts actual `testcase` elements and rejects failures/errors, a count differing
from positive `expected_tests`, or more than `max_skips` (default zero). This does not
discover filtered-out or unregistered tests: put that required coverage in the contract.

`gguf_mmproj` audits GenAI's `validate_gguf_mmproj.py` JSON format. Set the task's thresholds
explicitly; the runner supplies no default numerical acceptance:

```json
{
  "path": "/work/results/unified-pa.json",
  "kind": "gguf_mmproj",
  "first_token_matches": true,
  "min_choice_fraction": 0.9,
  "required_modalities": ["text", "image", "video", "multi_image", "image_chat", "modern_image", "modern_image_chat"],
  "required_api_checks": ["modern_image_chat", "beam_search", "cancel_and_reset"],
  "equals": {
    "reference_revision": "the-selected-pinned-llama-commit",
    "attention_backend": "PA",
    "kv_cache_precision": "f16",
    "request_reset_matches": true,
    "chat_reset_matches": true
  }
}
```

The report must be completed and passed. Required modalities/API checks must exist;
all emitted modalities must meet the thresholds and all emitted API checks must pass.
Greedy scores are recomputed from nonempty equal-length token/choice histories, including
first-token equality; inconsistent or nonfinite reported metrics fail. `equals` verifies
selected top-level provenance/settings, but cannot prove the oracle really ran that
revision or loaded the intended libraries. Check reference and runtime identity separately.

Failures keep their process return code and report audit errors. A failed process may
still produce useful report snapshots. Without `reports`, `passed` retains its original
meaning of process exit zero.

## Reuse and limitations

The runner fingerprints both Python modules, the manifest context, repository
HEADs and tracked diffs, file contents, and execution environment. It records
the actual command and provenance in each checkpoint. A changed case command,
metadata, timeout, or inputs invalidates that case; shared changes invalidate all
cases. Unchanged completed failures are reported without rerunning by default.
Attempt metadata is retained as `<case>-<timestamp>.json`, alongside its log and report
snapshots. Before refreshing an existing summary the runner archives it as
`summary-<timestamp>.json`. Per-case `<case>.json` and `summary.json` remain the current
view. Reports whose saved snapshot is missing or changed invalidate reuse.

It checks repository state and input size/mtime/ctime before and after each
case, and stops if they change. This detects ordinary concurrent edits but is
not a sandbox or an atomic snapshot. Use a stable build/worktree. It cannot prove
that installed binaries were built from the recorded sources, discover omitted
dependencies, or track remote service changes. The worker must verify those
conditions. Use a fresh output directory when provenance is uncertain.

Without report contracts, `passed` means only that the process exited zero. With
contracts, every report audit must also pass. The harness must fail on other unmet
metrics and unexpected skips; inspect its report before claiming scenario coverage.
Completed results survive interruption. A timeout or interrupt terminates the
case's process group. Failed logs contain raw diagnostics for the worker to
classify; the runner does not infer root causes or alter acceptance criteria.
