# Executable validation preflight

Use these helpers for repeated batches or fragile overlays, alongside the existing test
harness. They check coverage and link inputs; numerical gates remain in the tests.
Save every attempt outside the checkout. Run scripts with Python 3.9 or newer.
In the examples, `skill_scripts` points to this skill's `scripts/` directory and
`validation_reports` to the task's artifact directory outside the checkout.

## Compiler warning policy

Match the relevant CI warning configuration when compiling changed frontend code and tests,
including `CMAKE_COMPILE_WARNING_AS_ERROR` where enabled. Record the compiler and effective
flags; a successful run on one toolchain does not establish another's buildability. Brace
conditional assertion macros and use checked or known-safe integer conversions in tensor writes.

## Test selection and required fixtures

Capture `--gtest_list_tests` from the actual binary with the same filter and environment
as the subsequent run. Check its exit code before using the listing. Define a contract
with exact IDs or glob patterns (`*`, `?`, `[abc]`), required counts and absolute fixture paths:

Example contract; replace the patterns, counts and paths with the requested scenarios:

```json
{
  "selections": [
    {"pattern": "*", "count": 2},
    {"pattern": "DecoderChecks.PrefillAndCachedDecode", "count": 1},
    {"pattern": "EmbeddingChecks.TokenLookup", "count": 1}
  ],
  "required_files": ["/task/fixtures/decoder.npz"],
  "allowed_skips": []
}
```

Use counts established from the requested scenarios, not counts copied from an incomplete
listing. Include a `*` selection when the total must be exact. Overlapping selections are
independent checks. Allowed skips apply only to optional coverage; required scenarios that
skip remain incomplete. File presence does not establish fixture validity or model identity.

```bash
python "$skill_scripts/check_test_coverage.py" --contract "$validation_reports/coverage.json" \
    --listing "$validation_reports/selected-tests.txt" --output "$validation_reports/preflight.json"
```

After execution, retain the test process exit code and summarize its XML using the helper
from this skill revision. The JSON includes selected IDs, excluding suppressed/notrun
cases. GTest IDs are `classname.name`; pytest IDs use the XML's corresponding names.

```bash
python "$skill_scripts/summarize_tests.py" "$validation_reports/tests.xml" \
    --output "$validation_reports/test-summary.json" --process-exit 0
python "$skill_scripts/check_test_coverage.py" --contract "$validation_reports/coverage.json" \
    --listing "$validation_reports/selected-tests.txt" --summary "$validation_reports/test-summary.json" \
    --output "$validation_reports/coverage-result.json"
```

Replace `0` with the recorded process exit code. Summary-only checks can be used for pytest
with contracts derived from its XML IDs. With both inputs, listed and executed ID multisets
must match. Missing cases, duplicate cases, unexpected skips, suite/test failures and missing
or nonzero process status fail the coverage check. It does not replace runtime identity or
prove that a stale XML file belongs to the attempt: use unique paths and preserve input hashes.

## Overlay link inputs

Prefer normal CMake/Ninja targets. If an overlay is necessary, save its actual single
compiler-driver link command using POSIX shell quoting in a text file. List the required replacement objects and
superseded objects that must be absent:

```json
{
  "required_objects": ["/task/overlay/translation-unit.o"],
  "forbidden_objects": ["/task/base/translation-unit.o"]
}
```

```bash
python "$skill_scripts/audit_link.py" --command-file "$validation_reports/link-command.txt" \
    --directory /task/build --contract "$validation_reports/link-contract.json" \
    --output "$validation_reports/link-audit.json"
```

The helper reads but never executes the command. It expands response files relative to the
link working directory, canonicalizes object paths, rejects duplicate/missing/forbidden
objects and refuses shell command sequences or unresolved shell expansions (parameters,
globs, braces and tildes). Record expanded arguments; the helper never evaluates shell
expressions. Single-quote or escape literal shell characters, for example `'$ORIGIN'` in
a linker rpath. Response-file contents are parsed as literal arguments without shell
expansion. Supply the inner compiler command when a
wrapper prints `cd ... && ...`; do not pass a build script. Full findings stay in JSON;
the printed summary is bounded. A passing link audit does not prove objects are freshly
compiled, dependency-complete or present inside archives. Rebuild all translation units
affected by headers/macros using build dependencies, then verify new case registration
and [runtime identity](build-runtime.md) separately.

Both checks return nonzero on violated contracts or malformed inputs. The XML summarizer
returns parsing status only. Exercise the helpers with:

```bash
python -m unittest discover -s "$skill_scripts/tests" -v
```
