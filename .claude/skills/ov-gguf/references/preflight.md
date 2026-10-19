# Executable validation preflight

Use these helpers for repeated batches or fragile overlays, alongside the existing test
harness. They check coverage and link inputs; numerical gates remain in the tests.
Save every attempt outside the checkout. Run scripts with Python 3.9 or newer.

## Test selection and required fixtures

Capture `--gtest_list_tests` from the actual binary with the same filter and environment
as the subsequent run. Check its exit code before using the listing. Define a contract
with exact IDs or glob patterns (`*`, `?`, `[abc]`), required counts and absolute fixture paths:

```json
{
  "selections": [
    {"pattern": "*", "count": 2},
    {"pattern": "Architectures/GGUFArchitectureAccuracy.PrefillAndCachedDecodeMatchLlamaCPU/qwen3next", "count": 1},
    {"pattern": "GGUFEmbeddingAccuracy.RealCheckpointTokenEmbeddingsMatchLlamaCPU", "count": 1}
  ],
  "required_files": ["/task/models/embedding.gguf"],
  "allowed_skips": []
}
```

Use counts established from the requested scenarios, not counts copied from an incomplete
listing. Include a `*` selection when the total must be exact. Overlapping selections are
independent checks. Allowed skips apply only to optional coverage; required scenarios that
skip remain incomplete. File presence does not establish fixture validity or model identity.

```bash
python check_test_coverage.py --contract coverage.json --listing selected-tests.txt --output preflight.json
```

After execution, retain the test process exit code and summarize its XML using the helper
from this skill revision. The JSON now includes selected IDs, excluding suppressed/notrun
cases. GTest IDs are `classname.name`; pytest IDs use the XML's corresponding names.

```bash
python summarize_tests.py tests.xml --output test-summary.json --process-exit 0
python check_test_coverage.py --contract coverage.json --listing selected-tests.txt --summary test-summary.json --output coverage-result.json
```

Replace `0` with the recorded process exit code. Summary-only checks can be used for pytest
with contracts derived from its XML IDs. With both inputs, listed and executed ID multisets
must match. Missing cases, duplicate cases, unexpected skips, suite/test failures and missing
or nonzero process status fail the coverage check. It does not replace runtime identity or
prove that a stale XML file belongs to the attempt: use unique paths and preserve input hashes.

## Overlay link inputs

Prefer normal CMake/Ninja targets. If an overlay is necessary, save its actual single
compiler-driver link command in a text file and list the required replacement objects and
superseded objects that must be absent:

```json
{
  "required_objects": ["/task/overlay/eagle-test.o"],
  "forbidden_objects": ["/task/base/eagle-test.o"]
}
```

```bash
python audit_link.py --command-file link-command.txt --directory /task/build --contract link-contract.json --output link-audit.json
```

The helper reads but never executes the command. It expands response files relative to the
link working directory, canonicalizes object paths, rejects duplicate/missing/forbidden
objects and refuses shell command sequences. Supply the inner compiler command when a
wrapper prints `cd ... && ...`; do not pass a build script. Full findings stay in JSON;
the printed summary is bounded. A passing link audit does not prove objects are freshly
compiled, dependency-complete or present inside archives. Rebuild all translation units
affected by headers/macros using build dependencies, then verify new case registration
and [runtime identity](build-runtime.md) separately.

Both checks return nonzero on violated contracts or malformed inputs. The XML summarizer
returns parsing status only. Exercise the helpers with:

```bash
python -m unittest discover -s scripts/tests -v
```
