# Build and runtime identity

Use this when a validation run depends on multiple worktrees, an incremental rebuild,
an overlay install, or a dispatched plugin kernel. A source revision and version string
alone do not identify the executed implementation.

## Before diagnosing numerical output

Record the intended source worktree, base build, compiler/configuration, build wrapper,
and output directory. Prefer normal CMake/Ninja targets in an isolated build. If an
overlay is necessary, check the actual compile and link commands:

- Rebuild every translation unit affected by changed headers and macros. Include
  generated or cross-compiled instances of a source, such as ANY, AVX2 and AVX512 CPU
  executors; compiling only the ordinary source object may leave every dispatched
  implementation stale. Derive the set from `compile_commands.json` and target commands.
- Verify replacement objects occur in the final link, output is written into the
  overlay, and symlinks resolve inside it. Writing through a symlink can replace a base
  library. Inspect exact symlink targets before removing an overlay link.
- Use the [link audit](preflight.md#overlay-link-inputs) to require replacement objects exactly
  once and reject superseded objects. Verify added test IDs in the final binary; compilation
  alone does not establish that a test was linked.
- If upstream was merged, account for added translation units and changed dependency
  headers in the overlay. State which unchanged base objects are reused. Do not call an
  overlay a full clean build or treat an old embedded version string as the new commit.

## Runtime preflight

Before a broad test or checkpoint batch, preflight the runtime and prerequisites selected
by its actual commands. Check frontend discovery, each requested device (including a default
device wrapper when used), tokenizer extensions, fixture registration and required media/model
files. Missing optional data must produce explicit skips; missing required data blocks that
scenario. Do not require unused devices or export/download unrelated models.

Run a small conversion/inference through the same harness and launch
environment. Record intended core/frontend/plugin paths and hashes, then check the libraries
actually loaded after inference. On Linux, `/proc/self/maps` shows plugins loaded lazily; `ldd`
and pre-run hashes alone do not establish execution identity. Fix stale frontend or missing plugin
discovery before launching the batch. Reuse the preflight result only while its inputs and launch
environment remain unchanged; retain runtime identity from the actual validation process too.

Use [runtime_identity.py](../scripts/runtime_identity.py) in the inference process, or with
`--pid` while that process is alive after inference. Supply expected core/frontend/plugin paths
and SHA-256 values captured before launch:

```json
{
  "core": {"path": "/task/runtime/<resolved-core-library>", "sha256": "<expected SHA-256>"},
  "frontend": {"path": "/task/runtime/libopenvino_gguf_frontend.so", "sha256": "<expected SHA-256>"},
  "cpu": {"path": "/task/runtime/libopenvino_intel_cpu_plugin.so", "sha256": "<expected SHA-256>"}
}
```

```python
from runtime_identity import record_identity

# Call after inference, while the frontend, core and compiled model remain alive.
record_identity(expected, report_path)
```

The helper writes actual mapped paths/hashes and fails when an expected library is absent or
mismatched. Add its directory to the harness's Python path, or invoke it from another process:

```bash
python runtime_identity.py --pid <inference-pid> --expected expected-runtime.json --output runtime.json
```

For overlays, check frontend discovery as well as `LD_LIBRARY_PATH`: a core loaded from the base
directory can cause the frontend manager to scan that directory. Prefer a verified task-local
runtime directory. Record any Linux `LD_PRELOAD` workaround explicitly; it is not a general default.
Keep the report alongside build commands and untracked source dependencies. Separate the OpenVINO
runtime from the independent reference runtime.

## When a pass or plugin fix seems ineffective

Check the chain in order: the converted graph contains the intended attribute; the
compiled plugin receives it; the selected executor receives it; the changed kernel
variant is linked and loaded. Do this before a repeated full accuracy matrix.

For SDPA-to-PA changes, inspect each layer's window, token-type inputs and runtime flags,
including global layers. Test the flag's default and enabled behavior and its executor
cache key. A flag reaching the plugin does not prove the new executor is running.

## Freeze, integrate, report

Keep sources, build inputs and binaries stable for each delegated batch. Fetching remote
refs is read-only with respect to tracked files, but merges, commits and rebuilds alter
the tested inputs: wait for validation to finish. Integrate required upstream changes
before final validation where practical. If another merge before push changes tested
dependencies, rebuild and run the checks justified by those changes.

Use one short artifact inventory for a long task: branch heads, wrapper paths, resolved
binary hashes, active process/session, manifests, outputs, failures and next reproducer.
After compaction, resume from that inventory and the current summary rather than
rebuilding or rerunning completed work. Keep separate attempts and fixture provenance.
