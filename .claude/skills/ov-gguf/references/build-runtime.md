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
- If upstream was merged, account for added translation units and changed dependency
  headers in the overlay. State which unchanged base objects are reused. Do not call an
  overlay a full clean build or treat an old embedded version string as the new commit.

After an actual inference, record resolved module paths and loaded core/frontend/plugin
libraries. On Linux, `/proc/self/maps` from that process shows plugins loaded lazily;
`ldd` on the executable alone does not. Hash the actual files and put those paths in the
validation record, along with the build wrapper and explicit untracked source dependencies.
Separate the OpenVINO runtime from the independent reference runtime.

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
