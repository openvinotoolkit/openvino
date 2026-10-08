---
name: ov-validation-matrix
description: Run or resume long validation batches, including unit and integration suites, frontend/backend compatibility, model accuracy, and performance checks. Use for repeated test/configuration matrices in OpenVINO or related repositories; not for implementing fixes or debugging a single failure.
---

# Validation matrix

Keep test execution and logs outside the main conversation. The main agent owns
the acceptance contract and product changes; a validation worker runs bounded
batches against a stable build and reports evidence.

## Prepare the handoff

Before execution, record:

- Repository paths and revisions, build/install paths, interpreter or toolchain,
  dependency versions, and relevant runtime settings. Verify which binaries and
  libraries the tests actually load.
- Test selection, configurations, devices where applicable, and fixture/dataset
  revisions. For comparisons, record the baseline, metric, tolerance, and seeds.
- Required coverage and acceptance criteria: expected test counts/skips, numerical
  thresholds, or performance limits as applicable. Passing one suite or
  configuration does not establish coverage for the others.
- Commands, per-case timeout, artifact directory, and the permitted scope of
  setup/downloads. Use the existing test harness wherever possible.

Use the baseline and criteria chosen for the task. Never silently substitute a
reference, relax a threshold, reduce coverage, or mark a failure as expected.
For performance comparisons, preserve hardware, load, warmup, and repetition
settings; noisy measurements do not establish a regression or improvement.
For GGUF model batches, also read [GGUF validation details](references/gguf.md).
For incremental builds, runtime overlays, or multiple worktrees, read
[build and runtime identity](references/build-runtime.md) before numerical diagnosis.

Agree the required matrix early, then iterate on a small reproducer and its regression
test. Run the broader batch after a fix touches a shared path and one final batch after
the sources and runtime are stable. Expand or repeat it only for a new failure, changed
inputs, or a previously missing required scenario. Record excluded coverage explicitly.

## Delegate a bounded batch

When subagents are available, delegate long execution batches to one validation
worker using the user's configured model preference. Use `ov-validation-runner`
if configured; otherwise use a worker without prescribing a provider or model.
Give it this skill's absolute path, the handoff above, and the artifact paths.
Start with a fresh, compact brief instead of copying the implementation history.
If subagents are unavailable, follow the same procedure in the current agent.

Use a dedicated worktree/build or hold the validated source and build stable.
Do not let the main agent edit or rebuild the same inputs during validation.
The worker may write test artifacts and perform authorized setup, but must not
edit product code or acceptance criteria. Return missing prerequisites to the
main agent instead of starting an open-ended repair investigation.

## Run and resume

Read [the runner contract](references/runner.md) when preparing a batch. The
standard-library [batch runner](scripts/run_matrix.py) executes a JSON manifest,
writes one log per attempt, and checkpoints after each case. It runs serially
to avoid competing for CPU, device memory, or benchmark resources; an existing
harness may manage explicitly budgeted concurrency inside a case.

Put manifests, logs, and checkpoints outside the source tree, for example in
`~/.cache/ov-validation/<task>/`. Never commit downloaded datasets or run artifacts.
List all relevant build binaries, configuration, fixtures, and other inputs in
the manifest so content changes invalidate reuse. The runner also fingerprints
tracked source changes. Untracked source dependencies must be explicit inputs.
For build-validation cases, fingerprint source and build configuration as inputs,
not the outputs the case is expected to create. For tests of an existing build,
fingerprint the tested binaries and keep them stable throughout the batch.
Include build wrappers and untracked source files that participate in the build.

Inspect an existing output directory's summary before choosing to resume it. Use a new
directory for a different acceptance contract or experiment; resume the same experiment
deliberately. Give each case unique report paths. Optional `reports` contracts let the
runner check gtest counts/skips and GGUF modality metrics and preserve each attempt's
XML/JSON beside its log. Prefer these checks when the harness can exit zero with missing
coverage; prose in `metadata` is not an executable assertion.

Use one long-running command per batch and let the runner perform the loop.
Check its summary file for progress; do not poll once per test or paste raw logs
into chat. A completed failure is retained on resume. Retry only after the main
agent chooses a fix or a justified rerun; `--retry-failed` explicitly retries it.
Changed inputs invalidate prior results. Interrupted and stale cases never count
as passes. Stop on a user pause and retain checkpoints for completed cases.

## Report to the main agent

Return tested revisions/build identity, completed and remaining counts, and a
compact suite/configuration result table. Include failed case IDs, observed failure
signatures, a short supporting excerpt, reproduction commands, and artifact
paths. Distinguish harness exit status from established acceptance: the command
must enforce the agreed criteria, and missing or unexpected skipped coverage is a gap.
Quote selected test counts and skips separately from suites never registered by the
environment or filtered out. A standalone fixture adapter establishes only the test
bodies it runs; disclose the missing full harness. Tiny/random models test numerical
contracts, not natural-language quality. A known arithmetic difference remains a failed
parity check unless the user explicitly changes the acceptance contract.

Group equivalent observed errors without asserting a shared root cause. Escalate
unexpected results, source/build mismatches, missing scenarios, and proposed
acceptance changes. Let the main agent select fixes and subsequent batches.
Report useful checkpoints from the existing summary without an extra inference. Once
all requested batches and final input audits finish, tell the parent execution is done;
only then may it alter those repositories or binaries for final integration and push.
