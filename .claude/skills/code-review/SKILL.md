---
name: code-review
description: Review OpenVINO changes for correctness, compatibility, performance, security, testing, and maintainability. Use when reviewing a pull request, branch, commit, or proposed diff.
---

# OpenVINO Code Review

## Purpose

Provide a concise, evidence-based review of the proposed changes and their
directly affected execution paths. Prioritize functional correctness, safety,
compatibility, and material performance impact.

Do not perform a broad audit of unrelated legacy code. Keep the report concise
without shortening the investigation.

## Review workflow

### 1. Establish the baseline

Identify exactly which changes are being reviewed:

- **Pull request:** record the base and head SHAs and inspect the current PR
  diff.
- **Committed branch changes:** compare the merge base of the selected target
  and `HEAD` against `HEAD`. Record the comparison SHAs. Do not silently include
  uncommitted changes.
- **Specific commit:** compare against its parent or the explicitly selected
  baseline. State the baseline used for merge commits.
- **Uncommitted changes:** state whether the review covers staged changes,
  unstaged changes, or both. Distinguish index content from working-tree
  content.

Read the description and relevant linked requirements.

For local reviews, mark PR-only checks and discussion as not applicable.
Report local validation separately.

### 2. Gather relevant evidence

Read the diff, relevant surrounding code, and the component references selected
below.

For PR reviews:

- Inspect discussion, review summaries, and relevant inline threads for failure
  reports, constraints, and decisions affecting the changed behavior.
- Treat comments and author explanations as leads, not proof, including later
  replies.
- Classify material existing concerns as still applicable, fixed,
  outdated/inapplicable, or not yet verifiable at the reviewed head.
- Support that classification with current code, tests, or other direct
  evidence. An unresolved thread does not prove a defect remains; a resolved
  thread does not prove the implementation is correct.
- When reporting validation or merge status, inspect actual check conclusions,
  not just completion status. Distinguish passing, failing, pending, skipped,
  and unavailable results.

Prioritize final-head source and directly relevant tests or results over older
comments and summaries.

Passing compilation or CI establishes only the behavior and configurations
actually exercised. Approval is not technical proof of correctness. Do not
infer repository approval requirements when policy is unavailable.

### 3. Map scope and coverage

Identify affected components, input contracts, public APIs, execution paths,
shared infrastructure, and expected validation.

Before detailed inspection, enumerate materially different paths introduced or
changed by the diff. Track coverage internally using a compact list or matrix:

- path or variant and its triggering condition;
- inspected, partially inspected, or uninspected;
- relevant supporting test or measurement, if any;
- unsupported combinations and the guard that rejects them.

Track source inspection separately from validation. Reading a test is not
evidence that it ran, and a passing test does not prove every implementation
variant was exercised.

For cross-layer or multi-variant changes, trace:

input/API or transformation -> descriptor/state -> implementation selection
-> generated code or arguments -> execution -> fallback/rejection -> tests.

Inspect every materially affected variant, or verify a shared invariant that
covers it. Do not infer correctness from a similarly structured variant.

When evidence is truncated or missing:

- Recover it using targeted file reads, smaller queries, or pagination.
- Do not stop at an arbitrary file count or treat the first response as the
  complete change.
- Prioritize shared code and high-impact execution paths if resources are
  limited.
- If recovery is unavailable, state the specific coverage limitation. Do not
  present a partial review as complete.

### 4. Investigate changed behavior

Review functional correctness and safety first, then compatibility,
performance, test coverage, and maintainability.

Inspect directly affected callers and consumers, not only isolated changed
lines. Distinguish newly enabled behavior from changes to existing behavior.

Use the priority checklist and applicable component references below.

### 5. Challenge each candidate finding

Before reporting a defect:

1. Verify the source at the reviewed revision, distinguishing source content
   from diff markers, line numbers, and review annotations.
2. Identify the triggering input or configuration and a reachable execution
   path. Do not invent unsupported usage or undocumented requirements.
3. Identify the incorrect operation, violated invariant, or explicit mandatory
   requirement.
4. Check counterevidence: caller guarantees, validation, initialization,
   serialization, guards, fallback paths, tests, and existing contracts.
5. Explain the contribution of this change. Do not attribute an unchanged
   pre-existing issue to the PR.
6. State the impact and a minimal correction or narrowly defined verification
   action.

If these checks do not establish a defect, do not report one as confirmed.

### 6. Finalize

Recheck the reviewed revision and the evidence supporting each finding.

Report all material, well-supported, actionable findings in descending order of
impact. Consolidate repeated instances of one root cause.

## Priority checklist

### Correctness and compatibility

- Check shapes, ranks, element types, attributes, broadcasting, layouts,
  offsets, pitches, and padding.
- Include relevant dynamic, partially known, scalar, empty, boundary, and tail
  cases.
- Trace state updates, initialization, serialization, cache import/export, and
  error handling.
- Check consistency between selection conditions, descriptors, generated
  constants, argument binding, and execution.
- Verify optimized paths, fallback paths, and explicit rejection of unsupported
  combinations.
- Check backward compatibility for public APIs and serialized models.
- Check consistency of generated code, bindings, documentation, registrations,
  and implementations when contracts change.
- Check relevant platform, device, and runtime differences.

### Safety and concurrency

- Treat model metadata, file contents, and user-provided values as untrusted.
- Check bounds, allocation arithmetic, overflow, signed/unsigned conversions,
  and narrowing before unsafe arithmetic or memory access.
- Check ownership, lifetime, aliasing, memory reuse, and resource cleanup.
- Trace event dependencies, barriers, waits, and accesses to shared or cached
  mutable state.
- Consider multi-request and multi-stream execution when state can be shared.
- Report a race only after identifying shared state, concurrent access, and
  missing synchronization.
- Prefer explicit rejection of invalid input before unsafe execution.

### Performance and resource use

Look for a concrete mechanism:

- added copies, reorders, conversions, transfers, or full-tensor memory passes;
- additional launches, synchronization, or host-blocking waits;
- hot-path allocation or repeated invariant work;
- larger or longer-lived temporary storage;
- lost fusion, reduced parallelism, or slower implementation selection;
- increased compilation, loading, first-inference, or cache-import/export cost.

Trace allocation formulas against actual indexing bounds, supported workloads,
buffer lifetime, and memory reuse.

Separate kernel time, end-to-end latency/throughput, compilation/loading,
first inference, cache operations, and peak memory consumption.

For newly enabled paths, assess absolute resource use and expected performance.
For changed existing paths, seek comparison against the baseline under relevant
workloads and hardware.

A source-level mechanism may establish added work or allocation growth without
establishing a measurable regression. State that distinction. Do not claim a
measured slowdown or speedup without measurements.

### Tests and validation

- Expect regression coverage for fixes and behavioral changes when an existing
  suite can express the scenario.
- Check whether equivalent coverage already exists before requesting new tests.
- Request specific validation: identify the scenario, expected invariant, and
  what result would resolve the concern.
- Consider CI and build configuration requirements when relevant to the changed
  surfaces or explicitly required by repository policy.

### Architecture and maintainability

- Check responsibility boundaries, reuse of established abstractions,
  ownership, and lifetime.
- Flag duplication or complexity only when it creates a concrete correctness,
  compatibility, performance, or substantial maintenance risk.
- Check that hardware/runtime workarounds explain their constraint and include
  a tracking reference or removal condition where appropriate.
- Separate architectural or product trade-offs from implementation defects.
  Ask a focused question when the necessary intent is unavailable.
- Avoid unrelated refactors, speculative abstractions, and style preferences
  enforced by tooling.

## Component guidance

Load references relevant to changed behavior, not just file location. Paths are
relative to the repository root and include tests within each component.

Apply multiple references for cross-component changes. Select guidance for
shared tests and documentation according to the behavior exercised or described.

| Changed paths or behavior | Reference |
| --- | --- |
| `src/core/`, `src/inference/` | [Runtime and core](references/runtime.md) |
| `src/common/transformations/`, `src/common/low_precision_transformations/`, `src/common/offline_transformations/`, `src/common/snippets/`, `src/core/src/pass/`, `src/core/src/pattern/`, `src/core/include/openvino/pass/` transformation passes and tests under frontends or plugins, `src/bindings/python/tests/test_transformations/` | [Transformations](references/transformations.md), plus owning-component guidance |
| `src/plugins/intel_gpu/`, `src/inference/include/openvino/runtime/intel_gpu/`, Intel GPU tests, kernels, selection, runtime, memory, layouts, caching, precision, or platform enablement | [Plugins](references/plugins.md); also apply [Bindings and public APIs](references/bindings.md) for public API or user-visible contract changes |
| `src/plugins/`, including plugin-local tests | [Plugins](references/plugins.md) |
| `src/tests/functional/` shared operator and inference behavior tests | [Runtime and core](references/runtime.md) and [Plugins](references/plugins.md), as exercised |
| `src/frontends/`, `src/bindings/python/src/openvino/frontend/`, `src/bindings/python/src/pyopenvino/frontend/`, framework conversion tests in `tests/layer_tests/` and `tests/model_hub_tests/` | [Frontends](references/frontend.md); also apply [Bindings and public APIs](references/bindings.md) for Python API or language-boundary changes |
| `src/bindings/`, public API changes under `src/core/include/` or `src/inference/include/` | [Bindings and public APIs](references/bindings.md) |
| `.github/workflows/`, `.github/actions/`, `.github/scripts/`, `.github/dockerfiles/`, Smart CI and dependency configuration under `.github/`, `cmake/`, `CMakeLists.txt` and `*.cmake` at any depth, dependency manifests and integration changes | [CI, build, and dependencies](references/ci.md) |

This skill defines the common evidence and reporting contract. Component
references add domain-specific checks and explicit requirements.

Distinguish:

- **Mandatory requirements:** explicit implementation or compatibility rules.
  Report a verified violation with the cited rule, applicability, and practical
  consequence. If impact is unknown, say so.
- **Investigative prompts:** instructions to inspect, trace, compare, or measure.
  Failure to find evidence is not automatically a code defect.
- **Recommendations:** preferences that require a material consequence before
  becoming actionable findings.

If a referenced file is unavailable, disclose the limitation and use available
applicable guidance without pretending the missing reference was loaded.

### Evidence checks before finalizing

- Before claiming tests, guards, replies, or handling are absent, search the
  relevant scope. Otherwise state "not verified" and identify the limitation.
- Check producer/caller contracts before proposing defensive checks. Do not
  silently discard invalid input unless that behavior satisfies the contract.
- Verify that a proposed fix preserves semantics. Do not describe an alternative
  as faster, cheaper, or negligible-cost without supporting evidence.
- Distinguish duplicate registrations from necessary related dependencies, and
  historical reviewer preferences from current technical requirements.

## Conditional review modes

### Revert pull requests

Treat a revert as a regression review:

- Identify the original change and the behavior or risk it addressed.
- Check whether the revert reintroduces that problem or changes compatibility.
- Do not request new tests or refactors for restored code unless needed to
  prevent a concrete regression.

### External contributions

Apply the same technical evidence threshold regardless of authorship:

- Prioritize correctness, buildability, changed-path risk, and relevant required
  checks.
- Identify unrelated changes that introduce material scope or review risk.
- Do not request redundant local validation when passing checks demonstrably
  cover the scenario.
- Distinguish change-induced failures from infrastructure, resource, network,
  or pre-existing failures.
- Do not infer lower quality from external authorship or AI assistance.

### Merged or closed pull requests

- State the PR status when it affects interpretation.
- For merged changes, describe findings as follow-up defects, validation needs,
  or questions rather than current merge blockers.
- Verify the reviewed revision explicitly. Do not assume later fixes are
  included unless inspected.
- Do not present historical decisions or failures as current repository state
  without current evidence.

## Review output

Start with the main assessment and identify the reviewed baseline. State
whether inspection was complete or partial.

### Defect findings

For each material defect, include:

- exact location, linked to the reviewed revision where possible;
- triggering condition and affected scope;
- failure mechanism and practical impact;
- evidence and relevant counterevidence;
- minimal correction direction.

Do not require a long repeated template when a short paragraph contains
the necessary evidence.

### Validation gaps and questions

Keep these separate from demonstrated defects:

- **Reported failure:** cite the result and environment; distinguish observed
  symptoms from an unverified root cause.
- **Validation gap:** identify the concrete uncertainty and the test,
  measurement, or invariant needed to resolve it.
- **Design question:** identify the missing requirement or owner decision and
  explain why it matters.

Do not disguise unsupported allegations as questions or generic test requests.

### Severity and confidence

Severity describes impact; confidence describes evidential strength.

- **Critical:** demonstrated severe unsafe execution, broad corruption, or
  comparable impact on a reachable supported path.
- **High:** demonstrated or strongly supported incorrect results, crashes,
  compatibility failure, or major resource/performance impact on a reachable
  supported path.
- **Medium:** material localized impact or a technically grounded unresolved
  risk requiring focused verification.
- **Low:** meaningful non-blocking maintenance or diagnostic improvement.

Do not assign high or critical severity solely because code is complex,
validation is missing, or a checklist was not completed.

Qualify uncertainty explicitly. Do not label a suspected failure as confirmed
or use low-confidence speculation as a blocking defect.

### Review summary table

End the review with a concise table covering material findings and significant
inspection or validation gaps. Use one row per root cause or material review
area; omit minor observations and inapplicable areas.

| Area | Assessment | Evidence / scope | Next action |
| --- | --- | --- | --- |

Use these assessments:
- **Confirmed defect** — a reachable defect is established at the reviewed revision.
- **Reported failure** — a failure was reported; applicability or root cause remains unverified.
- **Needs verification** — a concrete invariant, test, or measurement remains unverified.
- **Design question** — a requirement or owner decision is needed.
- **No issue found** — only within the explicitly inspected scope.
- **Not inspected** — a material affected path could not be reviewed.

Name the triggering condition or inspected scope and link supporting evidence
where available. Keep actions specific; use "None" when no action is needed.

The table must agree with the detailed findings. Summarize rather than repeat the full findings. 
Do not include secrets or sensitive data.
