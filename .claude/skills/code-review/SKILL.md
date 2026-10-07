---
name: code-review
description: Review OpenVINO changes for correctness, compatibility, performance, security, testing, and maintainability. Use when reviewing a pull request or proposed diff.
---

# OpenVINO Code Review

## Purpose

Provide a concise, evidence-based review of the proposed diff. Review changed
code and the directly affected neighboring code; do not perform a broad audit
of unrelated legacy code.

## Review workflow

1. Read the diff, commit or pull request description, and relevant surrounding
   code.
2. Identify the affected component, public APIs, runtime paths, input
   boundaries, and expected tests. Load the relevant component references
   from the table below.
3. Check for functional regressions first, then input safety, performance,
   compatibility, test coverage, and maintainability.
4. Before reporting a finding, confirm that the claimed problem exists in the
   source code. Distinguish source content from diff markers, line numbers,
   and other review annotations. Verify the failure against the configured
   behavior or documented requirements; do not assume additional triggers,
   unsupported usage, or stricter rules. If the available evidence does not
   establish a defect, omit the finding.

## Conditional review modes

### Revert pull requests

Treat a pull request that reverts a previously merged change as a regression
review rather than as entirely new functionality:

- Identify the original pull request or change from the title and description.
- Determine what behavior, bug, or risk the original change addressed.
- Check whether the revert reintroduces that issue or creates a new
  compatibility problem.
- Avoid requesting new tests, refactors, or style changes for restored code
  unless they are needed to prevent a concrete regression.
- If the original change and its rationale cannot be identified, state that as
  a validation gap rather than guessing.

### External contributions

For pull requests from external contributors, keep the review evidence-based
and focused on merge-blocking or materially risky behavior:

- Prioritize required CI results, buildability, changed-path risk, and
  correctness of the submitted diff.
- Verify that all commits stay within the scope described by the pull request.
  Call out unrelated component changes as an out-of-scope risk.
- Do not request extra local validation when required checks already cover the
  changed behavior and are passing.
- When checks fail, inspect the available evidence and distinguish
  change-induced failures from infrastructure, network, resource, or
  pre-existing test failures.
- Do not infer lower quality from external authorship; report only issues
  supported by the diff, tests, or CI evidence.

## Priority checklist

### Correctness and compatibility

- Check shape, element type, attributes, dynamic behavior, state, error paths,
  and cross-device behavior where relevant.
- Check backward compatibility for existing public APIs and serialized models.
- Check that generated code, bindings, documentation, and registrations remain
  consistent when an API or user-visible behavior changes.

### Safety

- Treat model metadata, file contents, and user-provided values as untrusted.
- Check bounds, overflow, signed/unsigned conversions, narrowing conversions,
  ownership, lifetime, and resource cleanup.
- Require validation before unsafe arithmetic or memory access, and prefer
  fail-fast errors over warning-only handling of invalid input.

### Performance

- Look for unnecessary tensor or container copies, allocations in hot paths,
  repeated invariant work, synchronization changes, and altered memory reuse.
- Require evidence before claiming a measurable performance regression.

### Tests and validation

- Expect regression coverage for bug fixes and behavioral changes when an
  existing test suite can express the scenario. Do not request a new test
  when equivalent coverage already exists.
- Check architecture, frontend, plugin, and dynamic-shape coverage where the
  changed behavior depends on them.
- Consider required CI and build metadata only when the diff changes those
  surfaces.

## Component guidance

Load only the references relevant to the changed behavior. Paths below are
relative to the repository root and include tests within each component.
Apply multiple references when a change crosses boundaries; for shared tests
or documentation, select by the behavior exercised or described.

| Changed paths or behavior | Reference |
| --- | --- |
| `src/core/`, `src/inference/` | [Runtime and core](references/runtime.md) |
| `src/common/transformations/`, `src/common/low_precision_transformations/`, `src/common/offline_transformations/`, `src/common/snippets/`, `src/core/src/pass/`, `src/core/src/pattern/`, `src/core/include/openvino/pass/`, transformation passes and tests under frontends or plugins, `src/bindings/python/tests/test_transformations/` | [Transformations](references/transformations.md), plus the owning component's guidance |
| `src/plugins/`, including plugin-local tests | [Plugins](references/plugins.md) |
| `src/tests/functional/` shared operator and inference behavior tests | [Runtime](references/runtime.md) and [plugins](references/plugins.md), as exercised |
| `src/frontends/`, `src/bindings/python/src/openvino/frontend/` (including PyTorch and JAX decoders), `src/bindings/python/src/pyopenvino/frontend/`, framework conversion tests in `tests/layer_tests/` and `tests/model_hub_tests/` | [Frontends](references/frontend.md); also apply [bindings](references/bindings.md) for Python API and language-boundary changes |
| `src/bindings/`, public API changes under `src/core/include/` or `src/inference/include/` | [Bindings and public APIs](references/bindings.md) |
| `.github/workflows/`, `.github/actions/`, `.github/scripts/`, `.github/dockerfiles/`, Smart CI and dependency configuration under `.github/`, `cmake/`, `CMakeLists.txt` and `*.cmake` at any depth, dependency manifests and integration changes | [CI, build, and dependencies](references/ci.md) |

## Review output

Report all material, high-confidence, actionable findings in descending order
of practical impact. Each finding should include:

- the exact changed location, linked when the review interface supports it;
- the failure mode and its impact on OpenVINO users, builds, or runtime; and
- a minimal, concrete fix direction.

Keep one issue per finding and use one representative location for a repeated
root cause. Omit speculative issues, style preferences enforced by tooling,
and unrelated refactor requests. Separate confirmed issues from questions,
assumptions, and validation gaps. Do not include secrets or sensitive data.

If no actionable issue is found, say so explicitly and summarize the areas
reviewed, tests or checks run, and any validation that remains unavailable.
