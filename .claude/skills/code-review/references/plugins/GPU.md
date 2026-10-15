## Purpose

Review OpenVINO Intel GPU pull requests for:

1. Code correctness and completeness
2. Functional regressions
3. Performance regressions and inefficient execution
4. Intel GPU platform and runtime compatibility
5. Architecture and code clarity
6. Validation and test coverage
7. Possible improvements

PR should be reviewed from code correctness and completeness point of view.
Functional correctness and performance are the primary review dimensions.

Architecture, maintainability, test quality, and code clarity are important
secondary dimensions. They become blocking only when they create a concrete
correctness, performance, compatibility, or significant maintenance risk.

## Review principles

Follow these principles throughout the review:

- Base every defect finding on changed code and a concrete execution condition.
- Distinguish confirmed defects from risks, validation gaps, and questions.
- Do not assume that successful compilation proves functional correctness.
- Do not assume that passing one test configuration proves platform coverage.
- Do not claim a performance improvement or regression without explaining
  the expected mechanism or citing provided measurements.
- Do not treat missing evidence as proof that a defect exists.
- Prefer a small number of actionable findings over many speculative comments.
- Inspect the final PR head, not only an earlier commit, stale review comment,
  author explanation, or summarized diff.
- Treat bot and human review comments as leads to investigate, not as evidence
  that a finding remains applicable at the final PR head.
- Do not claim that every path, load site, kernel variant, or runtime is covered
  unless each applicable variant was directly inspected or a shared invariant
  proves the coverage.
- Separate a proven defect from an invariant that still needs verification and
  from a test, benchmark, or platform-coverage gap.
- When evidence is incomplete, state exactly what was inspected and what remains
  unverified. Do not fill gaps with inference from similarly structured code.
- Review the whole affected execution path, not only isolated changed lines.
- Consider interactions across files when the implementation spans layers.
- Do not recommend approval or merge.
- Escalate architectural intent and product trade-offs to human GPU owners.

## Review workflow

Perform the review in the following order.

### Step 0: Establish the review baseline

Record the facts needed to ensure that the review targets the current change:

- PR head SHA, base SHA, and whether the PR changed while it was being reviewed
- Files and diff hunks at the final PR head
- CI/check conclusion for each relevant check: passed, failed, pending, skipped,
  cancelled, or neutral; do not summarize this merely as "completed"
- Existing approvals, change requests, unresolved review threads, and their
  relevance at the final PR head
- Existing bot or human findings and the commit/line to which each applies

For each relevant existing finding, classify it before relying on it:

1. **Still applicable** — the final PR head retains the triggering code.
2. **Fixed** — cite the final-head code or test that resolves it.
3. **Outdated or inapplicable** — explain the changed condition or incorrect
   premise.
4. **Not yet verifiable** — identify the missing file, variant, runtime, test,
   or platform evidence.

Do not describe an approval by an assignee or author as merge-readiness evidence.
If repository policy is unavailable, report required-approval status as unknown.

### Step 1: Understand the PR intent

Determine:

- What behavior is being added, removed, fixed, or optimized?
- Which user-visible or internal behavior is expected to change?
- What is the stated reason for the change?
- Is the PR primarily a bug fix, optimization, refactoring, enablement,
  cleanup, platform workaround, or test-infrastructure change?
- What evidence does the PR provide for correctness and performance?

Do not infer intent solely from the implementation when the PR description,
tests, or linked issue provide more direct information.

If intent remains unclear, identify the ambiguity as a question rather than
inventing a requirement.

### Step 2: Classify the affected components

Classify the PR into every applicable area:

- Graph transformation or graph optimization
- GPU program building
- Primitive implementation
- Kernel selector
- OpenCL kernel
- oneDNN integration
- OpenCL runtime
- Level Zero runtime
- Memory allocation or memory dependency handling
- USM or buffer management
- Model or GPU program caching
- Dynamic shape or dynamic batch handling
- Tensor layout, reorder, padding, pitch, or offset handling
- Precision conversion or quantization
- LLM, SDPA, Paged Attention, or KV-cache execution
- Device capability detection
- Platform enablement or platform-specific workaround
- Build system or configuration
- Unit, functional, validation, or performance tests
- Documentation or diagnostics

Use this classification to select the applicable review domains.

### Step 3: Determine affected execution scope

Identify the scope supported by evidence in the PR:

- Intel GPU generations or capability groups
- Integrated or discrete GPU
- Linux or Windows
- OpenCL or Level Zero runtime
- If relevant, both runtimes
- Static or dynamic shapes
- Single-stream or multi-stream execution
- Single-request or multi-request execution
- Relevant tensor layouts
- Relevant precisions
- Relevant batch sizes
- Relevant operation, model, or workload families
- Model compilation, first inference, steady-state inference, or cache import

Do not claim that all platforms are affected when the diff only establishes a
narrower scope.

When the exact platform scope cannot be established, report it as unknown or
as a validation gap.

### Step 4: Map the affected execution path and behavior matrix

Before reporting a cross-layer concern, map the changed path from its input
contract through implementation and test coverage. For example:

```text
graph transformation / API input
  -> primitive or descriptor
  -> implementation and kernel selection
  -> JIT constants and kernel arguments
  -> each relevant kernel variant
  -> fallback or rejection path
  -> reference, functional, and performance tests
```

For changes involving precision, quantization, kernel generation, dispatch, or
platform selection, build a compact behavior matrix. Include only applicable
dimensions, such as:

- device generation or capability guard;
- OpenCL and Level Zero paths when both are supported;
- optimized path, fallback path, and explicit rejection path;
- prefill/decode, first-token/second-token, paged/non-paged, or other kernel
  variants;
- precision, storage encoding, symmetric/asymmetric mode, and zero-point form;
- static/dynamic shapes, boundary/tail sizes, and layout variants;
- existing behavior changed by the PR versus newly enabled behavior.

Map tests and measurements to the matrix. Mark cells as **covered** only with
direct evidence. Mark all remaining relevant cells as **uncovered**, **unknown**,
or **intentionally unsupported**, and identify the guard that enforces an
unsupported cell.

### Step 5: Apply the relevant review domains

Always perform the functional-correctness and completeness reviews.

Perform the performance review when the PR:

- claims or implies a performance improvement;
- modifies an inference or compilation hot path;
- changes kernels, dispatch, synchronization, memory, fusion, or layouts;
- changes primitive or kernel selection;
- adds a platform-specific implementation;
- can affect memory use, compilation time, or model-loading time.

Perform the platform review when the PR:

- changes device or feature detection;
- introduces generation-specific behavior;
- changes OpenCL or Level Zero paths;
- uses a GPU extension or specialized hardware capability;
- changes Windows- or Linux-specific behavior;
- changes behavior for integrated or discrete GPUs.

Perform the architecture review when the PR:

- introduces a new abstraction or interface;
- moves responsibility between layers;
- duplicates logic across runtimes, platforms, or implementations;
- changes ownership, lifetime, caching, or synchronization;
- introduces a workaround into generic code;
- establishes a pattern expected to be reused.

If dedicated supporting skills are available, apply their instructions:

- `openvino-gpu-functional-correctness`
- `openvino-gpu-performance`
- `openvino-gpu-platform-coverage`
- `openvino-gpu-architecture`
- `openvino-gpu-test-completeness`

If a supporting skill is unavailable, perform the corresponding checks using
the minimum review requirements in this coordinator.

## Minimum functional-correctness review

Check changed behavior for:

- Static, dynamic, partially known, scalar, empty, and boundary shapes
- Rank-specific assumptions
- Tensor layout, pitch, offset, padding, and broadcasting
- Index and dispatch calculations
- Integer overflow and unsafe narrowing
- Out-of-bounds reads and writes
- Precision conversion, rounding, saturation, and accumulation
- Signed and unsigned arithmetic
- Quantization parameters, zero points, and scales
- In-place execution and memory aliasing
- Buffer size and lifetime
- Synchronization, barriers, events, and dependencies
- Multi-stream and multi-request safety
- Mutation of shared or cached state
- Kernel argument consistency
- JIT constant consistency
- Dispatch geometry and tail processing
- Kernel selector conditions
- Device-capability guards
- Unsupported combinations
- Fallback behavior
- OpenCL and Level Zero semantic consistency
- Error propagation and recovery

For each suspected defect, identify the exact condition under which it occurs.

For generated or multi-variant GPU kernels, trace the condition through:

- host-side selection and validation;
- descriptor or primitive state;
- JIT constant generation;
- kernel argument construction and binding;
- every kernel variant compiled for the affected configuration;
- fallback, build-failure, and unsupported-device behavior.

For packed or quantized data, verify every applicable load/decode/dequantize
site and distinguish storage representation from logical value representation.
In particular, check:

- signed versus unsigned storage;
- zero-point convention and where it is applied;
- scale/group indexing and broadcast rules;
- byte packing, nibble ordering, and address/pitch conversion;
- vector/tile tail behavior and whether padded lanes can be stored.

If only some variants can be inspected, report the result as partial. For
example, say "the decode kernel paths were inspected; the prefill variant was
not inspected" rather than claiming that all load sites are covered.

Do not report generic possibilities such as "this may cause a race" without
showing the shared state, concurrent access, and missing synchronization.

## Minimum performance review

Check whether the change can introduce:

- Additional kernel launches
- Additional host-device or device-host transfer
- Additional reorder, copy, conversion, or memory pass
- Additional synchronization or blocking wait
- Allocation or deallocation in a hot path
- Increased temporary-memory size or lifetime
- Loss of kernel or graph fusion
- Reduced parallelism, occupancy, vectorization, or subgroup efficiency
- Divergent control flow
- Uncoalesced or redundant memory access
- Repeated calculation that could be cached
- Repeated initialization that could be moved out of a loop
- More expensive primitive or kernel selection
- Increased compilation or model-load time
- Increased first-inference latency
- Cache invalidation or reduced cache reuse
- Silent fallback to a slower implementation
- Improvement on one GPU generation at the expense of another

Separate these measurement scopes:

1. Kernel execution
2. End-to-end inference latency or throughput
3. Compilation and model-loading time
4. First inference
5. Cache import or cache export
6. Host and device memory consumption

Do not equate a kernel-level speedup with an end-to-end improvement.

When reporting a performance concern, explain the expected mechanism. For
example:

- an added wait serializes otherwise asynchronous work;
- an added reorder creates another full tensor memory pass;
- a changed selection condition moves affected shapes to a generic kernel;
- a larger temporary buffer increases device-memory pressure;
- dispatch dimensions leave GPU execution resources underutilized.

If the mechanism is plausible but impact is not established, report a
validation gap rather than a confirmed regression.

For every performance-relevant selection, dispatch, layout, or kernel change,
separate:

- **newly enabled paths**, where absolute expected performance should be shown;
- **pre-existing paths changed by the PR**, which require comparison with the
  base commit for the affected workload and GPU generation;
- **correctness-driven trade-offs**, which should state the expected mechanism
  and the accepted scope rather than implying an unmeasured regression.

## Minimum architecture and clarity review

Check:

- Whether responsibility is placed in the correct GPU plugin layer
- Whether generic and GPU-generation-specific logic are separated
- Whether generic and runtime-specific logic are separated
- Whether existing abstractions and helpers are reused appropriately
- Whether duplication can cause implementations to diverge
- Whether ownership and lifetime are explicit
- Whether interfaces preserve established invariants
- Whether failure handling is consistent with surrounding code
- Whether names describe intent rather than implementation accidents
- Whether complex control flow can be simplified safely
- Whether comments explain hardware or runtime constraints
- Whether comments explain why a workaround is required
- Whether temporary workarounds have a removal condition or tracking reference
- Whether the design supports likely extensions without speculative abstraction

Do not request broad refactoring unless the current structure creates a
concrete correctness risk, performance risk, or substantial maintenance risk.

When the decision depends on future plans or undocumented architectural intent,
use a question for the responsible GPU owner.

## Minimum completeness review

Check whether the PR includes, where applicable:

- All required implementation branches
- OpenCL and Level Zero handling
- Optimized path and fallback path
- Supported and unsupported capability handling
- Static and dynamic shape handling
- Relevant precision handling
- Integrated and discrete GPU considerations
- Linux and Windows considerations
- Positive functional tests
- Negative tests
- Boundary and tail-condition tests
- A regression test that fails without the fix
- Performance measurements for performance claims
- Appropriate model, operation, shape, and batch coverage
- Memory measurements when memory behavior changes
- Compilation-time measurements when compilation changes
- Documentation for changed behavior
- Comments for non-obvious hardware constraints
- Removal of debug output and temporary code
- Handling or removal of TODOs introduced by the change

A missing test is not automatically a functional defect.

Report it as a validation gap unless the changed code itself demonstrates a
failure.

## Evidence rules

Every finding must contain enough evidence for the author to evaluate it
without guessing.

Acceptable evidence includes:

- Changed control flow
- Changed data flow
- Changed indexing or dispatch calculation
- A violated local or component invariant
- An unguarded platform-specific operation
- A missing case in a changed selection condition
- Inconsistency between implementation and an existing test
- Inconsistency between related runtime paths
- A benchmark or test result provided with the PR
- A new code path for which no safe fallback exists

Evidence from a final-head source is stronger than an indirect source. Use the
following order of preference:

1. Final-head changed code, complete file view, or exact diff hunk
2. Directly related final-head test, benchmark, or CI result
3. Call path, invariant, or existing implementation contract verified against
   the final head
4. Earlier review comment or author explanation, followed by final-head
   verification

An author response, bot finding, truncated file view, or inferred similarity to
another branch is not sufficient by itself to claim a defect is fixed or that
all variants are correct. It may justify a focused verification request.

Unacceptable evidence includes:

- Complexity alone
- Personal coding preference
- An unsupported assumption about future usage
- An unquantified claim that code is slow
- A generic assertion that more tests are needed
- Restating the code without identifying a problem
- Treating an unchanged pre-existing issue as introduced by this PR

If a pre-existing issue becomes more exposed or severe because of the PR,
explain the new contribution of the PR.

## Finding types

Use one primary category per finding:

- `FUNCTIONAL`
- `PERFORMANCE`
- `MEMORY`
- `CONCURRENCY`
- `PLATFORM`
- `RUNTIME`
- `NUMERICAL`
- `ARCHITECTURE`
- `COMPLETENESS`
- `TESTING`
- `VALIDATION GAP`
- `MAINTAINABILITY`

Use the category that describes the primary impact.

Do not create multiple findings for the same root cause merely because it has
several consequences. Explain secondary consequences in the Impact section.

## Severity rules

Severity expresses potential impact. Confidence expresses strength of
evidence. Do not combine the two concepts.

### CRITICAL

Use `CRITICAL` for a concrete issue that can cause:

- Out-of-bounds writes or unsafe memory corruption
- Use-after-free
- Corruption caused by an established race
- Deterministic broad crash on a supported configuration
- Broad silent data corruption
- Security-relevant unsafe execution
- Removal or bypass of a required safety guard
- Severe failure of a primary execution path with direct evidence

A Critical finding must identify:

- the concrete execution path;
- the triggering condition;
- the unsafe or incorrect operation;
- the affected scope.

Do not use Critical because code is complex, risky, or insufficiently tested.

Critical findings are blocking.

### HIGH

Use `HIGH` for a probable or established issue that can cause:

- Incorrect output on a supported configuration
- Crash on a supported but limited configuration
- Missing synchronization or dependency handling
- Overflow affecting allocation, indexing, or dispatch
- Broken or absent fallback for a supported configuration
- Incorrect device-capability handling
- Material OpenCL and Level Zero semantic inconsistency
- Major performance regression in a hot path
- Serialization or transfer added to a hot inference path
- Severe platform compatibility failure
- A semantic change introduced by an optimization without adequate protection

High findings are blocking unless evidence disproves the concern or a
responsible human owner explicitly accepts the behavior.

For a High finding, demonstrate that the affected configuration is supported or
explicitly promised by the PR, and show a reachable triggering path. Otherwise
use Medium for an invariant risk, validation gap, or verification request.

### MEDIUM

Use `MEDIUM` for:

- A credible localized edge-case risk
- An important validation or coverage gap
- An unconfirmed but technically grounded performance concern
- Missing handling for a non-primary scenario
- Reliance on an undocumented invariant
- Incomplete error handling outside the main execution path
- Missing boundary or negative testing
- Duplication likely to cause implementation divergence
- A meaningful architecture or maintenance problem

A Medium finding should normally be addressed by code, evidence, or a
documented design explanation.

### LOW

Use `LOW` for a non-blocking improvement such as:

- Unclear naming in complex GPU logic
- Missing explanation of a hardware or runtime constraint
- Unnecessarily complicated local control flow
- Minor meaningful duplication
- Weak diagnostics
- Localized test readability
- A maintainability issue with no current correctness impact

Do not use Low for formatting, automated style rules, or personal preference.

### QUESTION

Use `QUESTION` when:

- Architectural intent is unavailable;
- the behavior depends on product requirements not present in the PR;
- the correct platform scope cannot be established;
- an apparent inconsistency may be intentional;
- GPU owner knowledge is required.

A question must explain why the answer matters to correctness, performance,
compatibility, or maintainability.

Do not disguise an unsupported allegation as a question.

## Confidence rules

Assign one confidence level to every finding.

### High confidence

Use when the conclusion follows directly from:

- changed control or data flow;
- an explicit triggering condition;
- a clear language, API, kernel, or component invariant;
- a directly inconsistent test or runtime path.

### Medium confidence

Use when:

- the failure mechanism is technically credible;
- some caller, platform, runtime, or configuration context is missing;
- validation is required to establish actual impact.

### Low confidence

Use only for a meaningful review question requiring context that is not
available in the diff.

Do not report a blocking Critical or High finding with Low confidence.
Convert it to a Medium validation gap or a Question.

## Finding format

Use this format for each finding:

```text
[SEVERITY][CATEGORY] Short actionable title

Location:
path/to/file.cpp:line

Evidence:
Describe the changed code and the exact condition that triggers the concern.

Evidence status:
State whether this conclusion was verified against the final PR head. Identify
the inspected variant(s), relevant guard(s), and any uninspected variant(s).

Counterevidence:
List the validation, fallback, test, benchmark, or invariant that could prevent
the issue. Explain why it is sufficient, insufficient, or still unverified.
Use `None identified` only after checking the relevant path.

Impact:
Describe the correctness, performance, memory, compatibility, or maintenance
consequence.

Affected scope:
List the known platform, runtime, OS, precision, layout, shape, batch, and
workload conditions. Mark unknown dimensions as unknown.

Recommended action:
Propose a concrete code correction or narrowly defined validation action.

Required evidence:
State the test, benchmark, trace, comparison, or invariant needed to resolve
the finding.

Disposition:
Blocking defect, pre-merge verification, follow-up validation, or question.

Confidence:
High, Medium, or Low.

## Review summary table

After the detailed findings, include a concise `## Review summary` table.
Use one row per material review area or root cause, not one row per minor
observation. Include:

| Area | Assessment | Evidence / key condition | Required action |
|---|---|---|---|

Use one of these assessments:

- `✅ No issue found` — only for the inspected scope; state that scope in the
  evidence column.
- `⚠️ Needs verification` — an invariant, test, platform, or benchmark gap
  remains and no defect is proven.
- `❌ Confirmed defect` — a reachable defect is supported by final-head evidence.
- `❓ Design question` — product or architecture-owner input is required.
- `— Not applicable` — omit the row instead when possible.

The evidence column must name the final-head evidence and relevant execution
condition. The action column must be concrete: code change, targeted test,
benchmark, trace, or owner decision. Do not summarize passing CI as proof of
functional correctness, and do not use `correct` or `safe` without a stated
scope.

Include a final `Merge blockers` row only when it clearly distinguishes proven
blocking defects from pre-merge verification requests. Do not recommend
approval or merge.
