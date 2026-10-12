# OpenVINO Frontend Review

Apply to changes under `src/frontends/`, frontend Python decoders and bindings
under `src/bindings/python/src/openvino/frontend/` and
`src/bindings/python/src/pyopenvino/frontend/`, and their conversion tests.

## Review focus

- Keep frontend code hardware- and plugin-agnostic. Conversion must preserve
  framework semantics without targeting a specific device or plugin. Flag
  device-dependent conversion branches or backend-specific optimizations;
  those belong in the relevant plugin's transformation or execution pipeline.
- Check rank, dimension, element type, layout, optional inputs, attributes,
  opset or framework-version behavior, and dynamic-shape handling.
- Support the broadest dynamism allowed by framework semantics, including
  dynamic ranks, dimensions, and element types. Always assume input element
  types may be dynamic during conversion; avoid creating constants with a
  type taken from an input. When a constant must match an input's type, create
  it with a concrete type and use `ConvertLike` to match the input without
  requiring its type to be known during conversion.
- Work with `ov::Output<ov::Node>` values when constructing the graph. Avoid
  extracting nodes with `get_node_shared_ptr()` just to pass them as inputs
  to another operation: this discards the original output index. Passing a
  node implicitly selects its default output (normally output 0), which can
  miswire a multi-output producer or fail if it has no default output.
  Preserve the original output value or explicitly select `node->output(i)`.
- Verify that conversion emits the intended OpenVINO graph and preserves
  framework semantics for valid inputs.
- Check malformed, missing, out-of-range, duplicated, and unsupported values
  for deterministic validation and useful errors.
- Check registration tables, opset ranges, translator dispatch, normalization
  passes, and framework-specific entry points for consistency.
- Check that conversion does not depend on unavailable compile-time shape or
  type information when runtime graph computation is required.
- Check that user-visible frontend behavior has focused regression tests,
  including dynamic, boundary, and malformed-input cases where applicable.
- Check that conversion tests actually run inference and compare the results
  with expected values; a test that only prepares inputs and expected outputs
  verifies nothing. Flag expected values that look like placeholders; they
  should come from the framework or a trusted reference.
- When a change lets inputs that had to be constants be runtime values, require
  a positive test that feeds them at runtime and exercises output shapes that
  depend on them, not only rejection tests.
- Check that test tolerances fit the magnitude of the outputs. An absolute
  tolerance that ignores the output scale can hide real errors or make the test
  flaky.
- When a change adds support for an operation or case, check the frontend's
  expected-failure and skip lists for tests it now fixes. Newly passing strict
  xfails fail CI.
- Do not ask for frontend workarounds for plugin or reference-evaluator
  limitations, such as a missing f64 kernel. Report them against the plugin.

## Graph efficiency

Review the generated graph for efficiency even when numerical tests pass.
Look for duplicated operations, unnecessary tensor extraction and
reconstruction, redundant conversions, and repeated shape computations.
Prefer equivalent graphs that reduce computation and data movement while
remaining hardware- and plugin-agnostic.

Avoid creating equal nodes unnecessarily, but account for
[`ov::pass::SharedOpOptimization`](../../../../src/common/transformations/include/transformations/common_optimizations/shared_ops_optimization.hpp),
which can merge eligible nodes with matching operation types, inputs, and
attributes. Check whether the relevant pipeline runs this pass and whether
the nodes qualify. Duplication that the pipeline removes is normally a minor,
non-blocking recommendation. Operations on different inputs are not equal
merely because they perform the same kind of work. Also look for algebraic
simplifications and alternative decompositions that use fewer operations
without relying on duplicate elimination.

Attribute equality matters even when an attribute does not affect a specific
calculation: a `Reshape` with `special_zero=true` will not merge with one using
`special_zero=false`, even if their shared constant shape contains no zeros
and both produce the same result. When attribute variants are equivalent for
all supported inputs of the calculation, prefer the variant most commonly
used by surrounding translators and graph-building helpers to increase the
chance of merging. Check dynamic values and edge cases before treating an
attribute as irrelevant; do not change semantics merely to match other nodes.

Compare input, intermediate, and output tensor sizes, including element
counts and byte sizes, rather than judging efficiency by operation count
alone. Operations that consume or produce smaller tensors are usually more
efficient because they reduce computation, memory traffic, and temporary
storage. Three operations on small tensors can outperform two on large
tensors. Where semantics allow, consider reducing data before expensive
processing and avoiding early broadcasts or large intermediate tensors.
Account for operation cost, packing or copying, and the surrounding graph;
fewer nodes alone do not establish a performance improvement.

For complex tensors represented by `ComplexTypeMark`, consider operating on
the combined tensor from `get_data()` instead of extracting `get_real()` and
`get_imag()`, applying the same operation twice, and reconstructing the result.
The combined representation has shape `[..., 2]`; preserve that trailing
real/imaginary dimension, the complex-part type, and the framework's logical
shape and axes. Apply this recommendation only when the operation is
equivalent on the combined representation; operations that mix the components
or require different real/imaginary formulas may need separate handling.
Inspect how the input is represented and how consumers use it before
recommending a change that would itself add packing or extraction work.

For slicing, consider a single operation over the logical axes of the combined
tensor, leaving the trailing component dimension intact. Compute logical
shape bounds without extracting a component tensor solely for `ShapeOf`.
Balance any reduction in graph operations against the sizes of the tensors
processed and the cost of combining or separating the components.

## Element-type fidelity and practical tradeoffs

Aim to preserve the source framework's element types and promotion rules in
the converted graph: if the framework produces `i64`, the frontend should
normally produce `i64`; likewise for `f64`. Distinguish public input/output
types from internal constants, indices, and intermediate computations.
Plugin compilation may lower unsupported `i64`/`f64` computations to
`i32`/`f32`; that lowering belongs to the plugin and is not by itself a
frontend defect or a reason to require device-specific conversion logic.

Treat an `i64`/`f64` versus `i32`/`f32` fidelity difference alone as a
low-priority (`P3`), non-blocking issue. Accept a justified `i32` or `f32`
conversion when preserving `i64` or `f64` would cause conversion problems or
require substantial code changes, provided the affected values and intended
use make the tradeoff reasonable. Do not demand a broad refactor solely for
64-bit fidelity or flag every internal 32-bit constant as a dtype defect.

Check the actual range and precision requirements before recommending
narrowing. If the diff demonstrably introduces overflow, incorrect indexing,
a conversion failure, or material numerical error for a supported scenario,
assess that failure by its practical impact. Higher severity requires evidence
of such a failure, rather than the presence or absence of a 64-bit type alone.

## Framework-specific guidance

- ONNX frontend (`src/frontends/onnx/`): also apply [ONNX frontend review](onnx.md).
- PyTorch translators, decoders, and conversion tests: also apply
  [PyTorch frontend review](pytorch.md).

## Python decoders and bindings

- Review Python decoder changes as conversion logic. Check the graph, tensor
  types, shapes, constants, attributes, and input/output ordering exposed to
  the C++ frontend, including decoder and framework-object lifetimes.
- For [JAX](../../../../src/bindings/python/src/openvino/frontend/jax/jaxpr_decoder.py),
  check Jaxpr variable identity, literals and captured constants, primitive
  parameters, and output ordering against the C++ decoder contract.
- Apply [bindings guidance](bindings.md) when Python API, exception handling,
  ownership, or C++/Python interaction changes.

## Transformations

For frontend normalization and decomposition passes, also apply the
[transformation review guidance](transformations.md). Verify that pass ordering
and handling of framework placeholder nodes preserve conversion semantics.
These passes must follow the same hardware- and plugin-agnostic rules as
operation translators and Python decoders.

## Evidence and scope

Do not report unsupported framework behavior unless the diff claims to
support it or changes its error handling. Check that fallback nodes or
placeholder translators are not presented as full support.

Include concrete efficiency or dtype improvements as non-blocking
recommendations even when no correctness defects are found. Give the changed
location, the avoidable computation or data movement or the dtype tradeoff,
and a minimal fix direction. Keep confidence separate from severity: a
low-impact recommendation can be well supported. Do not claim a measured
speedup without profiling evidence, invent recommendations, or demand broad
refactors for minor improvements. Escalate only when evidence establishes a
material regression, including for external contributions.
