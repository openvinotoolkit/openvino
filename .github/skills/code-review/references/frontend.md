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

## Python decoders and bindings

- Review Python decoder changes as conversion logic. Check the graph, tensor
  types, shapes, constants, attributes, and input/output ordering exposed to
  the C++ frontend, including decoder and framework-object lifetimes.
- For PyTorch, check the affected
  [TorchScript](../../../../src/bindings/python/src/openvino/frontend/pytorch/ts_decoder.py)
  or [FX/export](../../../../src/bindings/python/src/openvino/frontend/pytorch/fx_decoder.py)
  path. Preserve mutation and alias semantics and verify that regression tests
  exercise the changed capture path.
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
