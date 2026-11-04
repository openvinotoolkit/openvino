# OpenVINO Frontend Review

Apply to changes under `src/frontends/`, frontend Python decoders and bindings
under `src/bindings/python/src/openvino/frontend/` and
`src/bindings/python/src/pyopenvino/frontend/`, and their conversion tests.

## Conversion semantics

- Keep conversion hardware- and plugin-agnostic. Device-specific rewrites
  belong in plugin pipelines; do not request frontend workarounds for missing
  plugin or reference-evaluator kernels.
- Check framework semantics, operator versions, optional inputs, defaults,
  layouts, and translator registration. Report unsupported behavior only
  when the diff claims support or changes its error handling; placeholder
  nodes are not full support.
- When modifying an existing conversion to support another operation mode or
  fix a bug, preserve the emitted graph for previously working cases unless
  a concrete benefit, such as improved performance or graph simplification,
  justifies the change.
  Equivalent graphs can break downstream transformation matching. Always
  flag changes to those cases for human review, stating the affected graph
  form, rationale, and transformation risk, even when the risk is justified.
- Preserve the dynamism allowed by the framework, including ranks, dimensions,
  and element types. Avoid requiring compile-time shape or type information
  for runtime calculations. To match a possibly dynamic input type, create a
  concretely typed constant and use `ConvertLike`.
- Connect graph inputs with `ov::Output<ov::Node>`. Passing
  `get_node_shared_ptr()` instead loses the output index and can miswire a
  multi-output producer; use the original value or `node->output(i)`.
- Review Python decoders as conversion logic: preserve types, shapes,
  constants, attributes, input/output order, and framework-object lifetimes.

## Graph efficiency

- Consider equivalent decompositions with less computation, data movement,
  extraction/reconstruction, or shape work, even when numerical tests pass.
  Look beyond duplicate elimination for ways to use fewer operations.
- Compare input, intermediate, and output element counts and byte sizes.
  Smaller tensors usually cost less to process and store: three small
  operations can outperform two large ones. Account for operation and copy
  costs; avoid early broadcasts or large intermediates when semantics allow.
- Avoid equal nodes unnecessarily, but account for
  [`ov::pass::SharedOpOptimization`](../../../../src/common/transformations/include/transformations/common_optimizations/shared_ops_optimization.hpp).
  It merges eligible nodes with matching operation types, inputs, and
  attributes when the pipeline runs it; duplication it removes is normally
  a minor issue. The same operation on different inputs is not a duplicate.
- Attribute equality matters even when values are equivalent for the current
  calculation: `Reshape` with `special_zero=true` will not merge with
  `special_zero=false`, even for a shared constant shape without zeros.
  When variants are equivalent for all supported inputs, prefer values common
  in surrounding translators/helpers to improve merging chances. Check dynamic
  values and edge cases before treating an attribute as irrelevant.
- For `ComplexTypeMark`, consider processing combined `get_data()` instead of
  extracting `get_real()`/`get_imag()`, processing both, and reconstructing.
  Preserve the trailing component dimension in `[..., 2]`, complex-part type,
  and logical axes/shape. Slicing can operate once over the logical axes, and
  shape bounds need not extract a component solely for `ShapeOf`. Balance
  packing/extraction costs and tensor sizes; separate formulas may be needed
  for operations that mix real and imaginary components.

## Element types

Prefer framework output types and promotion rules, including `i64` and `f64`.
Distinguish these from internal constants and intermediates; plugin lowering
of unsupported computations to `i32`/`f32` is not a frontend defect.

Dtype fidelity alone is a non-blocking `P3` issue. Accept justified `i32`/`f32`
conversions when preserving 64-bit types causes conversion problems or
substantial code changes and the range/precision tradeoff is reasonable.
Do not demand broad refactors or flag every 32-bit constant. Escalate based
on demonstrated overflow, wrong indexing, conversion failure, or material
numerical error for a supported scenario, rather than type width alone.

## Conversion tests

- Verify tests actually run inference and compare against the framework or a
  trusted reference, with tolerances appropriate to output magnitude.
- When constant-only inputs become runtime inputs, check positive coverage
  with runtime values and dependent output shapes, not only rejection cases.
- Check expected-failure and skip lists when support expands; newly passing
  strict xfails fail CI.

## Additional guidance

- ONNX: [ONNX frontend review](onnx.md).
- PyTorch: [PyTorch frontend review](pytorch.md).
- [JAX decoder](../../../../src/bindings/python/src/openvino/frontend/jax/jaxpr_decoder.py):
  check Jaxpr variable identity, literals/captured constants, primitive
  parameters, and output ordering against the C++ decoder contract.
- Frontend passes: apply [transformation guidance](transformations.md),
  checking pass ordering and framework placeholder handling.
- Python API or language-boundary changes: apply [bindings guidance](bindings.md).

Include concrete efficiency or dtype improvements as non-blocking
recommendations even with no defects. Explain the changed location, avoidable
work or tradeoff, and minimal fix; claim measured speedups only with profiling.
