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
- When matching user-provided or framework names to converted tensors, use
  the names the frontend actually produces, including generated, suffixed,
  or aliased names. Report an error or fall back explicitly when a match fails;
  do not silently drop user-specified shapes, types, or values.
- Handle zero-element tensors of any rank. Detect emptiness from the element
  count or a zero dimension, not by comparing against one specific shape.

## Downstream graph consumers

The emitted graph is consumed by weight compression (NNCF), plugin
transformations, serialization, the ONNX Runtime OpenVINO Execution Provider
(OVEP), and OpenVINO GenAI pipelines and tools, including `genai-cli`. A graph
that converts and infers correctly can still regress compressed accuracy,
model size, performance, or integration with these consumers.

- Treat model inputs and outputs (count, order, names, element types, and
  ranks) and model state as an interface. OVEP binds them to the original
  model's inputs and outputs, and GenAI relies on expected names, state, and
  runtime information. Flag changes to them, including extra inputs from
  constants that become parameters.

- When a change alters how weights reach `MatMul`, `Gather`, or `Convolution`
  (constant element type, `Convert` placement, or `Split`, `VariadicSplit`,
  `Slice`, `Reshape`, `Transpose` between the constant and its consumer), check
  that the weight remains a `Constant`, optionally followed by `Convert`,
  directly feeding the consumer. NNCF weight compression and plugin
  compressed-weight patterns rely on this. Request a before/after comparison
  of models exported with int8 and int4 weight compression (for example,
  `optimum-cli export openvino --weight-format int8`): `.bin` size and
  uncompressed f16/bf16/f32 weights on MatMul paths.
- Rank or layout changes at `MatMul` inputs or outputs, and new nodes between
  `MatMul` and its eltwise or attention consumers, can change plugin
  implementation selection (for example, GPU dynamic quantization depends on
  the FullyConnected input rank) and block fusions such as FullyConnected with
  bias/eltwise, MoE, and SDPA. Flag them and request prefill and decode
  measurements on CPU and GPU for representative weight types.

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
- Operator-level tests do not cover changes to shared dispatch, decoders,
  graph-wide passes, or weight handling. For such changes, ask which real
  models were converted and validated, covering the affected model families
  (for example, dense and MoE LLMs, VLMs, hybrid recurrent models, and
  quantized models), and treat missing or pending validation as a gap.
  Recommend splitting large changes that alter several capture paths or
  frontend stages at once.

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
