# PyTorch Frontend Review

Apply alongside [frontend guidance](frontend.md) to PyTorch translators,
normalization passes, Python decoders, and conversion tests. Keep shared
graph-efficiency and dtype-severity rules in the frontend reference.

## Capture paths and operator dispatch

- Identify the actual input: a scripted or traced TorchScript module, a
  `torch.export.ExportedProgram`, or an FX graph from a `torch.compile`
  backend. Inspect the captured graph and configured conversion path;
  sharing an FX decoder does not make export and compile graphs identical.
- Check the affected [TorchScript decoder](../../../../src/bindings/python/src/openvino/frontend/pytorch/ts_decoder.py)
  or [FX decoder](../../../../src/bindings/python/src/openvino/frontend/pytorch/fx_decoder.py).
  Scripting can retain control flow and typed containers; tracing can
  specialize Python branches and scalar values to the example inputs.
  Export has its own symbolic-shape constraints and graph representation.
  Distinguish capture limitations from defects introduced during conversion.
- Follow dispatch through the [operation table](../../../../src/frontends/pytorch/src/op_table.cpp)
  and [translation session](../../../../src/frontends/pytorch/src/translate_session.cpp).
  TorchScript names such as `aten::add` and FX names such as
  `aten.add.Tensor` may reach shared translators or explicit FX overrides.
  Confirm the overload and schema before sharing a translator: Tensor,
  Scalar, `out`, and in-place variants may require different handling.
- Check the graph before and after any configured decomposition or
  functionalization. These steps can replace an operator, remove mutations,
  or introduce `*_copy` and other ATen variants. Do not assume that they run,
  require them solely to hide a missing translator, or treat identical values
  as sufficient evidence that view and copy variants have identical semantics.
- Review framework-version-sensitive schemas, defaults, and decompositions
  against the versions supported by the repository. Avoid assuming that an
  operator always appears in the form produced by one local PyTorch version.

## Type alignment and promotion

- Use the existing [type-alignment helpers](../../../../src/frontends/pytorch/src/utils.cpp),
  including `align_eltwise_input_types`, `get_inputs_with_promoted_types`,
  `is_python_scalar_input`, and `align_output_types`, when they implement the
  affected operator's rules. Check [helper wrappers](../../../../src/frontends/pytorch/src/utils.hpp)
  before adding ad hoc casts or duplicating promotion logic.
- Distinguish Python scalars, zero-dimensional tensors, and tensors with
  dimensions. PyTorch promotion depends on both numeric category and operand
  kind; a Python scalar's stored `i64` or `f32` type does not by itself dictate
  the result type. Preserve decoder `PyScalar` information and pass the
  correct scalar flags to alignment helpers, even when the graph scalar is
  represented by an OpenVINO constant.
- Align operands before arithmetic when promotion is required. Casting the
  result afterward cannot repair overflow, truncation, or incorrect division
  that already occurred in the wrong computation type. Check output alignment
  separately, including boolean comparison outputs, reduction accumulation
  types, integer true division, and explicit `dtype` arguments.
- Apply operator-specific rules rather than promoting every operand alike.
  In-place operations must respect the destination tensor's dtype and allowed
  casts; an `out=` tensor can constrain output handling differently from an
  ordinary result. Scaling arguments such as `alpha` also have schema-specific
  restrictions. Do not silently accept casts or argument combinations that
  PyTorch rejects in the supported scenario.
- Do not rely on static decoder output types to compensate for missing
  promotion: metadata may be dynamic or unavailable in another capture path.
  Preserve runtime type alignment where needed. For creation operators and
  Python floating scalars, check explicit dtype, inferred dtype, and framework
  default dtype behavior rather than assuming all floating results are `f32`.
- When dtype decoding changes, keep the Python dtype mappings and the C++
  ScalarType interpretation consistent. Apply the shared frontend guidance
  when assessing `i64`/`f64` fidelity and practical narrowing tradeoffs.

## Decoder inputs, shapes, and mutation

- Check positional arguments, keyword-only arguments, omitted defaults,
  `None`, inlined scalars, and lists against the actual operator schema.
  TorchScript may expose values as graph inputs that FX presents as keyword
  arguments or omits because they equal defaults. Preserve missing versus
  explicit values when the distinction affects semantics.
- Check symbolic dimensions, shape-dependent arithmetic, and input/output
  container ordering. Do not turn export shape hints or example-input sizes
  into constants when the captured graph permits those values to vary.
  Follow parameter and buffer handling through the actual decoder entry point
  so they are not accidentally exposed as user inputs or reordered.
- Follow mutation through `NodeContext::mutate_input` and the translation
  session's alias handling. Updating only an operation's returned value can
  leave later uses of the original tensor or its views stale. Check mutation
  of a view and of its base, values computed before mutation, in-place return
  aliases, and `out=` destinations when the change affects them.
- Preserve the distinction between views and copies. Reshape may create a
  copy for non-contiguous input, and functionalized `*_copy` variants should
  not inherit view aliases merely because their values and shapes match.
  Verify the captured graph and decoder alias contract instead of inferring
  aliasing solely from an operation's name.

## Regression coverage

Use the [PyTorch layer-test harness](../../../../tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py)
to identify the exercised path. `PYTORCH_TRACING_MODE=EXPORT` selects export,
`TORCHFX` selects the compile backend, and the ordinary path uses TorchScript
with its scripting or tracing options. Confirm these settings in the target
checkout rather than relying on a test name or marker alone.

For shared translator changes, check coverage of the affected capture paths
when their schemas or metadata differ. For path-specific changes, require
coverage of that path without demanding unsupported capture modes. Verify
that the expected operator and overload survive capture and any decomposition;
otherwise a passing numerical test may never exercise the changed translator.

Choose cases that expose the changed rule: mixed dtypes and scalar kinds for
promotion, multiple permitted shapes for symbolic behavior, or observable
base/view updates for mutation. Check converted types when dtype behavior is
the subject of the change; numerical agreement alone does not establish type
fidelity. Distinguish capture, conversion, and plugin inference failures when
reporting evidence.
