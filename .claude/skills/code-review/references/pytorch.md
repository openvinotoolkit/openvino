# PyTorch Frontend Review

Apply alongside [frontend guidance](frontend.md) to PyTorch translators,
passes, decoders, and conversion tests.

## Capture and dispatch

- Identify the actual capture path: scripted/traced
  [TorchScript](../../../../src/bindings/python/src/openvino/frontend/pytorch/ts_decoder.py),
  `torch.export.ExportedProgram`, or `torch.compile` through the
  [FX decoder](../../../../src/bindings/python/src/openvino/frontend/pytorch/fx_decoder.py).
  Inspect the captured graph and configured decomposition/functionalization;
  sharing a decoder does not imply identical operators, metadata, or mutations.
  Do not assume decompositions run or require them just to hide missing support.
- Trace dispatch through the [operation table](../../../../src/frontends/pytorch/src/op_table.cpp)
  and [translation session](../../../../src/frontends/pytorch/src/translate_session.cpp).
  `aten::add` and `aten.add.Tensor` may use shared translators or FX overrides.
  Check overload schemas, keyword arguments, omitted defaults, `None`, and
  inlined scalars/lists before sharing logic across capture paths or versions.
- `context.input_is_none(i)` checks bounds and returns `true` when
  `i >= context.get_input_size()`, as well as for an explicit `None`. It can
  guard optional inputs without a separate range check; do not flag calls
  beyond the input count as out-of-bounds access.
- Distinguish capture specialization from conversion defects. Do not turn
  symbolic shape calculations into example-input constants, or reorder user
  inputs, parameters, buffers, and flattened container outputs.

## Type alignment

- Reuse [promotion helpers](../../../../src/frontends/pytorch/src/utils.cpp)
  (`align_eltwise_input_types`, `get_inputs_with_promoted_types`,
  `is_python_scalar_input`, `align_output_types`) and
  [wrappers](../../../../src/frontends/pytorch/src/utils.hpp) where their rules apply.
- Distinguish Python scalars, zero-dimensional tensors, and other tensors.
  Promotion depends on numeric category and operand kind, not just stored
  dtype. Preserve decoder `PyScalar` information and pass correct scalar flags.
- Align operands before arithmetic; casting afterward cannot repair overflow,
  truncation, or wrong division. Check result types separately, especially
  comparisons, reductions, integer true division, and explicit/default dtype.
  Decoder output metadata may be dynamic, so it cannot replace promotion logic.
- Apply operator-specific rules: in-place destination dtype, allowed `out=`
  casts, and scalar arguments such as `alpha` differ from ordinary promotion.
  When dtype decoding changes, keep Python mappings and C++ ScalarType
  interpretation consistent. Use shared frontend rules for 64-bit tradeoffs.

## Mutation and aliases

Follow `NodeContext::mutate_input` and session alias handling, not only the
returned value: later base/view uses must see updates, while values computed
before mutation remain unchanged. Check in-place return aliases and `out=`
destinations when affected. Preserve view/copy distinctions: non-contiguous
reshape may copy, and `*_copy` variants must not gain aliases merely because
values match. Use the captured graph and decoder alias contract as evidence.

## Regression coverage

Confirm the actual mode in the
[layer-test harness](../../../../tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py):
`PYTORCH_TRACING_MODE=EXPORT` selects export, `TORCHFX` selects the compile
backend, and the ordinary path uses TorchScript scripting/tracing options.
Check affected paths when shared logic receives different schemas or metadata;
do not demand unsupported capture modes.

Verify the expected operator/overload survives capture and decomposition.
For the changed rule, check mixed dtype/scalar kinds, multiple permitted
shapes, or observable mutation of bases/views. Assert converted types for
dtype changes; numerical agreement alone does not establish type fidelity.
