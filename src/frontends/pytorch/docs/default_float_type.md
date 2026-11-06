# Narrow default float type

A model that PyTorch built with a bf16 or f16 default dtype arrives at the frontend with every activation and
weight in that type. OpenVINO's standard form is different: weights are compressed behind a decompression
`Convert`, activations are f32, and the execution precision is chosen by the `INFERENCE_PRECISION_HINT` property.
When a decoder reports a narrow *default float type*, the frontend converts the model into that standard form.
Today only the `torch.compile` vLLM path does this; `convert_model`, TorchScript and torch.export conversions are
unaffected unless their decoder is given the type explicitly.

## Decoder interface

| Where | API |
|---|---|
| C++ | `TorchDecoder::get_default_float_type()`, last virtual method of the interface; returns `element::f32` unless overridden |
| Python | `TorchDecoderBase.get_default_float_type()`; `TorchFXPythonDecoder(..., default_float_type=...)` and `TorchScriptPythonDecoder(..., default_float_type=...)` accept a torch float dtype or an OpenVINO type |

Only the top-level decoder is read. A non-floating-point value is rejected by the decoder and by `normalize()`.
The `torch.compile` backend passes `options["default_float_type"]` to the decoder when the `canonical_precision`
option is on. The vLLM preset turns it on, and the vLLM plugin sets `default_float_type` to vLLM's model dtype
(see [vLLM integration](vllm_integration.md#the-vllm-preset)).

## Conversion steps

1. `TranslateSession::translate_graph` records a non-f32 type in the model's rt_info as `default_float_type`.
2. `FrontEnd::normalize()` reads it back and, for a narrow type, runs:
   * `CanonicalizeFloatPrecision` (below);
   * `ov::pass::MarkCompressedFloatConstants`, which marks the new Converts as decompression;
   * `ov::pass::EnableDecompressionConvertConstantFolding`, so the model is in the same state as one read from
     IR (the no-folding flag isn't serialized) and the plugin decides which Converts to keep.
3. The rt_info entry stays on the converted model, so the caller can pick the inference precision from it.

`MarkCompressedWeightsCast` runs in every `normalize()`, independent of the default float type. It marks the
trailing precision cast of a weight-decompression chain, for example
`Constant(i4) -> Convert(f16) -> Multiply(scale) -> Convert(bf16) -> MatMul`, as decompression, so the CPU
plugin folds the chain into `FullyConnectedCompressed` instead of keeping a runtime Convert and Multiply per
layer.

## `CanonicalizeFloatPrecision`

[transforms/canonicalize_float_precision.cpp](../src/transforms/canonicalize_float_precision.cpp)

* **Weights.** Narrow Constants with more than 1024 elements stay narrow and are read through a `Convert(f32)`.
  Smaller ones (scalars, eps, rotary tables) are folded to f32.
* **Activations.** The rest of the graph is retyped from bf16/f16 to f32 by `ov::pass::ConvertPrecision`. That
  includes operations whose output type is an attribute, such as Range, Eye and RandomUniform.
* **Boundaries.** Model inputs and outputs keep their types: narrow inputs are read through a Convert, and each
  output gets a Convert back to its original type that keeps its tensor names. The `__pa__` side-channel
  Parameters keep their type; their KV-cache precision is set later by the backend.
* **Explicit rounding.** A cast to bf16/f16, or a Range with a narrow output type, keeps its rounding when the
  value reaches a rounding-sensitive consumer: an operation with an integer or boolean output (Convert,
  comparisons, TopK indices, ...) or Floor, Ceiling, Round, Sign, Mod, FloorMod. The search follows
  value-preserving operations: widening or same-type float Converts and pure data movement (Reshape,
  Squeeze/Unsqueeze, Transpose, Gather, Slice/StridedSlice, Concat, Broadcast, Tile, Split/VariadicSplit).
  `x.to(torch.bfloat16).float().to(torch.int32)` therefore still returns 2 for 1.999. Narrow *arithmetic*
  is not rounded: its float math runs in f32 by design.
* **Scope.** Models with subgraph operations (If, Loop, ...) are left unchanged, because a body keeps the types
  it was converted with.

## Tests

The `test_canonicalize_float_precision_*`, `test_decoder_default_float_type` and `test_*weights*cast*` cases in
[test_torch_frontend.py](../../../../tests/layer_tests/py_frontend_tests/test_torch_frontend.py) cover the
pass, the decoder API and `MarkCompressedWeightsCast`. Rounding cases are parametrized in
`test_canonicalize_float_precision_keeps_rounding_before_sensitive_consumer`; add new patterns there with an
input whose narrow rounding changes the integer result. See [testing](vllm_tests.md) for how to run them.
