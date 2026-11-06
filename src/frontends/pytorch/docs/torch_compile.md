# torch.compile backend internals

`torch.compile(model, backend="openvino")` is implemented in
[torchdynamo/](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/) on top of the FX decoder. The
user-facing options are documented in the
[torch.compile article](../../../../docs/articles_en/openvino-workflow/torch-compile.rst). This page covers the
internal flow and the parts the vLLM integration relies on.

## Flow

| Step | File | What happens |
|---|---|---|
| Entry | [backend.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/backend.py) | `openvino()` receives the dynamo graph. With `aot_autograd` it goes through `aot_autograd(fw_compiler=fx_openvino)` with the backend's decompositions, otherwise straight to `fx_openvino` |
| Freezing or decomposition | backend.py | `fx_openvino`: with `aot_autograd`, weights are folded into Constants (`_freeze_static_inputs`, below); without it, `make_fx` applies the decompositions |
| Partitioning | [partition.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/partition.py), [op_support.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/op_support.py) | `CapabilityBasedPartitioner` groups supported nodes into `fused_*` submodules; unsupported nodes stay in PyTorch |
| Conversion | [compile.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/compile.py) | Each partition: `TorchFXPythonDecoder` -> `fe.convert()` -> input shapes -> `core.compile_model` |
| Execution | [execute.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/execute.py) | `OpenVINOGraphModule` runs the compiled partition, with caches of compiled models and infer requests |

## Weight freezing

`replace_params_with_constants` pairs `params_flat` with the leading placeholders, so it only freezes weights that
dynamo kept as module attributes. Since PyTorch inlines built-in modules unconditionally (2.13 made
`inline_inbuilt_nn_modules` a no-op), every weight is a graph input and `params_flat` is empty. The weights
would then reach OpenVINO as Parameters: a Transpose per MatMul, no constant folding, and wrong results.

`_freeze_static_inputs` freezes by `fw_metadata.static_input_indices` instead, which marks the weights on every
supported PyTorch version. It needs the real tensors, which `openvino()` stashes because `aot_autograd` hands the
compiler FakeTensors. Mutated or output-aliased inputs aren't frozen. If the indices are missing or the stash
doesn't match, the upstream helper runs as before.

## Fallback

* A failure while partitioning or compiling (`fx_openvino`) falls back to `compile_fx` (inductor), unless
  `no_fallback` is set.
* A failure while executing a partition falls back to the partition's PyTorch graph, permanently for that
  partition, unless `no_fallback` is set.

Both checks resolve `no_fallback` through the vLLM preset when the vLLM package is present.

## Compiled-model caches

[execute.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/execute.py) keeps compiled models
per partition and input signature. Two additions avoid recompiling on dynamo re-traces:

* **Structural cache.** The key is a hash of the graph's structure, so it survives re-traces that only rename
  placeholders or reorder them. Literal arguments are part of the key, and so is `options["model_id"]`, so two
  models with the same architecture don't share a compiled model. On by default only under the vLLM preset;
  `OV_STRUCTURAL_CACHE=1/0` forces it.
* **Shape-agnostic compile.** Under the vLLM preset, or for graphs with integer inputs, a graph whose integer
  inputs can all be rebuilt from a tensor dimension (or that has none, with `dynamic_shapes` on) compiles to one
  dynamic model for all shapes. The cache key then keeps only dtypes and ranks. `OV_SHAPE_AGNOSTIC_CACHE=0`
  disables it.

`execute.clear_caches()` drops these caches together with the vLLM binding caches, which are keyed by `id()` of
the dropped objects. Compiles made during vLLM's profile run are freed right after use
(`OV_EVICT_PROFILE_COMPILE=1`, default).

## Hooks for integrations

The generic files call the [vLLM package](vllm_integration.md) at fixed points, and each call is a no-op for
graphs and options without vLLM markers:

| Call site | Hook | Purpose |
|---|---|---|
| `op_support.OperatorSupport` | `vllm.maybe_register_pa_op` | Mark `ov_ext` attention operations supported |
| `Partitioner.make_partitions` | `vllm.maybe_rewrite_paged_attention`, `vllm.keep_paged_attention_partitioned` | Rewrite vLLM operations before partitioning; keep attention in OV after it |
| `compile.openvino_compile` | `compile_hooks.decoder_default_float_type` | Decoder `default_float_type`, see [narrow default float type](default_float_type.md) |
| | `compile_hooks.apply_post_convert` | `vllm_model` rt_info, register side-channel Parameters, fix Concat ranks |
| | `compile_hooks.apply_input_shapes` | Replace integer inputs with `ShapeOf` of their source tensor, set input shapes |
| | `compile_hooks.apply_post_config` | CPU config defaults, KV-cache Parameter types |
| `execute.openvino_execute` | `runtime_hooks.run_pa_infer` | Bind side-channel inputs and run inference |

Other options added for all callers: `perf_count` sets `PERF_COUNT=YES`, and `OV_INFERENCE_NUM_THREADS` sets
`INFERENCE_NUM_THREADS` on CPU when the config doesn't. `op_support` also lists `aten._grouped_mm`,
`aten.empty_like`, `aten.floor_divide` and `aten.logical_and/or/xor` as supported.
