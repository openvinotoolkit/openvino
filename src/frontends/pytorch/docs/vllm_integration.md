# vLLM integration architecture

The `torch.compile` OpenVINO backend can run the per-step forward of vLLM's CPU model runner. vLLM keeps
scheduling, batching, the KV-cache pool and sampling; OpenVINO executes the model graph, including paged
attention. This page describes how the pieces fit together for developers of the frontend and the backend.
Installation, usage, environment variables and troubleshooting are in the
[user guide](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/README.md).

## Components

| Location | Role |
|---|---|
| [torchdynamo/vllm/](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/) | vLLM plugin package: model-runner patches, FX rewrites, side-channel binding, runtime hooks |
| [torchdynamo/](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/) | Generic `torch.compile` backend; calls the vLLM hooks at fixed points, see [torch.compile backend](torch_compile.md) |
| [src/op/paged_attention.cpp](../src/op/paged_attention.cpp), [src/op/gdn_attention.cpp](../src/op/gdn_attention.cpp) | Translators of the `ov_ext` operations the rewrites produce, see [custom operations](custom_operations.md) |
| [src/transforms/canonicalize_float_precision.cpp](../src/transforms/canonicalize_float_precision.cpp) | f32-activation form for bf16/f16 models, see [narrow default float type](default_float_type.md) |
| [NormalizeVLLMRoPE](../../../common/transformations/src/transformations/common_optimizations/normalize_vllm_rope.cpp) | Common optimization, run only for models with the `vllm_model` rt_info flag |
| [tests/vllm_tests](../../../../tests/vllm_tests/) | End-to-end and unit tests, see [testing](vllm_tests.md) |

The generic backend has no vLLM knowledge. Every vLLM-specific step lives in the `vllm` package and is reached
through a fixed set of hooks ([listed here](torch_compile.md#hooks-for-integrations)), each of which is a no-op
for graphs and options without the vLLM markers.

## Lifecycle of a model

```
vllm.general_plugins entry point -> plugin.register()
  patches CPUModelRunner.load_model, reload_weights, CPUWorker.finish_weight_update

CPUModelRunner.load_model (patched)
  1. load weights with the SGL kernel disabled
  2. options = {"aot_autograd": True, "vllm": True, "model_id": N, "default_float_type": model dtype}
  3. _own_kv_cache_update: attention layers report forward_includes_kv_cache_update=True
  4. install the fused sampler, widen the CPU affinity, raise the dynamo cache limit
  5. model.forward = torch.compile(model.forward, backend="openvino", options=options)
  6. optionally compile lm_head on OpenVINO (OV_LM_HEAD=1)

first forward (and every dynamo re-trace)
  aot_autograd -> fx_openvino -> Partitioner.make_partitions
     maybe_rewrite_paged_attention: attention / GDN / MoE rewrites
     CapabilityBasedPartitioner, then keep_paged_attention_partitioned
  per partition: openvino_compile
     TorchFXPythonDecoder(gm, default_float_type) -> fe.convert()
     compile_hooks.apply_post_convert / apply_input_shapes / apply_post_config
     core.compile_model

every step
  OpenVINOGraphModule -> openvino_execute -> runtime_hooks.run_pa_infer
     side_channel binds __pa__* inputs from vLLM's forward context, then req.infer()
```

`model_id` scopes the compiled-model caches to one model, so two models with the same architecture never share
compiled graphs. Weight reloads (`reload_weights`, `finish_weight_update`) drop those caches and recompile.

## The vLLM preset

`options["vllm"] = True` expands into per-flag defaults ([preset.py](../../../bindings/python/src/openvino/frontend/pytorch/torchdynamo/vllm/preset.py));
a flag set explicitly by the caller wins:

| Flag | Preset | Effect |
|---|---|---|
| `paged_attention` | on | Rewrite vLLM attention, GDN and MoE call sites into OV-translatable operations |
| `pa_translate` | on | Mark `ov_ext.paged_attention` / `ov_ext.gdn_attention` as supported, so they stay in OV partitions; with `paged_attention`, OV owns the KV-cache write |
| `no_fallback` | on | A failing conversion or OV execution raises instead of falling back to PyTorch |
| `canonical_precision` | on | Pass the model dtype to the decoder as `default_float_type` |
| `dynamic_shapes` | off | Tensor inputs are made fully dynamic only when every integer graph input could be tied to a tensor dimension (one compiled model then serves all token counts); otherwise they keep their traced shapes |
| `gdn_translate`, `moe_translate` | on (not preset flags) | Individual switches for the GDN and MoE rewrites |

The preset also fills the CPU compile config: `DYNAMIC_QUANTIZATION_GROUP_SIZE=32`, and an
`INFERENCE_PRECISION_HINT` / `KV_CACHE_PRECISION` pair derived from the model's float type
(`precision_config`). The two always name the same type, because the CPU PagedAttention kernel is only built for
matching pairs: bf16 on AMX-BF16 hardware (for f16 and f32 models too), f32 otherwise.
`OV_INFERENCE_PRECISION_HINT` and `OV_KV_CACHE_PRECISION` override the pair.

## Graph rewrites

vLLM's custom operations arrive in the FX graph as `auto_functionalized_v2` higher-order operations that OV
can't translate. `maybe_rewrite_paged_attention` replaces them before partitioning:

| vLLM operation | Replaced by | Lowered to |
|---|---|---|
| `vllm.unified_attention_with_output` | `ov_ext.paged_attention(q, k, v, layer_name, scale, kv_sharing_target)` | `PagedAttentionExtension` |
| `vllm.cpu_gdn_attention_core` (Gated DeltaNet) | `ov_ext.gdn_attention(...)` | `PagedCausalConv1D` + `PagedGatedDeltaNet` |
| `_C.cpu_fused_moe` | A tiled MoE block in aten ops | Three `GatherMatmul`s (CPU `ConvertTiledMoeBlockTo3GatherMatmuls`) |

`scale` and `kv_sharing_target` are read from the live layer (`layer.impl.scale`,
`layer.kv_sharing_target_layer_name`). A failing GDN or MoE rewrite leaves those call sites on vLLM's operation;
the attention rewrite is mandatory when OV owns the KV-cache write (below).

## KV-cache ownership

OV's PagedAttention writes the new K/V into the paged cache itself. The plugin therefore moves that write from
vLLM to OV: `_own_kv_cache_update` gives each attention layer a backend subclass that reports
`forward_includes_kv_cache_update=True`, so vLLM drops its separate `unified_kv_cache_update` call. From then on
vLLM's own attention op would read a cache nobody writes, and several guards make sure it never runs:

* The model is refused at load if a layer uses an attention feature the translator can't express: ALiBi slopes,
  attention sinks, logits soft-capping or non-decoder attention (`_unsupported_attention_features`).
* An attention call site that can't be rewritten raises (`maybe_rewrite_paged_attention`).
* After partitioning, an attention call left outside every partition gets a partition of its own, and one OV
  doesn't support raises (`keep_paged_attention_partitioned`). Without this the partitioner would drop
  single-node partitions and run the call's eager fallback.
* A layer that shares another layer's KV cache must be in the same partition as its target, or compilation
  raises: the translator can only reuse the target's K/V inside one conversion.
* `no_fallback` turns conversion and execution failures into errors, and the dynamo cache limit is raised so a
  recompile storm does not silently drop to eager.
* vLLM's profile and warm-up runs (no attention metadata) run the eager graph on purpose; their output is never
  used. A real step without a forward context fails in the binder instead.

## Side channel

The KV cache, block tables and sequence metadata are not FX graph inputs. The translators create extra
Parameters for them, named `__pa__<layer>__<field>` (per layer) and `__pa__shared__<field>` (shared by all
layers), plus `__pa__gdn__...` for GDN state. `compile_hooks.register_pa_parameters` adds them to the model's
inputs, and at every step `side_channel.py` binds them from `vllm.forward_context.get_forward_context()`:

| Parameter | Source |
|---|---|
| `__pa__<layer>__key_cache`, `value_cache` | The layer's `kv_cache` tensor, split into its K and V halves without a copy; a vLLM block size that is a multiple of 32 is viewed as 32-token blocks |
| `__pa__<layer>__block_indices`, `block_indices_begins` | `attn_metadata.block_table` and `seq_lens`, flattened per sequence |
| `__pa__<layer>__sliding_window` | `layer.impl.sliding_window` (0 for full attention) |
| `__pa__shared__seq_lens`, `query_start_loc` | `attn_metadata`; `past_lens` and `max_context_len` are derived in the graph |
| `__pa__gdn__<layer>__gdn_conv_state`, `gdn_recurrent_state` | Compact copies of the layer's conv and SSM state slots, scattered back after inference |

The KV-cache tensors alias vLLM's buffers, so OV writes the real cache. Their Parameters are retyped to
`KV_CACHE_PRECISION` before compilation (`retype_kv_cache_parameters`): a type mismatch would make the CPU plugin
insert a Convert, and the in-place write would land in that Convert's temporary. With KV sharing, the sharing
layer is bound to its target layer's cache.

## Execution fast path

`runtime_hooks.infer_with_pa` binds the positional inputs and the side-channel inputs in port order. With
`OV_FAST_INFER=1` (the plugin's default) it calls `set_tensor` only for ports whose value changed and reuses
output views of statically-shaped outputs. Binding failures fall back to the plain `req.infer()` call;
failures after inference started propagate, because inference has already updated the KV cache and GDN state.

## Invariants for changes

* Keep vLLM knowledge in the `vllm` package; generic backend files only call the existing hooks.
* Anything that can leave an attention call outside OV, or split a KV-sharing pair, must raise.
* Side-channel Parameter names are a contract between the translators and `side_channel.py`; change both
  together.
* New CPU-config defaults go through the preset so caller-supplied keys keep priority.
