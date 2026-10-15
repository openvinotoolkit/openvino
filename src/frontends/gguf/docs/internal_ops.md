# GGUF Frontend — Internal Operations Policy

## Summary

OpenVINO frontends are, as a rule, thin translators: they emit only ops from the
public opsets so that the produced `ov::Model` is portable and serializable to IR.
The GGUF frontend makes a **scoped, deliberate exception**: it is allowed to emit
ops from the shared internal opset (`ov::op::internal`, defined in `core`'s
`dev_api`) when doing so materially simplifies translation or lets the target device
use a fused kernel instead of a hand-built subgraph.

This is a pragmatic tradeoff, not a general license. It applies to the GGUF frontend
only, and only to ops that already ship as part of OpenVINO core (with registered
`type_info`, shape inference, a reference implementation, and plugin support). The
frontend never defines its own operations.

## Why this is acceptable here

Unlike a frontend-private op, a shared `ov::op::internal` op is known to the rest of
the stack:

- It has a registered `type_info`, so plugins link against it, and passes,
  matchers, and `visualize_tree` recognize it.
- Plugins that support it natively keep it (e.g. the fused CPU/GPU
  `GatedDeltaNet` kernels); plugins that do not can decompose it in their own
  pipeline. No mandatory frontend-side decomposition is forced on every consumer.
- Shape inference and a reference `evaluate` already exist and are tested in core,
  so `PartialShape` propagation and CPU fallback work without extra frontend code.

The GGUF frontend is also a `LINKABLE_FRONTEND` consumed directly by the
`llama.cpp` `ggml-openvino` backend and by OpenVINO GenAI, both of which build and
run the model in-process on a device. Neither relies on IR serialization, so the
main cost of internal ops (below) does not affect the primary use cases.

## The cost: models are not IR-serializable

An `ov::op::internal` op is **not** part of any IR serialization opset. A model that
contains one cannot round-trip through `ov::save_model` / the IR frontend, which also
means the following flows do not work for such a model:

- `ov::save_model`, `ovc`, and `benchmark_app -o`
- model caching (`ov::cache_dir`) and offline `compile_model` blob export
- weightless-cache flows

**Important behavioral note:** serialization does not currently fail loudly. An
internal op has no opset `version_id`, so `ov::pass::Serialize` writes it with
`version="experimental"` and `save_model` *succeeds*. The resulting IR is unloadable
and fails only later, at deserialize time, with an opaque "operation is not
registered" error. Do not rely on serialization to reject these models — treat any
model that went through an internal-op translation path as non-serializable by
construction.

If a serializable model is required, use a translation path built only from public
opset ops. Where an internal-op path has a core-op fallback, that fallback is
serializable — see the example below.

## Current internal ops emitted

| Op | GGML op | Path | Fallback |
|---|---|---|---|
| `ov::op::internal::GatedDeltaNet` | `GGML_OP_GATED_DELTA_NET` | `translate_gated_delta_net` (scalar gate) | `translate_gated_delta_net_ref` — a serializable `Loop` scan, used for per-key-dimension gating (`kda`) and as a portable fallback |

See [`src/op/gated_delta_net.cpp`](../src/op/gated_delta_net.cpp).

## Guidance for adding a new internal-op path

1. Only emit ops that already exist in `ov::op::internal` (core `dev_api`) with
   plugin support — do not define new ops in the frontend.
2. Prefer keeping a core-op-only reference path (as `translate_<op>_ref`) so callers
   that need serialization, or devices without native support, have a route.
3. Update the table above and note the non-serializable consequence.
4. Be aware `dev_api` internal ops are marked "under development and subject to
   change"; their inputs/attributes may shift between releases.
