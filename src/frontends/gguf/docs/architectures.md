# Adding and porting architectures

The native `.gguf` path builds an OpenVINO graph with no llama.cpp dependency. Builders emit nodes in
the GGML operation vocabulary (`GGML_OP_MUL_MAT`, `GGML_OP_ROPE`, ...), so the same converters in
[`src/op/`](../src/op) serve native files and llama.cpp cgraphs.

Every architecture is an `ArchitectureDefinition`: a handler id, the `general.architecture` string,
a `ModelBuilder` factory and an optional metadata predicate. The built-in catalog and an external
`ArchitectureExtension` consume the same definition and pipeline, so code written as a plugin can be
promoted unchanged. Registration and packaging are covered in [extensions.md](extensions.md).
A vision/audio branch inside an mmproj file is a [projector component](extensions.md#extend-mmproj-with-a-projector-component), not an architecture.

## Choose the smallest implementation

| Requirement | Implementation |
|---|---|
| Existing decoder topology, new name | Catalog row or `make_decoder_architecture(name, rope)` |
| Decoder facts that tensors/metadata cannot determine | Add a callback returning `DecoderOptions` |
| Custom layer order using existing sublayers | `ModelBuilder` with `configure_decoder`, `decoder_attention`, `decoder_ffn` |
| Different model family | `ModelBuilder` with generic operations; follow [porting from llama.cpp](#port-from-llamacpp) |

Prefer structural detection to options. A custom topology owns its embedding, normalization,
residual and output ordering; the shared sublayer methods do not assemble those.

## Builder layout

| File | Responsibility |
|---|---|
| [`graph_emitter.hpp`](../src/builder/graph_emitter.hpp) | `add_op` / `add_input` / `add_weight` through the shared converters and shape inference |
| [`blocks/`](../src/builder/blocks) | Reusable fragments: `common`, `ffn` (dense/GeGLU/MoE), `attention`, `gated_delta_net`, `qkv_repack` |
| [`decoder_config.hpp`](../src/builder/decoder_config.hpp) | Per-architecture detection and per-layer accessors |
| [`arch/decoder_builder.cpp`](../src/builder/arch/decoder_builder.cpp) | Decoder assembly order |
| [`arch_registry.cpp`](../src/builder/arch_registry.cpp) | The `architectures` catalog returned by `builtin_architectures()` |
| [`model_kind.hpp`](../src/builder/model_kind.hpp) | Diagnostic family detection for an unclaimed file |
| [`gguf_builder.cpp`](../src/builder/gguf_builder.cpp) | Parse, resolve the definition, invoke its builder |

One generic `DecoderBuilder` covers the llama family. Unlike llama.cpp it derives tensors from the
file's tensor table, so a compatible architecture needs only registration.

## Reuse the decoder

Add a catalog row in [`arch_registry.cpp`](../src/builder/arch_registry.cpp) with the RoPE mode from
the reference implementation (`Normal` rotates pairs, `Neox` halves, `Interleaved` is multimodal RoPE):

```cpp
{"your-arch", RopeMode::Neox},
```

Custom builders such as Mamba and `clip` register a factory instead and own their positional encoding:

```cpp
{"mamba2", make_mamba2_builder},
```

When tensor names cannot determine a choice, define the architecture with
`make_decoder_architecture(name, rope, options_callback)` and register its factory in the catalog.
Options in [`decoder_options.hpp`](../dev_api/openvino/frontend/gguf/builder/decoder_options.hpp)
such as `qk_norm_after_rope`, `post_norm_only`, `normalize_expert_weights` and `rope_skip_period`
describe ordering and routing facts, not execution plans. Unset fields keep native detection; the
callback runs before dependent SWA, RoPE and KV configuration is resolved. For example, EXAONE4's
`post_attention_norm` is a post-norm, while GPT-OSS uses the same name for a pre-FFN norm.

### Auto-detected features

`decoder_config_from_meta()` reads hyperparameters once and `DecoderConfig` resolves them; the
topology builder never rereads metadata.

| Feature | Detected from |
|---|---|
| Per-head Q/K norm (qwen3, hunyuan) | `blk.0.attn_q_norm.weight` |
| Full-width Q/K norm (OLMoE, MiniMax M2) | `attn_q_norm.weight` width == `n_head*head_size` |
| Q/K/V and output biases | `blk.0.attn_q.bias`, `blk.0.attn_output.bias` |
| Fused QKV (phi-3, minicpm) | `blk.0.attn_qkv.weight` |
| Fused gate+up FFN (phi-3) | Absence of `blk.0.ffn_gate.weight` |
| MoE routing | `ffn_gate_exps.weight` on the first routed layer |
| Shared experts | `ffn_*_shexp.weight`, including files without `expert_shared_count` |
| Dense-lead / interleaved MoE | `leading_dense_block_count`, `interleave_moe_layer_step` |
| RoPE frequency factors | `rope_freqs.weight` |
| Scalar scales (minicpm) | `embedding_scale` / `residual_scale` / `logit_scale` |
| Soft-caps (gemma2/3) | `attn_logit_softcapping` / `final_logit_softcapping` |
| Sliding-window attention | `attention.sliding_window(_pattern)`, or sinks |
| Per-layer KV heads | `attention.head_count_kv` as an array |

Per-layer variation goes through `DecoderConfig` accessors, not inline ternaries in `build_layer()`:
`layer_is_swa`, `layer_is_moe`, `layer_head_size`, `layer_n_head_kv`, `layer_kq_scale`,
`layer_rope_config` and `is_recurrent_layer`. Extend them for a new per-layer dimension.

### Extend the decoder topology

The generic block is `norm -> QKV -> RoPE -> attention -> norm -> FFN/MoE -> residual`. Past
extensions include MoE routing (`blocks::moe_ffn()`), gpt-oss sinks and OAI SwiGLU, jais2 biased
LayerNorm and ReLU² FFN, gemma2/3 post-norms and soft-caps, gemma4 per-layer embeddings and shared KV,
and qwen35 Gated DeltaNet layers that return the same sublayer output as attention. To add one:

1. Add detection in `DecoderConfig`. Prefer weight presence; use `arch == "..."` only when the tensor
   table is ambiguous (for example GeGLU versus SwiGLU).
2. Emit it behind that flag in the relevant `blocks/` function, or in `DecoderBuilder::build_layer()`
   when it changes sublayer order.
3. Add a converter only if a GGML operation is missing; see [how_to_add_op.md](how_to_add_op.md).

Do not add `DecoderConfig` flags for non-decoder families such as encoders; write a `ModelBuilder`.

## Write a custom builder

Implement `ModelBuilder::build()` with a `GgufGraphContext`. `BuildContext` borrows metadata and
weights for the synchronous factory/build call; a builder may copy it but must not use it afterward.
The returned graph retains weight storage.

- Scalar getters `get_int`, `get_float`, `get_bool`, `get_str` return `std::optional`; use
  `value_or` only for defaults present in the reference. Array getters return typed vectors.
- `tensors.require(name)` loads a mandatory weight, `tensors(name)` returns an empty value for an
  absent optional weight, and `tensors.has(name)` checks presence without emitting a node.
- Inputs have a known rank of at most four; dimensions may be dynamic.

For a custom decoder layer order, initialize the shared blocks once:

```cpp
auto dimensions = graph.configure_decoder(RopeMode::Neox, options);
graph.build_inp_pos();
graph.build_attn_inp_kv();
// Inside the custom layer loop, after the appropriate normalization:
auto attention = graph.decoder_attention(layer, normalized_input);
auto feed_forward = graph.decoder_ffn(layer, normalized_ffn_input);
```

These call the existing attention, Gated DeltaNet and dense/GeGLU/MoE blocks. The returned dimensions
are a snapshot. `decoder_layer_parameters(layer)` supplies resolved heads, scale and RoPE parameters
for a custom attention fragment. The Qwen3 builder in
[`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp) owns its layer order
this way; `VisionEncoderBuilder` in the same file is a larger non-decoder family.

### Declare model contracts

Graph nodes alone do not tell consumers how to maintain state or form masks:

- `configure_decoder` records resolved RoPE and sliding-window metadata automatically.
- For custom RoPE, call `configure_rope(config)` before emitting nodes. Set `config.per_op` for
  per-node tables and `config.is_imrope` for multimodal positions, then pass a matching `rope_config`.
- `set_sliding_window(tokens)` records the window of a custom attention implementation.
- `add_recurrent_state(input, update)` declares an overwritten state for `GGUFMakeStateful`. It is not
  the declaration for an append-only KV cache; reuse the shared attention blocks for that.
- `set_output(value)` appends an output; `set_primary_output(value)` puts it before auxiliary outputs
  such as KV caches. `finish()` validates the graph and seals the context.

State and GenAI adaptation remain caller choices; see [runtime.md](runtime.md).

## Port from llama.cpp

Use this route when the shared decoder blocks cannot express the computation: a new encoder, a
different recurrent network or an encoder-decoder family. llama.cpp is the reference and oracle,
not a dependency of the result.

### 1. Trace the reference

Record the llama.cpp revision, checkpoint, quantization, preprocessing and runtime options. Paths are
relative to that checkout:

| Source | What to extract |
|---|---|
| `src/models/<arch>.cpp`: `load_arch_hparams` | Required metadata, defaults, variants |
| Same file: `load_arch_tensors` | Tensor names and dimensions, optional and tied weights |
| Same file: `build_arch_graph` | Inputs, layer order, branches, residuals, output selection |
| `src/llama-model.cpp`, `src/llama-arch.cpp`, `src/llama-hparams.h` | Shared metadata, `LLM_KV_*` / `LLM_TENSOR_*` names, derived dimensions |
| `src/llama-graph.cpp` and the memory implementation | Helper bodies, masks, cache and recurrent updates |
| `tools/mtmd/models/` | Vision/audio graphs, patching, pooling and projection |
| Conversion scripts and `gguf-py/gguf/` | Exported names, packing, transpositions |

```sh
rg -n 'load_arch_hparams|load_arch_tensors|build_arch_graph|::graph' src/models/<architecture>.cpp
rg -n 'build_lora_mm|build_norm|build_ffn|build_attn' src/llama-graph.cpp
```

Expand every helper: `build_lora_mm` can add a weight scale and LoRA updates, and `build_norm` selects
RMS, layer or group normalization with optional bias. Write down the full input-to-output sequence,
including embedding scales, learned positions, final norm, pooling, output rows and logit transforms.
Matching tensor names do not imply the same computation.

### 2. Define the file and model interface

Write down the literal `general.architecture` and any distinguishing predicate; required metadata,
types and defaults; required and optional tensors with logical GGML dimensions; each input's name,
type and dynamic axes; outputs; and cache/state, positions and masks. Reject missing mandatory fields.
Derive weight dimensions from `GgufValue::ne` or its shape, not packed byte sizes. GGML
`[width, items, 1, 1]` is OpenVINO `[1, 1, items, width]`; that is not a batch guarantee.

An encoder need not call `configure_decoder` or emit logits. An encoder-decoder port must build both
computations and their connection; registration does not generate them.

### 3. Translate graph construction

Port one fragment at a time, keeping operand order. Use shared blocks only where semantics match.

| llama.cpp | Builder |
|---|---|
| `create_tensor(tn(...))` | Look up the GGUF name through `GgufTensors` |
| `ggml_new_tensor_*` for runtime data | `add_input(name, type, shape)` |
| `ggml_add` / `ggml_sub` / `ggml_mul` | `node("GGML_OP_ADD" / "GGML_OP_SUB" / "GGML_OP_MUL", {a, b})` |
| `ggml_mul_mat(ctx, weight, x)` | `node("GGML_OP_MUL_MAT", {weight, x})` |
| `ggml_rms_norm(ctx, x, eps)` | `node("GGML_OP_RMS_NORM", {x}, 0, {{"eps", eps}})`; learned scale is a separate multiply |
| `ggml_reshape_*` | `GGML_OP_RESHAPE` case 6 with OpenVINO-order `reshape_target`, optional `special_zero` |
| `ggml_permute` | `GGML_OP_PERMUTE` case 1 with OpenVINO-order `perm` |
| `ggml_view_*` | Converter attributes describing the logical slice; never a ggml pointer |
| `ggml_top_k(ctx, scores, k)` | `GGML_OP_TOP_K` with `int64_t` attribute `k` |
| `ggml_cpy` / `ggml_set*` for state | The updated tensor and its state relationship, not an in-place side effect |
| `cb(...)`, scheduling | Nothing to emit |

Layout rules:

- `ggml_permute` gives each source axis's destination; OpenVINO `perm` lists source axes in output
  order. For GGML axes `axes[i]`, `perm[3 - axes[i]] = 3 - i`. Derive it from axis meanings.
- VIEW case 3 accepts `view_slice = {ov_axis, start, length}` and optional `view_reshape` for a
  contiguous subrange. Strided views need a suitable converter or explicit layout operations.
- CONT case 1 is a passthrough, but `ggml_cont_2d/3d/4d` still carries a reshape. Keep real transposes.
- Preserve dynamic token/item axes. A one-token graph does not validate broadcasting.

### 4. Implement a whole-model builder

This complete builder implements an **illustrative** `example-encoder` contract. Its reference reads
`block_count`, `embedding_length` and an RMS epsilon (default `1e-5`), consumes feature vectors and
applies `cur += down(silu(gate(norm(cur))) * up(norm(cur)))` per layer with tensors
`blk.N.{norm,gate,up,down}.weight`, optional `.bias` and `output_norm.weight`:

```cpp
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace example {
using namespace ov::frontend::gguf;

class FeatureEncoder : public ModelBuilder {
public:
    explicit FeatureEncoder(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto& metadata = graph.metadata();
        const auto count = metadata.get_int("example-encoder.block_count");
        const auto width = metadata.get_int("example-encoder.embedding_length");
        OPENVINO_ASSERT(count && *count > 0, "Missing or invalid block_count");
        OPENVINO_ASSERT(width && *width > 0, "Missing or invalid embedding_length");
        const auto eps = static_cast<float>(
            metadata.get_float("example-encoder.attention.layer_norm_rms_epsilon").value_or(1e-5));
        OPENVINO_ASSERT(eps > 0, "RMS epsilon must be positive");
        auto tensors = graph.tensors();
        auto cur = graph.add_input("features", ov::element::f32, {1, 1, -1, *width});

        const auto linear = [&](const std::string& base, const GgufValue& input) {
            auto out = graph.node("GGML_OP_MUL_MAT", {tensors.require(base + ".weight"), input});
            if (auto bias = tensors(base + ".bias")) {
                out = graph.node("GGML_OP_ADD", {out, bias});
            }
            return out;
        };

        for (int64_t layer = 0; layer < *count; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto residual = cur;
            auto norm = graph.build_norm(cur, tensors.require(prefix + "norm.weight"), eps);
            auto gate = graph.node("GGML_UNARY_OP_SILU", {linear(prefix + "gate", norm)});
            auto up = linear(prefix + "up", norm);
            auto down = linear(prefix + "down", graph.node("GGML_OP_MUL", {gate, up}));
            cur = graph.node("GGML_OP_ADD", {residual, down});
        }

        graph.set_output(graph.build_norm(cur, tensors.require("output_norm.weight"), eps));
        return graph.finish();
    }

private:
    BuildContext m_context;
};

ArchitectureDefinition new_family_architecture() {
    return {"example.feature-encoder", "example-encoder", [](const BuildContext& context) {
                return std::make_shared<FeatureEncoder>(context);
            }};
}
}  // namespace example
```

Replace the contract and layer body with the traced reference. Do not keep its activation,
normalization or residual order because dimensions happen to match. The
[projector example](../examples/architecture_extension/projector.cpp) is a smaller runnable family.

### 5. Handle missing operations and state

Inventory the operations in the expanded reference helpers against [`op_table.cpp`](../src/op_table.cpp).
For a missing one, express it exactly with existing nodes or add a converter: externally as a
[`ConversionExtension`](extensions.md#add-or-override-an-operation-converter)
(`ConverterRegisteredAfterLoadInfersBuilderValues` is a working test), or built in through
[how_to_add_op.md](how_to_add_op.md). Converters are the only operation-support layer: never add a
builder-specific enum, shape rule or `GgufGraphContext` method. An unsupported quantization format
belongs to the weight loader in [`src/quant/`](../src/quant).

For state, make previous-state inputs, the update and retained outputs explicit and declare them as
in [Declare model contracts](#declare-model-contracts).

## Tensor operations and shapes

`node` is the single operation API: it invokes the registered converter immediately, and the
resulting `GgufValue` exposes OpenVINO's inferred `shape()`, `type()` and reversed-GGML `ne(i)`.

```cpp
auto sum = graph.node("GGML_OP_ADD", {a, b});
auto projected = graph.node("GGML_OP_MUL_MAT", {weight, sum});
auto experts = graph.node("GGML_OP_TOP_K", {scores}, 0, {{"k", int64_t{2}}});
auto heads = graph.node("GGML_OP_RESHAPE", {projected}, 6,
                        {{"reshape_target", std::vector<int64_t>{0, -1, n_heads, head_size}},
                         {"special_zero", true}});
auto merged = graph.node("GGML_OP_RESHAPE", {heads}, 2, {{"merge_heads", true}});
auto half = graph.node("GGML_OP_CPY", {input}, 0, {{"dst_type", ov::element::f16}});
```

Operands follow GGML order (weight before activation). `op_case` selects a converter's semantic
variant; attributes carry operation parameters (`reshape_target`, `perm`, `view_slice`, `repeats`,
GGML-order `concat_axis`, TopK `k`), never inferred output metadata. Architecture code does not supply
output types; converters apply the [result-type rules](how_to_add_op.md#translator-shape), and only
casts and `IM2COL` take `dst_type`. The builder API is not a drop-in `llm_graph_context`; local
functions may name repeated fragments without extending it.

## Promote an extension into OpenVINO

1. Move the definition source and header into `src/builder/arch/`, unchanged.
2. Include the header in `arch_registry.cpp` and add the name, factory and optional predicate to the
   `architectures` catalog. A plain decoder needs only a name and RoPE row.
3. Add the source to the explicit `FRONTEND_SRCS` list in `tests/CMakeLists.txt` and keep its tests.
4. Upstream any plugin converters into the shared table.
5. Update [supported_models.md](supported_models.md) with the validation actually run.

No dispatch branch or alternate graph implementation is needed. The tests compile the projector
example both as a library and into the catalog.

## Validate

Follow [testing.md](testing.md) for commands and [acceptance](testing.md#acceptance), in increasing
scope:

1. **Interface:** load a small nonzero fixture through the actual plugin; check missing-field errors,
   optional/tied weights, inputs and outputs, and CPU compilation.
2. **Fragments:** compare new layouts and operations against real ggml CPU with distinct dimensions
   and nonuniform values. llama.cpp's `cb` labels help match intermediates.
3. **Whole graph:** compare identical inputs. Decoders: full logits through prefill and several cached
   decode steps, multiple lengths, nonzero positions, state resets and window boundaries. Encoders:
   features or projections directly.
4. **Real checkpoint:** repeat with the real model and quantization, aligned tokenizer, preprocessing
   and precision. Record tested and untested variants.
5. **Regression:** keep a fixture. Decoders reuse [`gen_arch_accuracy.py`](../tests/gen_arch_accuracy.py)
   ([instructions](../tests/test_data/arch_accuracy/README.md)); a new family needs its own oracle.
   Update the fixture manifest and fingerprints, and add model-hub coverage. Shared changes rerun the
   existing architecture regressions.
