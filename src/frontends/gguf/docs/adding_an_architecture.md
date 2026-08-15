# Adding a new architecture to the GGUF frontend

The native `.gguf` path builds an OpenVINO graph from a GGUF file with **no llama.cpp
dependency**. It emits nodes in the GGML op vocabulary (`GGML_OP_MUL_MAT`, `GGML_OP_ROPE`,
`GGML_OP_FLASH_ATTN_EXT`, ...) that reproduce llama.cpp's cgraph topology, so the same op
translators (`src/op/*.cpp`) run for both the native path and the llama.cpp cgraph path.

## How the builder is laid out

`src/builder/` is layered; each layer knows strictly less than the one above it:

| File | Responsibility | Knows about |
|---|---|---|
| [`graph_emitter.hpp`](../src/builder/graph_emitter.hpp) | `add_op` / `add_input` / `add_weight` + shape & type bookkeeping | nothing about transformers |
| [`blocks/`](../src/builder/blocks) | reusable graph fragments: `common` (norm/scale/bias), `ffn` (dense/GeGLU/MoE), `attention`, `gated_delta_net`, `qkv_repack` | a decoder layer |
| [`decoder_config.hpp`](../src/builder/decoder_config.hpp) | all per-architecture detection + per-layer accessors | one model's hyperparameters |
| [`arch/decoder_builder.cpp`](../src/builder/arch/decoder_builder.cpp) | the order a decoder is assembled in | the whole decoder family |
| [`arch_registry.cpp`](../src/builder/arch_registry.cpp) | which architectures are accepted, and their RoPE mode | names only |
| [`model_kind.hpp`](../src/builder/model_kind.hpp) | which model *family* a file holds | raw metadata |
| [`gguf_builder.cpp`](../src/builder/gguf_builder.cpp) | parse → detect family → dispatch to a `ModelBuilder` | the entry point |

A single generic `DecoderBuilder` covers the whole "llama family" of decoder-only transformers.
This is deliberately **not** llama.cpp's one-file-per-architecture layout: llama.cpp needs that
because every architecture enumerates its tensors by hand, whereas this builder derives them from
the tensor table, so a same-family architecture costs zero lines of code.

## The 90% case: add a name

Most new architectures in the transformer family need **no code** — the builder auto-detects
their structure from the GGUF tensor table and metadata. To enable one, add its
`general.architecture` string to `verified_archs()` (or `experimental_archs()`) in
[`arch_registry.cpp`](../src/builder/arch_registry.cpp):

```cpp
const std::set<std::string>& experimental_archs() {
    static const std::set<std::string> archs = {
        "llama-embed", "exaone4", ...,
        "your-arch",   // <-- add here
    };
    return archs;
}
```

Then check whether RoPE is NEOX (rotate-halves) or NORMAL (rotate consecutive pairs) for the
arch and, if NEOX, add it to `arch_uses_neox_rope()` in the same file (mirror
`llama_model_rope_type` in llama.cpp). That is the whole change for a same-family arch.

### What is auto-detected (no code needed)

`DecoderConfig` infers structure from the presence of layer-0 weight tensors and from metadata:

| Feature | Detected from |
|---|---|
| Per-head Q/K norm (qwen3, hunyuan) | `blk.0.attn_q_norm.weight` |
| Full-width Q/K norm (OLMoE) | `attn_q_norm.weight` width == `n_head*head_size` |
| Q/K/V projection biases (qwen2) | `blk.0.attn_q.bias` |
| Output-projection bias | `blk.0.attn_output.bias` |
| Fused QKV (phi-3, minicpm) | `blk.0.attn_qkv.weight` |
| Fused gate+up FFN (phi-3) | absence of `blk.0.ffn_gate.weight` |
| MoE routing (OLMoE, gpt-oss, qwen3moe) | `blk.<lead>.ffn_gate_exps.weight` |
| Shared experts | `expert_shared_count` metadata + `ffn_*_shexp.weight` |
| Hybrid dense-lead MoE | `leading_dense_block_count` metadata |
| RoPE freq factors (llama-3, phi-3) | `rope_freqs.weight` |
| Scalar scales (minicpm) | `embedding_scale` / `residual_scale` / `logit_scale` metadata |
| Soft-caps (gemma2/3) | `attn_logit_softcapping` / `final_logit_softcapping` metadata |
| Sliding-window attention | `attention.sliding_window(_pattern)` metadata, or sinks |
| Per-layer KV heads | `attention.head_count_kv` as an array |

All hyperparameters are read once in `decoder_config_from_meta()`
([`src/quant/gguf.cpp`](../src/quant/gguf.cpp)) and resolved once in `DecoderConfig`; the topology
builder never re-reads GGUF metadata.

## Per-layer values: use the accessors, don't inline

Architectures with per-layer variation (SWA layers, variable KV heads, per-layer head sizes)
are handled by the per-layer accessors on `DecoderConfig` — the single source of truth,
so the topology stays declarative:

- `layer_is_swa(il)` — sliding-window layer? (per-layer flag array or period)
- `layer_head_size(il)` — head size (SWA layers may differ, e.g. gemma4)
- `layer_n_head_kv(il)` — KV head count (may vary per layer)
- `layer_kq_scale(il)` — attention softmax scale (`1/sqrt(layer_head_size(il))` unless overridden)
- `layer_rope_config(il)` — RoPE config (SWA layers may use a different freq_base / n_dims)
- `is_recurrent_layer(il)` — linear-attention (GDN) layer of a hybrid stack

If a new arch adds a per-layer dimension, extend these accessors rather than adding a new
inline ternary in `build_layer()`.

## The 10% case: structurally novel architectures need code

The generic decoder builder assumes the standard block: `norm -> QKV -> RoPE -> attention ->
norm -> FFN/MoE -> residual`. Architectures that break this shape need a new detected flag on
`DecoderConfig` plus a branch in the relevant block. Examples of what required code in the past:

- **MoE routing** — `blocks::moe_ffn()` (`MUL_MAT_ID` / `GatherMatmul`, top-k, gated activation).
- **gpt-oss** — attention sinks (5th `FLASH_ATTN_EXT` input), OAI gated activation
  (`GGML_GLU_OP_SWIGLU_OAI`), softmax-after-topk gating.
- **gemma2/3** — post-attention / post-FFN norms, attention & final-logit soft-caps.
- **gemma4** — per-layer input embeddings, shared-KV layers, per-op RoPE (SWA vs global differ).
- **qwen35** — a hybrid stack where 3 of every 4 layers are a Gated DeltaNet block
  (`blocks::gated_delta_net()`) instead of attention. It returns the same thing the attention
  block returns — the sublayer output before the residual — so the shared FFN tail is reused.

To add such a feature:
1. Add a detection line in `DecoderConfig`'s constructor (prefer weight-presence over an
   arch-name check — it generalizes to future archs; only fall back to `arch == "..."` when the
   tensor table is genuinely ambiguous, e.g. GeGLU-vs-SwiGLU).
2. Add the emission behind that flag in the relevant `blocks/` function, or in
   `DecoderBuilder::build_layer()` when it is about the ORDER of sublayers rather than their
   contents.
3. Add the op translator in `src/op/` if the feature needs a GGML op not yet handled, and
   register it in `op_table.cpp`.

## Adding a new model FAMILY (mmproj, audio, encoder-decoder)

An architecture is data; a **family** is code. A family is a distinct graph shape with its own
inputs and its own notion of a layer — a vision/mmproj encoder and an audio encoder are each one,
and neither is a causal decoder. Do **not** add flags to `DecoderConfig` for them.

Instead:

1. Detect it in [`model_kind.cpp`](../src/builder/model_kind.cpp). mmproj files set
   `general.architecture = "clip"` and carry `clip.has_vision_encoder` / `clip.has_audio_encoder`
   (llama.cpp `tools/mtmd/clip-impl.h`), so `detect_model_kind()` already classifies them; the
   check runs *before* any decoder hyperparameter is read, because those keys do not exist there.
2. Add a metadata reader next to `decoder_config_from_meta()` for that family's key layout, and a
   config struct next to `DecoderConfig`.
3. Subclass [`ModelBuilder`](../src/builder/model_builder.hpp) in `arch/`, reusing `GraphEmitter`
   and `blocks/common`. A ViT needs its own attention — non-causal, no KV cache, no RoPE — so it
   will not reuse `blocks::attention`; this is the same split llama.cpp makes between
   `llm_graph_context` and `clip_graph`.
4. Add a branch in `build_ggml_graph_from_gguf()`.

Nothing in the decoder family changes.

The downstream side already tolerates a non-decoder graph: `TranslateSession`'s LLM-specific
preprocessing is self-gating (`add_rope_sin_cos` only fires when `inp_pos` exists,
`add_sliced_mask` only when the mask and `token_len_per_seq` inputs exist), and the LLM-specific
passes (`MakeStateful`, `AdaptToGenAI`) are caller-registered rather than built in.

## Verifying a new architecture

1. **Converts + compiles**: convert through the frontend, then `core.compile_model(m, "CPU")`.
   The frontend is not auto-selectable, so ask for it by name:
   `fe = FrontEndManager().load_by_framework("gguf"); m = fe.convert(fe.load("model.gguf"))`.
2. **Graph is sane**: check the op-type histogram and that attention fused to
   `ScaledDotProductAttention` and MoE to `GatherMatmul`.
3. **Numerics**: run generation through OpenVINO GenAI (`greedy_causal_lm model.gguf "..."`) and
   compare to native llama.cpp (`build-ref/bin/llama-cli`) on the same prompt — the greedy tokens
   should match (small drift after ~dozens of tokens is expected from kernel differences).
4. **No graph regression** for existing archs: `tests/test_arch_conversion.cpp` converts every
   architecture fixture and asserts a pinned `(op count, input count)` fingerprint, so any
   restructuring of a supported architecture shows up there.
