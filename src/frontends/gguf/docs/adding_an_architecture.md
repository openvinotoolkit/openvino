# Adding a new architecture to the GGUF frontend

The native `.gguf` path builds an OpenVINO graph from a GGUF file with **no llama.cpp
dependency**. A single generic builder (`TransformerBuilder` in
[`src/builder/gguf_builder.cpp`](../src/builder/gguf_builder.cpp)) covers the whole "llama
family" of decoder-only transformers. It emits nodes in the GGML op vocabulary
(`GGML_OP_MUL_MAT`, `GGML_OP_ROPE`, `GGML_OP_FLASH_ATTN_EXT`, ...) that reproduce llama.cpp's
cgraph topology, so the same op translators (`src/op/*.cpp`) run for both the native path and
the llama.cpp cgraph path.

## The 90% case: add a name

Most new architectures in the transformer family need **no code** — the builder auto-detects
their structure from the GGUF tensor table and metadata. To enable one, add its `general.architecture`
string to `supported_archs()` in `gguf_builder.cpp`:

```cpp
const std::set<std::string>& supported_archs() {
    static const std::set<std::string> archs = {
        "llama", "qwen2", "qwen3", "phi3", ...,
        "your-arch",   // <-- add here
    };
    return archs;
}
```

Then check whether RoPE is NEOX (rotate-halves) or NORMAL (rotate consecutive pairs) for the
arch and, if NEOX, add it to `arch_uses_neox_rope()` (mirror `llama_model_rope_type` in
llama.cpp). That is the whole change for a same-family arch.

### What is auto-detected (no code needed)

The builder infers structure from the presence of layer-0 weight tensors and from metadata:

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

All hyperparameters are read once in `config_from_meta()`
([`src/quant/gguf.cpp`](../src/quant/gguf.cpp)); the builder never re-reads GGUF metadata.

## Per-layer values: use the accessors, don't inline

Architectures with per-layer variation (SWA layers, variable KV heads, per-layer head sizes)
are handled by the per-layer accessors on `TransformerBuilder` — the single source of truth,
so the main `build()` loop stays declarative:

- `layer_is_swa(il)` — sliding-window layer? (per-layer flag array or period)
- `head_size(il)` — head size (SWA layers may differ, e.g. gemma4)
- `n_head_kv(il)` — KV head count (may vary per layer)
- `kq_scale(il)` — attention softmax scale (`1/sqrt(head_size(il))` unless overridden)
- `layer_rope_config(il)` — RoPE config (SWA layers may use a different freq_base / n_dims)

If a new arch adds a per-layer dimension, extend these accessors rather than adding a new
inline ternary in the loop.

## The 10% case: structurally novel families need code

The generic builder assumes the standard decoder block: `norm -> QKV -> RoPE -> attention ->
norm -> FFN/MoE -> residual`. Architectures that break this shape need new code, added as a
new feature flag + a branch in the relevant `build_*` helper (or, for a genuinely different
graph shape, a dedicated builder — see "When to split the builder" below). Examples of what
required code in the past:

- **MoE routing** — `build_moe_ffn()` (`MUL_MAT_ID` / `GatherMatmul`, top-k, gated activation).
- **gpt-oss** — attention sinks (5th `FLASH_ATTN_EXT` input), OAI gated activation
  (`GGML_GLU_OP_SWIGLU_OAI`), softmax-after-topk gating.
- **gemma2/3** — post-attention / post-FFN norms, attention & final-logit soft-caps.
- **gemma4** — per-layer input embeddings, shared-KV layers, per-op RoPE (SWA vs global differ).

To add such a feature:
1. Add a detection line in the constructor (prefer weight-presence over an arch-name check —
   it generalizes to future archs; only fall back to `arch_str == "..."` when the tensor table
   is genuinely ambiguous, e.g. GeGLU-vs-SwiGLU).
2. Add the emission behind that flag in `build()` or the relevant `build_*ffn` helper.
3. Add the op translator in `src/op/` if the feature needs a GGML op not yet handled, and
   register it in `op_table.cpp`.

## When to split the builder (future)

`TransformerBuilder` inlines all feature variation via `m_has_*` flags in one `build()` loop.
That is fine for the transformer family. If a **structurally different family** lands (Mamba/SSM,
RWKV, MLA/deepseek-v2 latent attention, encoder-decoder), do **not** add more flags to the shared
loop — introduce an `ArchBuilder` base class with a per-family subclass (mirroring llama.cpp's
`src/models/*.cpp` one-file-per-arch design), dispatched by arch name. The current generic
builder becomes the default dense/MoE-transformer subclass.

## Verifying a new architecture

1. **Converts + compiles**: convert through the frontend, then `core.compile_model(m, "CPU")`.
   The frontend is not auto-selectable, so ask for it by name:
   `fe = FrontEndManager().load_by_framework("gguf"); m = fe.convert(fe.load("model.gguf"))`.
2. **Graph is sane**: check the op-type histogram and that attention fused to
   `ScaledDotProductAttention` and MoE to `GatherMatmul`.
3. **Numerics**: run generation through OpenVINO GenAI (`greedy_causal_lm model.gguf "..."`) and
   compare to native llama.cpp (`build-ref/bin/llama-cli`) on the same prompt — the greedy tokens
   should match (small drift after ~dozens of tokens is expected from kernel differences).
4. **No graph regression** for existing archs: the op-signature of every supported model must be
   unchanged (a quick sha256 over sorted `(op_type, output_shape)` pairs is a good gate).
</content>
