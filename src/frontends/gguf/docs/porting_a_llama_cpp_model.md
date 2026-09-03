# Porting a llama.cpp model to the GGUF frontend

A new architecture can be added to the GGUF frontend **at runtime**, from outside the OpenVINO
build, by registering an [`ArchitectureExtension`](../dev_api/openvino/frontend/gguf/extension/architecture.hpp).
No frontend rebuild, no OpenVINO rebuild — a downstream project such as OpenVINO GenAI can enable a
model on its own.

This document covers the whole range: a same-family architecture that needs no code at all, and a
structurally new family (a vision encoder, an audio encoder, an encoder-decoder) written by porting
the corresponding `llama.cpp` model file.

For the in-tree route — adding an architecture to the frontend itself — see
[adding_an_architecture.md](adding_an_architecture.md). The decision between them is at the end.

## Three tiers of effort

Pick the lowest one that fits. Most architectures are Tier 1.

| Tier | What the architecture needs | What you write |
|---|---|---|
| 1 | Same family, structure derivable from the file | A name and a RoPE mode |
| 2 | Same family, plus a fact the file does not state | A `configure` callback over `DecoderConfig` |
| 3 | Its own graph shape — including a non-decoder family | A `ModelBuilder`, against the builder SDK |

### Tier 1 — a name and a RoPE mode

The generic decoder builder derives structure from the GGUF tensor table and metadata: QK-norm,
projection biases, fused QKV, MoE routing, shared experts, sliding-window attention, soft-caps,
per-layer KV head counts. For an architecture in that family, everything except the RoPE mode is
already known, and the RoPE mode is the one thing a GGUF file does not record.

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"

core.add_extension(std::make_shared<ov::frontend::gguf::ArchitectureExtension>(
    "my-arch", ov::frontend::gguf::RopeMode::Neox));
```

`RopeMode::Neox` rotates halves (qwen, phi3, gemma, ...); `RopeMode::Normal` rotates consecutive
pairs (llama, minicpm, ...). Mirror `llama_model_rope_type` for the architecture. Getting it wrong
produces a model that converts and generates nonsense, so check it against llama.cpp rather than
guessing.

### Tier 2 — plus a configuration hook

When one property genuinely cannot be detected — the classic case is GeGLU vs SwiGLU, which the
tensor table does not distinguish — adjust the auto-detected config:

```cpp
std::make_shared<ArchitectureExtension>(
    "my-arch", RopeMode::Neox,
    [](ov::frontend::gguf::DecoderConfig& cfg) {
        cfg.is_geglu = true;
    });
```

The callback runs after structural detection, so it sees a fully populated
[`DecoderConfig`](../dev_api/openvino/frontend/gguf/builder/decoder_config.hpp) and can override any
field. Prefer letting detection do the work and overriding only what it cannot know.

### Tier 3 — your own builder

For a graph shape the decoder builder does not have — a hybrid stack, an encoder, a projector —
supply a `ModelBuilder`. This is also the **only** route for a non-decoder family, and it is not
limited by anything the frontend already implements.

```cpp
std::make_shared<ArchitectureExtension>(
    "clip",
    [](const BuildContext& c) { return std::make_shared<MyVisionBuilder>(c); },
    [](const GgufMetadata& m) { return m.get_key_or("clip.has_vision_encoder", false); });
```

The third argument is a **match predicate**, for a file whose `general.architecture` does not
identify it. mmproj files are the reason it exists: they all call themselves `"clip"` and are told
apart by a metadata flag (llama.cpp `tools/mtmd/clip-impl.h`). Without a predicate, the extension
claims files that name its architecture.

An extension that claims a file is consulted **before** the built-in family detection, which is what
lets it handle a file the decoder builder would otherwise reject.

## Porting a llama.cpp model file

A llama.cpp model file (`src/models/<arch>.cpp`) has three parts, which map directly:

| llama.cpp | Here |
|---|---|
| `load_arch_hparams` | `GgufHparams`, or `metadata().get_key(...)` for anything unusual |
| `load_arch_tensors` | nothing — weights are looked up by name, not declared up front |
| `build_arch_graph` / the `graph` ctor | `ModelBuilder::build()`, against `GgufGraphContext` |

`load_arch_tensors` has no counterpart on purpose. It exists in llama.cpp because every architecture
enumerates its tensors by hand; here they are read straight from the file's tensor table, so a
tensor the file lacks simply yields an empty value.

### Construct-by-construct mapping

[`GgufGraphContext`](../dev_api/openvino/frontend/gguf/builder/graph_context.hpp) is the counterpart
of `llm_graph_context` and is named after it, so most lines port by substitution:

| llama.cpp | Here |
|---|---|
| `ggml_tensor *` | `GgufValue` |
| `NULL` tensor, `if (w)` | empty `GgufValue`, `if (w)` — same idiom |
| `model.layers[il].attn_norm` | `tensors.layer(il).attn_norm` |
| `model.tok_embd` | `tensors("token_embd.weight")` |
| `ml.get_key(LLM_KV_..., x)` | `ctx.metadata().get_key("<arch>.key", x)` |
| `hparams.n_embd_head_v()` | `ctx.hparams().n_embd_head_v()` |
| `n_tokens` | `ctx.n_tokens()` |
| `build_inp_embd(model.tok_embd)` | `ctx.build_inp_embd(...)` |
| `build_inp_pos()` | `ctx.build_inp_pos()` |
| `build_attn_inp_kv()` | `ctx.build_attn_inp_kv()` |
| `build_norm(cur, w, NULL, LLM_NORM_RMS, il)` | `ctx.build_norm(cur, w, eps)` |
| `build_ffn(...)` | `ctx.build_ffn(...)` |
| `build_attn(inp, wo, wo_b, Q, K, V, ..., scale, il)` | `ctx.build_attn(il, Q, K, V, wo, wo_b, scale)` |
| `build_lora_mm(w, cur)` | `ctx.build_lora_mm(w, cur)` |
| `ggml_add` / `ggml_mul` / `ggml_scale` | `ctx.add` / `ctx.mul` / `ctx.scale` |
| `ggml_rope_ext(...)` | `ctx.rope_ext(x, pos, freq_factors, cfg, rope_case)` |
| `ggml_reshape_3d(ctx, x, a, b, c)` | `ctx.reshape(x, {a, b, c})` |
| `ggml_soft_max` / `ggml_silu` / `ggml_gelu` | `ctx.soft_max` / `ctx.silu` / `ctx.gelu` |
| `cb(cur, "name", il)` | `ctx.cb(cur, "name", il)` — kept so these lines survive untouched |
| `res->t_logits = cur; ggml_build_forward_expand(gf, cur);` | `ctx.set_output(cur);` |

Two places where a port cannot be a copy:

- **`ggml_permute`.** `ctx.permute` takes axes in the shape's own `[ne3, ne2, ne1, ne0]` numbering,
  the reverse of ggml's `ne` order, so the axis list has to be translated.
- **Shapes.** `ctx.reshape` takes ggml `ne` order (fastest-varying first), matching
  `ggml_reshape_*`, but `GgufValue::shape()` is stored reversed. Use `GgufValue::ne(i)` to read a
  dimension the way `t->ne[i]` does.

Everything else — including output shapes, which the wrappers infer — needs no bookkeeping.

### Anything not covered

`ctx.raw_op(...)` appends a node in the GGML op vocabulary directly. If the operation is one the
frontend does not translate yet, register an `ov::frontend::ConversionExtension` for it alongside
the architecture extension; the two compose, so a genuinely new operation still does not require a
frontend change. See [how_to_add_op.md](how_to_add_op.md) for what a translator does.

### A worked example

`tests/test_architecture_extension.cpp` contains two complete builders, both written against
nothing but the SDK headers, and both covered by tests:

- **`Qwen3PortBuilder`** — a port of llama.cpp's `src/models/qwen3.cpp`. Read it side by side with
  the original; the structure, ordering and naming are deliberately preserved. The test asserts it
  builds a real KV-cached decoder whose attention fuses to one SDPA per layer.
- **`VisionEncoderBuilder`** — an mmproj vision encoder: patch embeddings, non-causal attention
  blocks, projector. It is a family the frontend has no code for at all, and it is registered by
  metadata predicate rather than by name.

## Packaging as a shared library

For a genuinely rebuild-free workflow, ship the extension in its own library:

```cpp
OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ArchitectureExtension>("my-arch", RopeMode::Neox),
});
```

```cpp
core.add_extension("libmy_arch.so");
```

Link it against `openvino::frontend::gguf`, which installs the SDK headers used above. Extensions
arriving this way are wrapped in an `ov::detail::SOExtension`; the frontend unwraps them, so
registration works identically to the in-process form.

Extensions are held **per `FrontEnd` instance**, like conversion and transformation extensions, so
registrations on one `Core`/`FrontEnd` do not leak into another.

## Verifying a new architecture

1. **It converts.** The frontend is not auto-selected, so ask for it by name:
   `fe = FrontEndManager().load_by_framework("gguf")`, then `fe.convert(fe.load("model.gguf"))`.
2. **The graph is sane.** Check the op histogram: attention should collapse to a single SDPA per
   layer and MoE routing to a grouped matmul, not a long chain of primitives. A layout mistake shows
   up here as a decomposed attention.
3. **The numbers are right.** Generate through OpenVINO GenAI and compare against
   `llama.cpp`'s `llama-cli` on the same prompt. Greedy tokens should match; small drift after
   dozens of tokens is expected from kernel differences. This is the step that catches a wrong RoPE
   mode, which nothing structural will.
4. **Nothing else regressed**, if you changed shared code: `tests/test_arch_conversion.cpp` pins a
   graph fingerprint for every supported architecture.

Declare an architecture `Maturity::Verified` only after step 3. Until then leave it
`Experimental` (the default), which converts but warns once, so a user knows it is best-effort.

## Extension or in-tree?

Ship an **extension** when the architecture is yours to maintain, when you need it in a released
OpenVINO you cannot rebuild, or when it is not ready to be supported for everyone.

Contribute **in-tree** when the architecture belongs to a family the frontend already supports and
would benefit every user — a Tier-1 architecture is a one-line change to
[`arch_registry.cpp`](../src/builder/arch_registry.cpp) plus a fixture, and then it is covered by
the frontend's own regression tests rather than yours.

The two use the same machinery, so moving an architecture from one to the other is mechanical.
