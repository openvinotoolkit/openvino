# OpenVINO GGUF Frontend — design & developer guide

The GGUF frontend converts GGUF model files into an `ov::Model`. It is a standard
`ov::frontend::FrontEnd` (registered as `"gguf"`, library `openvino_gguf_frontend`) and is the
path OpenVINO GenAI and the llama.cpp `ggml-openvino` backend take to run llama.cpp-style models
on OpenVINO devices.

> [!IMPORTANT]
> The frontend is deliberately kept out of model auto-detection (`is_hidden_frontend` in
> `src/frontends/common/src/manager.cpp`): it is not listed by `available_front_ends()` and not
> selected by `load_by_model`, so **`core.read_model("model.gguf")` does not work**. Reach it
> either by linking `openvino::frontend::gguf` and constructing `FrontEnd` directly (what GenAI
> and the llama.cpp backend do), or by name:
>
> ```python
> fe = FrontEndManager().load_by_framework("gguf")
> model = fe.convert(fe.load("model.gguf"))
> ```
>
> Nothing else is needed after `convert()`: the frontend runs its own normalization, and the only
> step `read_model` adds is `update_v10_model()`, which applies solely to legacy IR v10. Enabling
> `core.read_model` later is just a matter of dropping `"gguf"` from that list.

For "how do I add a new model architecture", see [adding_an_architecture.md](adding_an_architecture.md).
This document explains the *design* — the layering, the two decoder paths, why the frontend loads
GGUF itself instead of depending on llama.cpp, and the memory-consumption model.

## Layering

```
 .gguf file / live ggml_cgraph
          |
          v
   GgufDecoder  (abstract)  ── include/openvino/frontend/gguf/decoder.hpp
     |            \
     |             \__ two concrete implementations (see "Two decoder paths")
     v
  TranslateSession  ── src/translate_session.cpp
     |   walks the decoder's nodes, calls one op translator per node
     v
  op translators  ── src/op/*.cpp  (GGML_OP_MUL_MAT -> MatMul, GGML_OP_ROPE -> RoPE, ...)
     |   + normalization passes (SetRows lowering; caller extensions such as MakeStateful)
     v
   ov::Model  (stateless; stateful when the caller registers MakeStateful)
```

The **`GgufDecoder` interface** is the single seam. It is node-scoped: `visit_subgraph` hands the
translator a decoder bound to one node, and per-node accessors (`get_op_type`, `get_input_shape`,
`get_attribute`, `get_op_case`, ...) refer to that node; model-scope accessors (`get_model_inputs`,
`get_model_output_names`, and the optional `get_model_extra_inputs` / `get_tokenizer_config`) answer
whole-model questions. The optional ones have do-nothing defaults, so a decoder implements only what
it actually knows — which is what lets two very different decoders satisfy one interface (the native
builder answers all of them, the cgraph decoder none). Note there is no weight accessor: weights are
nodes, see "Weights: the GGML_OP_NONE convention".

Note what the interface deliberately does *not* have: anything describing the **execution mode**.
A decoder describes ggml *operations*, not a deployment, so there is no `is_stateful` / `is_static`.
Conversion always produces a stateless graph; see "Statefulness".

Everything downstream of the decoder (translators, passes, the produced `ov::Model`) is shared by
both decoder paths — so a new op or a graph fix benefits both at once.

## Two decoder paths (one frontend)

The frontend is fed by **two** `GgufDecoder` implementations, deliberately:

1. **`GgufBuilderDecoder`** (`src/builder/`) — the *native* path. Loading a `.gguf` path
   parses the container (`src/quant/gguf.cpp`), and a `ModelBuilder` for the detected model family
   (today `DecoderBuilder`, `src/builder/arch/decoder_builder.cpp`) builds a `GgufGraph` (a flat,
   topologically-ordered node list in the GGML op vocabulary), which `GgufBuilderDecoder` exposes.
   **No llama.cpp in the process.** This is the default and the path GenAI uses.

2. **`GgmlOvDecoder`** (lives in llama.cpp's `ggml/src/ggml-openvino` backend, not in this repo) —
   wraps a live `ggml_cgraph` that llama.cpp already built. Used when OpenVINO is a *backend inside
   llama.cpp*; it links `openvino::frontend::gguf` and calls `FrontEnd::convert()` directly.

Both produce the exact same op vocabulary, so they share the translators and passes verbatim.

### Why the native path does not use llama.cpp to load the model

The obvious alternative — have the native path call into llama.cpp to build the cgraph (reusing path
2's `GgmlOvDecoder`) — was considered and rejected for the default path. llama.cpp's graph builder
is not a standalone library: it is entangled with `llama_model` loading and the `libllama`/`libggml`
runtime, so using it means linking substantial llama.cpp + ggml into the OpenVINO/GenAI process
(whether via submodule, FetchContent, or a prebuilt lib — the mechanism is not the issue, the
dependency is). The drawbacks that made the native builder the default:

- **A second model-loading runtime + ~2x transient memory.** llama.cpp would allocate the full
  model into ggml tensors just to *build* the cgraph, and OpenVINO would then re-materialize the
  weights as `Constant`s — roughly double the host memory during load. The native builder mmaps the
  file and zero-copies weight bytes straight into OpenVINO `Constant`s (see "Memory model").
- **Version coupling.** llama.cpp has no stable ABI and GGUF/architecture/tensor-naming conventions
  change quickly; pinning a version means chasing upstream, not pinning means breakage.
- **Binary size, build matrix, supply-chain.** Shipping libllama+libggml (CPU kernels, quant
  kernels, tokenizer, sampling) inside OpenVINO enlarges the binary, adds a CMake/SIMD/backend
  matrix to build on every target, and adds a CVE/provenance surface.
- **Loss of self-containment.** Converting a `.gguf` works in any OpenVINO deployment with
  nothing extra; a llama.cpp dependency would break that.

The upside of llama.cpp — instant coverage of ~130 architectures — is real. The intended way to get
it *without* burdening the default path is to keep `GgmlOvDecoder` as an **optional, build-gated
alternative decoder** feeding the same frontend, never as the default. The native builder stays the
self-contained default.

## Weights: the GGML_OP_NONE convention

Weights are surfaced uniformly as `GGML_OP_NONE` leaf nodes (there is no `get_model_weights`
special path in the graph walk). `translate_weight` (`src/op/weight.cpp`) turns each leaf into a
compressed **decompression subgraph** — `Constant(low-bit) -> Convert -> Subtract(zp) ->
Multiply(scale) -> Reshape` — never a fully-materialized f32 constant. Two payload shapes are
accepted by the same translator:

- Native builder: the parser already extracted `weight`/`scales`/`zp` tensors; the leaf carries
  them as attributes and `translate_weight` calls `make_weight_node(base, weights, qtypes)`.
- cgraph path: the leaf carries the raw ggml bytes; `translate_weight` extracts them itself.

Both build the *identical* compressed OpenVINO subgraph, so **inference speed and compile memory
are independent of which path/payload was used** (verified: OLMoE compile peak 8673 MB via
GGML_OP_NONE vs 8672 MB via the older eager path).

## op_case: one numbering for both ingest paths

Several ggml ops cover structurally different uses that need different OpenVINO subgraphs — a
`GGML_OP_RESHAPE` splitting a projection into heads is not the reshape that merges them back. The
decoder disambiguates with an `op_case` attribute, read via `NodeContext::get_op_case()`.

**`op_case` describes the tensor operation, not which decoder produced it.** The cgraph decoder
derives it by inspecting the ggml node (`ggml-decoder.cpp::compute_op_case`); the native builder
sets it when it emits a node, choosing the case that matches what it is doing. Both therefore land
on the *same* case for the same operation, which is what keeps one translator body serving both
paths — and what makes the two graphs comparable node-for-node.

A case that exists only to mean "this came from the builder" is a defect: it splits a shared
translator into two bodies that then drift apart. Three cases are legitimately builder-only, each
because the operation itself genuinely differs rather than for numbering reasons:

| case | op | why |
| --- | --- | --- |
| 100 | `FLASH_ATTN_EXT` | The builder keeps q/k/v ggml-natural so the order is Concat -> GQA tile -> one Transpose -> SDPA, the shape a plugin's own attention-fusion pass expects; permuting first would block that fusion. A deliberately *better* graph, not an equivalent one. |
| 104 | `VIEW` | Takes a second (shape-reference) input the cgraph path does not supply, so it has a different arity than the shared cases. |
| 10 | `GET_ROWS` | llama.cpp reshapes `probs` before the MoE gating gather, so the cgraph decoder sees a different input shape that the generic path already handles. |

Anything else should reuse a shared case, which usually means the builder describing its node the
way the shared translator expects (e.g. supplying `view_slice` for a plain single-axis shrink rather
than a bespoke attribute) or emitting the same node *count* ggml does instead of fusing steps.

## Memory model

Understanding where memory goes matters for large models (MoE especially). Load/convert is
plugin-independent; compile and inference depend on how the target plugin handles the compressed
weight subgraph the frontend emits, so those numbers are plugin-specific — measured here on CPU,
which is the only plugin verified so far.

**Load / convert (plugin-independent):**
- The GGUF file is memory-mapped (`ov::load_mmap_object`); non-quantized tensors are zero-copy views
  into the mmap. Quantized tensors are *repacked* once into a single `AlignedBuffer` (u4/u8 weights
  + f16 scales + integer zero-points in OpenVINO's compressed layout) and wrapped as `Constant`s
  via a `SharedBuffer` — so there is no second full-model host allocation and weights are never
  expanded to f32 at load. Measured: OLMoE-1B-7B q4_0 (3.9 GB file) read peak ≈ 3.9 GB.
- Because the graph keeps weights *compressed*, conversion memory is roughly the file size, not
  the dequantized (f32) model size.

**Compile / `compile_model` (plugin-dependent — CPU measured):**
- Weights stay compressed through compilation only if the target plugin recognizes the compressed
  weight subgraph and keeps the decompression fused into its own kernel rather than folding it to
  a plain f32 constant. Measured on CPU: OLMoE compile peak ≈ 8.7 GB (vs a ~53 GB blow-up when
  that recognition doesn't happen for the MoE experts).
- A plugin's compressed-weight recognition can also be narrower for some specific quantization
  types than others, which can make a model's experts fall off the compressed path for that type
  alone even though other types on the same model stay compressed; see
  [`supported_models.md`](supported_models.md) "MoE expert weights and quantization choice" for a
  measured CPU example (Q2_K) and its impact.
- KV cache precision: the stateful KV cache is f16 (set in `translate_session`); leaving it at a
  narrower default type has caused NaNs in some decode paths on CPU.

**Inference:** on a plugin that keeps weights compressed through compile, they are decompressed on
the fly, with no persistent f32 weight copy.

**Rule of thumb:** peak host memory ≈ `max(file_size_for_read, compressed_graph + plugin_scratch)`,
NOT the dequantized model size — *provided* the target plugin recognizes the compressed weight
subgraph. If a future change makes a model's weights expand to f32 at compile on a given plugin,
that is the regression to look for (compile peak jumping toward the dequantized size).

## Statefulness & the attention backends

**The frontend is universal: conversion always produces a STATELESS graph.** Every KV cache is an
explicit model `Parameter`, written by a `SetRows` placeholder op and returned as a `Result`. That is
the same shape optimum-intel exports before applying its own
`apply_make_stateful_transformation` — and it is why the decoder interface carries no execution-mode
flag.

**Statefulness is a caller concern**, chosen by registering a transformation extension. Extensions
run in the frontend's normalization stage *ahead of* the built-in `LowerSetRowsStateless`, so a
registered pass consumes the KV-cache `SetRows` ops and the default lowering only ever sees the ones
left over (e.g. MoE routing writes, which stay stateless either way):

```cpp
// Stateless (the default): caches are inputs/outputs, SetRows -> ScatterUpdate.
auto stateless = fe.convert(fe.load("model.gguf"));  // plain FrontEnd, no extension

// Stateful: caches become Variables (ReadValue / Concat / Assign).
ov::frontend::gguf::FrontEnd fe;
fe.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
    ov::frontend::gguf::pass::MakeStateful()));
auto stateful = fe.convert(fe.load("model.gguf"));
```

`ov::Core::add_extension` works too when going through a `Core` (it forwards its
extensions to the frontend before `load`), but it is global; driving the frontend directly scopes the
choice to one conversion, which is what GenAI does.

Three consumers, three combinations, one frontend:

| Consumer | Extension registered | Result |
| --- | --- | --- |
| plain `FrontEnd::convert` | none | stateless graph, gguf-native IO |
| OpenVINO GenAI | `MakeStateful` + then `AdaptToGenAI` | stateful graph, GenAI IO |
| llama.cpp `ggml-openvino` | its own `LlamaCppToStateful` | stateful graph, its own cache layout & mask re-slicing |

`MakeStateful` (`include/openvino/frontend/gguf/make_stateful.hpp`) is decoder-agnostic: it infers
each cache's append axis from the cache `Parameter`'s single dynamic axis (or takes it explicitly for
a preallocated cache), and re-splits the placeholder's flattened rows against the cache layout. It
scopes itself to cache growth only and does **not** touch the attention mask — a dynamically-sized
mask (what the native builder emits) needs no change, while a preallocated fixed mask window must be
re-sliced by the caller, which is what llama.cpp's own extension does.

`AdaptToGenAI` (`src/pass/adapt_to_genai.cpp`, run by GenAI after conversion) then rewrites the
gguf-native IO (`inp_tokens`/`inp_pos`/`self_kq_mask`/...) into GenAI's contract
(`input_ids`/`attention_mask`/`position_ids`/`beam_idx` -> `logits`), so a GGUF model behaves like an
optimum-intel export. The two concerns are separate passes precisely because they are independent:
cache form vs IO contract.

Either way the stateful graph is shaped so that a plugin's own attention-fusion pass can fold it
into a single fused SDPA-with-cache kernel, where the plugin has one.

The result is valid under **both** attention backends: plain stateful SDPA inference, and
`ov::pass::SDPAToPagedAttention` (the transform GenAI's ContinuousBatching adapter applies for
`ATTENTION_BACKEND=PA`). There is no mode flag — one graph serves both.

That works because the two backends disagree only about *where the token count lives*. Plain
inference feeds `input_ids` as `[1, tokens]`; `SDPAToPagedAttention` rewrites the `Parameter` to
rank-1 `[tokens]` and splices an `Unsqueeze(axis=1)` in front of its consumers, so the body sees
`[tokens, 1]` and PA's hardcoded flattens read the count out of dim 0. Since ggml activations are
`[batch, tokens, heads, head_size]` with `batch == 1`, `[1, T, H, D]` and `[T, 1, H, D]` are
element-for-element the same buffer.

**The invariant to preserve: no node may pin the leading two dims to constants.** `AdaptToGenAI`
derives them from the live `input_ids`, and the op translators reshape with `special_zero=true` so a
`0` copies dim 0 through instead of writing a literal `1` (`reshape` cases 1/2, `rope`'s
bhsd/paired/to_bhls targets, `set_rows`, `MakeStateful`'s row re-split). Two traps when touching
these:

- **OV broadcasts elementwise operands from the right.** A rank-4 activation against a rank-3 one
  *appears* to work while dim 0 is a literal batch 1 (`[1,1,T,E]` right-aligns onto `[1,T,E]`), then
  silently forms a `T x T` outer product once tokens move to dim 0. **Activations are uniformly
  rank-4** (ggml's `[batch, tokens, heads, head_size]`) — there is no second convention to pick
  between, and a translator that emits rank 3 reintroduces exactly this bug.
- **`get_rows` lowers to `Gather(act, ids, axis=1, batch_dims=1)`** = `act[i, ids[i,j]]`, so an
  identity selection is `ids[i,j] == j` — an index along axis 1 replicated over axis 0, *not* a
  `0..tokens` range. A range only coincides with the identity when axis 0 is a batch of 1.

A `Convert` between the KV-cache `Concat` and SDPA also silently disables PA: `StateManagementPattern`
admits none, so `TranslateSession` runs `EliminateConvert` after `ConvertConvertLike` to drop the
no-op ones. If PA conversion regresses to 0, check for a reintroduced `Convert` there first.

Verifying PA is actually in use requires looking at the **compiled runtime graph**, not the
`ov::Model` — see "Measuring performance correctly" in
[`supported_models.md`](supported_models.md).

## Tokenizer metadata

A GGUF file embeds not just the weights but the full tokenizer (vocab, merges, scores, token
types, special-token ids, pre-tokenizer regex, chat template) under its `tokenizer.*` metadata
keys. The frontend carries that metadata out on the converted model so a consumer can build a
matching OpenVINO tokenizer/detokenizer **without re-opening the `.gguf` and without any GGUF
parser of its own** — the model object is self-describing.

Mechanism (native path):
1. The builder scrapes every `tokenizer.*` key into an `ov::AnyMap` keyed by the sub-key after the
   last dot (`model`, `tokens`, `merges`, `scores`, `token_type`, `pre`, `bos_token_id`,
   `chat_template`, ...), each value being a `std::string` / `std::vector<std::string>` /
   `ov::Tensor` — `extract_tokenizer_config` in `gguf_builder.cpp`, surfaced through the decoder's
   `get_tokenizer_config()`.
2. `TranslateSession` attaches it to the converted model's **runtime info** as a
   `GGUFTokenizerMetadata` attribute under `gguf_tokenizer_metadata_key()`
   (`include/openvino/frontend/gguf/tokenizer_metadata.hpp`).
3. The attribute is deliberately **non-serializable** (`is_copyable() == false`, empty
   `to_string()`): the vocab+merges are large and only meaningful in-memory between conversion and
   tokenizer construction, so it is dropped on clone and emitted as an empty placeholder if the IR
   is serialized (it never bloats the XML). It is an in-process handoff, not part of the saved model.
4. A consumer (OpenVINO GenAI) reads `model->get_rt_info()[gguf_tokenizer_metadata_key()]` and
   builds the OpenVINO tokenizer/detokenizer from it — see GenAI's `create_tokenizer_from_model`
   (`gguf_utils/gguf_tokenizer.cpp`), which turns the map into BPE/Unigram tokenizer models via
   `openvino_tokenizers`. So converting the `.gguf` + this rt_info is enough for GenAI to produce
   both the inference model and its tokenizer.

### With and without a llama.cpp dependency

The tokenizer path is exactly where the frontend's llama.cpp-independence pays off, and it behaves
correctly on both decoder paths:

- **Without llama.cpp (native `.gguf` path, the default).** The frontend parses the `tokenizer.*`
  keys itself and emits the `GGUFTokenizerMetadata` rt_info. The whole tokenizer round-trip
  (`.gguf` file -> OpenVINO tokenizer) happens with **no llama.cpp and no separate GGUF/tokenizer
  library in the consumer** — the OpenVINO model is the single source of truth. This is what lets
  the frontend + GenAI stand alone.

- **With llama.cpp (the cgraph / `GgmlOvDecoder` path).** When OpenVINO runs as a backend *inside*
  llama.cpp, llama.cpp already owns the tokenizer natively (it parsed the same `tokenizer.*` keys
  to build its own `llama_vocab`). There is nothing for the frontend to hand off, so `GgmlOvDecoder`
  leaves `get_tokenizer_config()` empty and **no rt_info is attached** — tokenization is done by
  llama.cpp, the frontend only produces the compute graph. `TranslateSession` attaches the
  metadata only when `get_tokenizer_config()` is non-empty, so the same code serves both paths
  with no branching in the consumer.

In other words: the tokenizer metadata is populated by whichever side *owns* GGUF parsing. On the
native path that is the frontend (so it exports the metadata for a llama.cpp-free consumer); on the
cgraph path that is llama.cpp (so the frontend stays out of the tokenizer's way). Either way the
`GgufDecoder::get_tokenizer_config()` seam is the single contract, and no consumer needs to link
llama.cpp to obtain a tokenizer.

### How the tokenizer is constructed (consumer side)

The frontend only *exports* the metadata; turning it into a runnable tokenizer is the consumer's
job, described here only because it defines the contract the frontend must satisfy. The reference
consumer is OpenVINO GenAI (`src/cpp/src/gguf_utils/gguf_tokenizer.cpp`): it builds a
tokenizer/detokenizer pair of **`ov::Model`s** out of `openvino_tokenizers` ops (dispatched on the
GGUF `tokenizer.ggml.model` key — SentencePiece for `llama`/`plamo2`, byte-level BPE for
`gpt2`/`gemma4`) so tokenization runs on an OpenVINO device like any other model, with no bespoke
tokenizer engine and no llama.cpp.

The GGUF metadata keys the frontend must preserve verbatim, because that consumer relies on them:
`model`, `tokens`, `merges`, `scores`, `token_type`, `pre` (pre-tokenizer id), the special-token
ids (`bos_token_id`/`eos_token_id`/`unknown_token_id`/`padding_token_id`), `add_bos_token` /
`add_space_prefix` flags, and `chat_template`.

**Serialization caveat.** Because `GGUFTokenizerMetadata` is non-serializable, it exists only on the
in-memory model straight out of the frontend. A model that was serialized to IR and reloaded has no
such rt_info; GenAI then falls back to re-reading the `.gguf`
(`create_tokenizer_from_config` → `tokenizer_config_from_meta`), which needs the file but still no
llama.cpp. The rt_info path is the fast in-process handoff; the file path is the durable fallback.

## Source map

| Path | Contents |
|---|---|
| `include/openvino/frontend/gguf/` | public headers: `decoder.hpp`, `frontend.hpp`, `make_stateful.hpp`, `adapt_to_genai.hpp`, `tokenizer_metadata.hpp`, `set_rows_op.hpp` |
| `src/frontend.cpp` | FrontEnd: `.gguf` magic sniff + native load path; live-decoder path; extensions |
| `src/translate_session.cpp` | graph walk, weight seeding, normalization passes (caller extensions then built-ins), tokenizer rt_info |
| `src/op/*.cpp` | one op translator per GGML op |
| `src/builder/` | native `.gguf` graph builder + `GgufBuilderDecoder`; layered as `graph_emitter` (arch-agnostic node emission), `blocks/` (reusable fragments), `decoder_config` (architecture detection), `arch/decoder_builder` (topology), `arch_registry` / `model_kind` (what is accepted, and which family) |
| `src/quant/` | GGUF container parser (`gguf.cpp`), dequant fill fns (`gguf_quants.cpp`), weight-node construction (`weights.cpp`) |
| `src/pass/` | `LowerSetRowsStateless` (built-in), `MakeStateful` + `AdaptToGenAI` (caller-registered) |
| `src/helper_ops/` | internal `SetRows` placeholder op |
| `tests/` | C++ op/dequant tests (in CI); standalone python dev/bench scripts |

## Testing

- C++ unit tests (`tests/*.cpp`, target `ov_gguf_frontend_tests`) cover op translators and weight
  dequant against real-ggml reference `.npy` fixtures, and run in CI.
- A graph-fingerprint check (sha256 over sorted `(op_type, output_shape)` pairs of the converted
  model, per architecture) is the recommended cheap regression gate for any builder change — it
  proves the produced graph is unchanged. Accuracy is validated opt-in against llama.cpp (WWB-style)
  and by comparing greedy tokens on the same prompt.

See [testing_architecture.md](testing_architecture.md) for the full tier breakdown and what each
one can/cannot prove.
</content>
