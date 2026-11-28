# Running converted GGUF models

`convert()` returns a stateless graph with llama.cpp-style inputs. `adapt_to_genai()` turns a
converted model into the stateful models with the OpenVINO GenAI interface. Multimodal projector
files use their own [encoder contract](mmproj.md#graph-boundary).

## Split GGUF files

The native frontend loads split models directly without writing a merged GGUF. Pass any shard
with the standard `model-00001-of-00003.gguf` filename; the loader reads `split.count` and
`split.no` and opens the other expected shards in the same directory. All shards of the
chosen quantization must be available locally.

```python
from openvino.frontend import FrontEndManager

frontend = FrontEndManager().load_by_framework("gguf")
model = frontend.convert(frontend.load(
    "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
))
```

For renamed shards or files in different directories, supply a complete list of string paths.
List order does not matter; the loader uses `split.no` to select shard zero's model metadata.

```python
model = frontend.convert(frontend.load([
    "/models/part-a.gguf", "/other-disk/part-b.gguf", "/models/part-c.gguf"
]))
```

C++ accepts a `std::vector<std::string>` or `std::vector<std::filesystem::path>`, and also
separate path arguments to `load()`:

```cpp
auto input = frontend->load(std::vector<std::string>{
    "/models/part-a.gguf", "/other-disk/part-b.gguf", "/models/part-c.gguf"});
auto model = frontend->convert(input);
```

An explicit list must contain every shard; the loader does not fill in omitted entries.
Inputs must belong to one split model. Missing shards,
duplicate shard indices or tensor names, conflicting metadata and inconsistent total tensor
counts produce load errors. Each shard keeps its own mapping; quantized weights use the same
repacking as single-file inputs. This avoids a merged disk copy but does not reduce conversion
or compilation memory requirements or extend architecture and quantization coverage.

## Stateless contract

The default decoder graph mirrors the llama.cpp backend interface:

| Input | Meaning |
|---|---|
| `inp_tokens [1,1,1,T]` i32 | Token IDs of this step |
| `inp_pos [1,1,1,T]` i32 | Positions; four sections per token for interleaved M-RoPE (`qwen35`, `qwen35moe`) |
| `inp_out_ids [1,1,1,N]` i32 | Rows whose logits are produced |
| `self_kq_mask [1,1,T,KV]` f32, `self_kq_mask_swa` | Additive causal mask; the second one for sliding-window layers |
| `inp_kv_idx`, `token_len_per_seq` | KV write slots and per-sequence length, when the graph consumes them |
| `cache_k_*`, `cache_v_*`, recurrent states | Previous cache/state, returned as matching outputs |

`qwen4exp` declares all of its caches as recurrent states: head-major `qsa_cache_k_l*` /
`qsa_cache_v_l*` `[1,kv_heads,keys,head]` grown by concatenation, the QSAIndexer states
`qsa_block_keys_l*` (one pooled key per complete block), `qsa_pending_keys_l*` and
`qsa_pending_rope_l*`, the n-gram convolution `ple_conv_state_l*` and `ngram_history`, the previous
`ngram_size - 1` tokens stored relative to the n-gram EOS token so that zero starts a sequence. A
state with a dynamic axis starts empty. Valid key counts come from the last row of `self_kq_mask`.

The output is `logits`. [`run_gguf_model`](../../../../tests/model_hub_tests/gguf/utils.py) is a
working Python driver for two steps. This is the only route available from Python: the GGUF passes
below have no Python bindings, and `DecoderTransformationExtension` cannot be constructed from Python.

## Stateful and GenAI conversion

Convert without GenAI-specific extensions, then pass the result to
[`adapt_to_genai()`](../include/openvino/frontend/gguf/adapt_to_genai.hpp). It returns the named
models GenAI uses; callers do not register or order individual passes. The same call adapts a
[projector file](mmproj.md#using-the-encoders-and-a-language-model).

```cpp
#include <openvino/frontend/gguf/adapt_to_genai.hpp>
#include <openvino/frontend/manager.hpp>

using namespace ov::frontend::gguf;
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
auto models = adapt_to_genai(frontend->convert(frontend->load("model.gguf")));
auto language = models.at("language");
```

`adapt_to_genai()` modifies and returns the converted decoder as `language`, and throws for a
model that is already adapted. `GenAIExtension` is deprecated: it calls `adapt_to_genai()` at
the end of `convert()` and returns `language`.

Internally, [`GGUFMakeStateful`](../include/openvino/frontend/gguf/make_stateful.hpp) replaces each KV cache
Parameter/Result pair with a Variable, adds `beam_idx` and reorders caches by it. It accepts both
the frontend's cache-write operation and its lowered stateless form, so it runs on a converted model. Sliding-window
caches can be left stateless with `skip_caches`. Recurrent states are declared through the
`gguf_recurrent_states` rt_info key.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) then exposes GenAI's
interface. Select the mode with `GenAIAdaptOptions::input`:

| Mode | Inputs | Output |
|---|---|---|
| `IDS_TO_LOGITS` (default) | `input_ids [B,T]`, `attention_mask [B,KV]`, `position_ids [B,T]`, `beam_idx [B]` | `logits [B,N,vocab]` |
| `EMBEDS_TO_LOGITS` | `inputs_embeds [B,T,D]` instead of `input_ids`; see [multimodal language models](mmproj.md#using-the-encoders-and-a-language-model) | `logits [B,N,vocab]` |
| `EMBEDS_TO_LOGITS_TEXT` | Text embeddings `[B,T,D]`, with text `position_ids [B,T]` even for M-RoPE models | `logits [B,N,vocab]` |

Both embedding modes also return the token lookup as `text_embeddings` and, for architectures with
per-layer token embeddings, `text_embeddings_per_layer`.

- The number of output rows `N` depends on the model and attention backend. Standard SDPA decoders
  generally select the last token per sequence; PagedAttention can expose a row for each token.
- The leading dimensions are derived from the live inputs, so one graph serves both SDPA and
  `ov::pass::SDPAToPagedAttention`.
- The LM head is connected directly to the logits Result with a rank-3 input, so GenAI's
  slice-before-matmul optimization projects only the sampled rows during prefill.
  A scalar output scale is moved to the activation input when the head has no other consumers;
  softcap output chains remain recognizable without expanding compressed weights.
  Models with other post-head processing, such as Inkling's padded-vocabulary trimming, project
  every prompt row instead.
- Sliding-window masks use the `gguf_swa_window_size` rt_info value when the file records a window length.
  Noncausal sliding-window attention is not supported by GenAI adaptation.
- Gated-DeltaNet states (`qwen35`, `qwen35moe`) and LFM2 short-convolution states
  (`lfm2`, `lfm2moe`) get a dynamic batch, follow `beam_idx` and mask
  left padding. `qwen4exp` states follow `beam_idx` as well, and left-padded positions read as the
  n-gram EOS token in the n-gram hash. Mamba 2 and `nemotron_h` states stay at one sequence: greedy, batch-1 decoding only.
  A consumer must reset, not trim, the states listed in `gguf_recurrent_states`.
- In `EMBEDS_TO_LOGITS` mode, hashed n-gram embeddings (`qwen4exp`) move to a separate lookup
  model, `GenAIExtension::get_ngram_embedding_model()`, with the optimum-intel IO: `input_ids [B,T]`
  and `ngram_history [B,n-1]` (absolute ids, EOS for a new sequence) produce
  `ngram_embeddings [B,T,1,width]` and `new_ngram_history`. The language model takes
  `ngram_embeddings`; the caller keeps the history, resets it per sequence and reorders it for beams.
- The pass does not create a tokenizer, sampler or pipeline, and it does not adapt encoder models.

### Multi-token prediction (MTP)

Some checkpoints store NextN draft blocks after the target layers
(`<arch>.nextn_predict_layers`). Ordinary conversion excludes these blocks and returns
logits only. Request MTP at load time to convert the target and its draft in one call:

```cpp
auto combined = frontend->convert(frontend->load("model-with-mtp.gguf", LoadOptions{true}));
auto models = adapt_to_genai(combined, {pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS_TEXT});
auto target = models.at("language");
auto text_embeddings = models.at("text_embeddings");
auto draft = models.find("mtp_draft");  // absent when the file has no NextN block
```

The conversion returns one model; `adapt_to_genai()` splits and adapts it. Registered
`DecoderTransformationExtension` passes run separately on the target and draft before the default
stateless cache lowering. Their sinks, variables, and model metadata are preserved when the
components are combined and split. Without such an extension the combined model is stateless.
Draft input/output and variable names use an `mtp.` prefix in the combined model, repeated if
needed to avoid collisions; their original names are restored when the components are split.
Both models return `last_hidden_state [B,T,D]` for every token, after final
normalization and before any output-head Hadamard transform. The draft also takes
`hidden_states [B,T,D]`; pair the target hidden state at token `t` with the embedding of token
`t+1`. In the embeddings modes the draft takes the same text embeddings as the target, so
`text_embeddings` serves both. A file without NextN blocks still converts as a target that
exports hidden states, for a draft supplied from another file.
MTP conversion requires a native GGUF file and cannot be combined with `GenAIExtension`;
use `adapt_to_genai()` for GenAI-compatible models. Registered transformations must comply
with the chosen adaptation's input/output contract. On CPU, distinct output names for multiple
Results sharing one tensor can be lost during compilation.

The draft owns the hidden-state normalization, EH projection, NextN block and output head.
Its head is the NextN head when present, otherwise the target `output.weight` or, for a tied
checkpoint, `token_embd.weight`. Target and draft caches are independent. The draft reuses
the target's weight nodes, so tensors read by both models, such as the token embedding or a
shared output head, are unpacked once and share host buffers. Per-layer KV-head, recurrent
and sliding-window metadata may describe the target layers or all stored layers; when the
draft KV-head count is omitted, it is inferred from the NextN key projection.

The frontend supplies graphs, not a sampler. A consumer must connect the target and draft
models to its MTP strategy (llama.cpp calls its equivalent option `--spec-type draft-mtp`).

`DecoderBuilder` builds drafts only for the families accepted by `has_mtp_draft()` in
`decoder_builder.cpp`, currently `qwen35`; `qwen4exp` builds its own (see below), and other
families reject the MTP load option. To add
a family, check that its NextN block builds as a regular layer of that family and compare the
draft with the ggml MTP graph. A separate `nextn.embed_tokens.weight` table (GLM-4.5) and
chained multi-block drafts (`step35`) are not supported yet.

### Qwen3.8-Flash-Next MTP

`qwen4exp` builds its draft in its own builder through the `GgufGraphContext` MTP API
(`share_weights_from`, `set_mtp_draft`, `mark_mtp_draft_only`), because its residual is four
hyper-connection streams rather than one hidden vector. `last_hidden_state` and the draft input
`hidden_states` are the hyper-connection residual before the output mixer, `[B,T,hc*n]`, not a
normalized hidden state. The draft runs one block-sparse NextN block with its own caches and
mixes its streams through `nextn.hc_head_*` before the output projection. With
`LoadOptions{true}`:

- a model with an embedded NextN block converts to the target and its draft, as above;
- Unsloth's target files have no NextN metadata and convert to a target that exports
  `last_hidden_state`;
- a self-contained head such as Unsloth's `mtp-*.gguf`, which stores only the NextN block,
  `token_embd` and `output`, converts to the draft alone: `adapt_to_genai()` returns only
  `mtp_draft`. Loading such a file without `LoadOptions{true}` fails.

Heads with `nextn_shared_target_tensors` borrow the target's embeddings and are rejected.
In `EMBEDS_TO_LOGITS_TEXT` mode the target keeps its n-gram lookup, so `input_ids` stays an
input next to `inputs_embeds`. GenAI's MTP pipeline needs paged attention, which `qwen4exp`
does not support yet, so the frontend currently supplies only the graphs.

`GGUFMakeStateful` remains independently available through `DecoderTransformationExtension`
for consumers that keep GGUF IO. The llama.cpp backend uses its own stateful lowering for its
cache-slot and fixed-mask semantics, without GenAI adaptation.

## Tokenizer metadata

Conversion attaches the file's `tokenizer.ggml.*` and `tokenizer.chat_template` values to the
model's rt_info as [`GGUFTokenizerMetadata`](../include/openvino/frontend/gguf/tokenizer_metadata.hpp),
under `gguf_tokenizer_metadata_key()`, when the decoder provides them. Values are strings, string vectors or tensors. A consumer such
as GenAI can build the tokenizer from it without reopening the file. The attribute is intentionally
in-memory only: it is dropped on `clone()` and not written to IR, so read it from the converted model.

## Recommended sampling

When the file has `general.sampling.*` keys, conversion stores them in the model's rt_info under
`generation_config`, using Hugging Face `generation_config.json` names: `top_k`, `top_p`, `min_p`,
`temperature` (from `temp`) and `repetition_penalty` (from `penalty_repeat`). `do_sample` is `false`
only when `temp` is zero or negative, which matches llama.cpp. Values are strings and survive IR
serialization. GenAI uses them as the default generation config.

## Internal operations and serialization

[`SetRows`](../include/openvino/frontend/gguf/set_rows_op.hpp) is a conversion placeholder for
`GGML_OP_SET_ROWS`. `GGUFMakeStateful` consumes cache writes; `LowerSetRowsStateless` lowers the rest.
It never survives into the converted model.

These core `ov::op::internal` operations can remain in the converted model:

| Operation | Emitted for | Alternative |
|---|---|---|
| `SelectiveSSM` | `SSM_SCAN`, Mamba 2 scalar decay | None |
| `GatedDeltaNet` | `GATED_DELTA_NET`, scalar gate | `Loop` reference path for per-key gating or `force_ref`; not available with `split_outputs` |
| `GatherMatmul` | `MUL_MAT_ID`, eligible constant expert weights | Generic Gather/MatMul for other weights |
| `QSAIndexer` | Qwen Sparse Attention block selection (`qwen4exp`) | None |
| `NGramHashFunction` | Hashed n-gram embedding rows (`qwen4exp`) | None |

See [`ssm_scan.cpp`](../src/op/ssm_scan.cpp), [`gated_delta_net.cpp`](../src/op/gated_delta_net.cpp)
and [`mul_mat_id.cpp`](../src/op/mul_mat_id.cpp) for exact selection conditions.

Block-sparse attention emits `QSAIndexer` directly, with the index query RMSNorm and RoPE inside
the operation, and the public opset17 `SparseAttention`
([`sparse_attention.cpp`](../src/op/sparse_attention.cpp)), so no plugin fusion is needed.
`QSAIndexer` takes the block-key cache as is and returns it grown to the complete blocks of the
longest row, the contract of the CPU implementation.

Internal operations are outside the public IR opsets. Saving IR can succeed and still produce a file
the IR frontend cannot read back, so test the save/read round trip. Compiled-model export, import
and caching are plugin-specific; test them on the target device. In-process conversion needs no IR.

A new internal-op path should keep a public-op reference path where practical, test both against
the same oracle and update the table above. Frontend placeholders need a normalization lowering.
