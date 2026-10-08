# Running converted GGUF models

`convert()` returns a stateless graph with llama.cpp-style inputs. A caller-registered `GenAIExtension`
turns a language model into a stateful model with the OpenVINO GenAI interface. Multimodal projector
files use their own [encoder contract](mmproj.md#graph-boundary).

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

The output is `logits`. [`run_gguf_model`](../../../../tests/model_hub_tests/gguf/utils.py) is a
working Python driver for two steps. This is the only route available from Python: the GGUF passes
below have no Python bindings, and `DecoderTransformationExtension` cannot be constructed from Python.

## Stateful and GenAI conversion

Register `GenAIExtension` before `convert()`. The frontend creates stateful caches before
stateless cache lowering, completes normalization (including recurrent-operation fusion), then
adapts the normalized graph to GenAI. Callers do not register or order individual passes.

```cpp
#include <openvino/frontend/gguf/extension/genai.hpp>
#include <openvino/frontend/manager.hpp>

using namespace ov::frontend::gguf;
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension(std::make_shared<GenAIExtension>());
auto model = frontend->convert(frontend->load("model.gguf"));
```

Internally, [`GGUFMakeStateful`](../include/openvino/frontend/gguf/make_stateful.hpp) replaces each KV cache
Parameter/Result pair with a Variable, adds `beam_idx` and reorders caches by it. Sliding-window
caches can be left stateless with `skip_caches`. Recurrent states are declared through the
`gguf_recurrent_states` rt_info key.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) requires a stateful model and
exposes GenAI's interface. Select the mode in the `GenAIExtension` constructor:

| Mode | Inputs | Output |
|---|---|---|
| `IDS_TO_LOGITS` (default) | `input_ids [B,T]`, `attention_mask [B,KV]`, `position_ids [B,T]`, `beam_idx [B]` | `logits [B,T,vocab]` |
| `EMBEDS_TO_LOGITS` | `inputs_embeds [B,T,D]` instead of `input_ids`; see [multimodal language models](mmproj.md#using-the-encoders-and-a-language-model) | `logits [B,T,vocab]` |

- The leading dimensions are derived from the live inputs, so one graph serves both SDPA and
  `ov::pass::SDPAToPagedAttention`.
- The LM head is connected directly to the logits Result with a rank-3 input, so GenAI's
  slice-before-matmul optimization projects only the sampled rows during prefill.
- Sliding-window masks use the `gguf_swa_window_size` rt_info value when the file records a window length.
- Gated-DeltaNet states (`qwen35`, `qwen35moe`) get a dynamic batch, follow `beam_idx` and mask
  left padding. Mamba 2 and `nemotron_h` states stay at one sequence: greedy, batch-1 decoding only.
  A consumer must reset, not trim, the states listed in `gguf_recurrent_states`.
- The pass does not create a tokenizer, sampler or pipeline, and it does not adapt encoder models.

`GGUFMakeStateful` remains independently available through `DecoderTransformationExtension`
for consumers that keep GGUF IO. The llama.cpp backend uses its own stateful lowering for its
cache-slot and fixed-mask semantics, without GenAI adaptation.

## Tokenizer metadata

Conversion attaches the file's `tokenizer.ggml.*` and `tokenizer.chat_template` values to the
model's rt_info as [`GGUFTokenizerMetadata`](../include/openvino/frontend/gguf/tokenizer_metadata.hpp),
under `gguf_tokenizer_metadata_key()`, when the decoder provides them. Values are strings, string vectors or tensors. A consumer such
as GenAI can build the tokenizer from it without reopening the file. The attribute is intentionally
in-memory only: it is dropped on `clone()` and not written to IR, so read it from the converted model.

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

See [`ssm_scan.cpp`](../src/op/ssm_scan.cpp), [`gated_delta_net.cpp`](../src/op/gated_delta_net.cpp)
and [`mul_mat_id.cpp`](../src/op/mul_mat_id.cpp) for exact selection conditions.

Internal operations are outside the public IR opsets. Saving IR can succeed and still produce a file
the IR frontend cannot read back, so test the save/read round trip. Compiled-model export, import
and caching are plugin-specific; test them on the target device. In-process conversion needs no IR.

A new internal-op path should keep a public-op reference path where practical, test both against
the same oracle and update the table above. Frontend placeholders need a normalization lowering.
