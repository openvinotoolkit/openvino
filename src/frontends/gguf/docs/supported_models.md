# GGUF Frontend — Supported Models

This document lists the model architectures the GGUF frontend can convert and run
end-to-end. An architecture is listed as **Supported** only when at least one *real*
(non-synthetic) model of that architecture has been verified to load, convert, and
produce coherent output through the frontend.

Verification is done by running a real `.gguf` through the OpenVINO backend swap in
`llama.cpp` (`llama-completion`, CPU device, stateful execution) and confirming the
generated text is coherent and matches the pure-ggml CPU reference for the same prompt.

## Supported architectures

Each row was verified with the named real model.

| Architecture | Verified model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |
| `qwen35`  | Qwen3.5-4B (Q4_K_M)               | Hybrid GatedDeltaNet + full-attention layers; partial-rotary IMROPE; interleaved Q/gate joint projection; f16 KV cache. |

`llama`, `qwen2`, `qwen3`, `olmoe`, `gemma4`, and `qwen35` were verified with a fresh
end-to-end run of the model named above. `qwen3moe` and `gemma3` were verified in earlier
development on the models named above.

Quantization formats verified in the above runs: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

## How verification was performed

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run counts as verification only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage (e.g. `hunyuan`) is **not** counted as supported.

## Architectures accepted by the native `.gguf` builder

Everything above is about the **llama.cpp cgraph** path. This section covers the *other*
decoder — the native `.gguf` builder (`DecoderBuilder` in
[`src/builder/arch/decoder_builder.cpp`](../src/builder/arch/decoder_builder.cpp)), which is what
OpenVINO GenAI uses. The two paths share all op
translators but have separate architecture lists.

The builder's accept-list is the union of two sets, both defined in
[`src/builder/arch_registry.cpp`](../src/builder/arch_registry.cpp):

- **`verified_archs()`** — convert + compile + generation checked against a reference on a
  real checkpoint.
- **`experimental_archs()`** — expected to work via the builder's GGUF-tensor-table
  auto-detection, but not end-to-end verified. These convert and emit a one-time
  `OPENVINO_WARN` so callers know they are best-effort.

Anything not in either set is rejected with an explicit `OPENVINO_ASSERT` at load time
rather than converting into a silently wrong graph.

### `verified_archs()` — 10 architectures

| Architecture | Notes |
|---|---|
| `llama` | llama-2 / llama-3 |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm |
| `phi3` | fused QKV |
| `minicpm` | NORMAL rope + scalar embedding/residual/logit scales |
| `olmoe` | OLMoE 1B-7B (MoE) |
| `qwen35` | Qwen3.5/3.6 (and the Ternary-Bonsai backbone): hybrid Gated-DeltaNet + full attention, M-RoPE, interleaved query+gate projection. Greedy / batch 1 only |
| `gpt-oss` | MoE + attention sinks + SWA + OAI gated activation |
| `gemma3` | post-norms + final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings, shared KV |

### `experimental_archs()` — 20 architectures

| Architecture | Notes |
|---|---|
| `llama-embed` | Bidirectional LLaMA (embedding model, no causal mask) |
| `exaone4` | EXAONE 4.0: NEOX rope, post-norms (attn + ffn) |
| `plamo3` | PLaMo-3: NEOX rope, post-norms (attn + ffn) |
| `smollm3` | SmolLM3: NORMAL rope + SWA |
| `hunyuan-moe` | NEOX rope, MoE routing, QK-norm |
| `glm4moe` | GLM 4.5 MoE: 1 dense lead layer, MoE + attn post-norm |
| `exaone-moe` | EXAONE MoE: SWA + MoE, shared expert |
| `minimax-m2` | Minimax M2: pure MoE |
| `ernie4_5-moe` | Ernie 4.5 MoE: NORMAL rope, dense lead layers + MoE stride |
| `bailingmoe2` | BailingMoe V2: MoE + shared expert + QK-norm |
| `maincoder` | Maincoder-1B: NORMAL rope, QK-norm (auto-detected) |
| `mistral3` | Ministral-3B: NORMAL rope, dense |
| `muse-glimmer` | Muse Glimmer (Meta Onyx): NORMAL rope on SWA layers only (global layers are NoPE), sigmoid attention output gate, QK-norm, pre+post norms, final logit soft-cap |
| `mellum` | JetBrains Mellum: pure MoE |
| `deepseek2-ocr` | DeepSeekOCR: dense lead layers + MoE |
| `jais2` | JAIS-2: dense (biases auto-detected) |
| `hunyuan-dense` | Demoted from `verified_archs()`: degenerate output through the builder (see below) |
| `qwen3moe` | Qwen3 MoE; same topology as `olmoe`. Demoted: degenerate output through the builder |
| `gemma` | Gemma 2B / 7B. Demoted: throws through the builder (see below) |
| `gemma2` | post-norms + attention soft-cap. Demoted: degenerate output through the builder |

RoPE flavor is **not** in these tables because it is a separate switch: archs listed in
`arch_uses_neox_rope()` use NEOX (rotate-halves), everything else uses NORMAL (rotate
consecutive pairs). Adding an arch to the accept-list without also classifying its RoPE is
the most common way to get a model that loads and produces garbage.

### Measured status through OpenVINO GenAI

Every architecture above was run through GenAI on CPU (`gguf_arch_check`, greedy, SDPA
backend) on the checkpoint named below. "Generates" means the model answered *"The capital
of France is"* correctly and coherently; **`llama.cpp` ref** is the same `.gguf` through
`llama-cli` on the default ggml CPU backend, which distinguishes a frontend bug from a
model/checkpoint that is simply weak on the prompt.

| Arch | Set | Model used | GenAI | llama.cpp ref |
|---|---|---|---|---|
| `llama` | verified | Llama-3.2-1B-Instruct Q4_K_M | generates | generates |
| `qwen2` | verified | Qwen2.5-0.5B-Instruct Q4_K_M | generates | generates |
| `qwen3` | verified | Qwen3-0.6B Q8_0 | generates (reasoning preamble) | same |
| `phi3` | verified | Phi-3-mini-4k-instruct Q4 | generates | generates |
| `minicpm` | verified | MiniCPM-2B-dpo Q4_K_M | generates | generates |
| `hunyuan-dense` | experimental | Hunyuan-0.5B-Instruct Q4_K_M | **degenerate** | generates |
| `olmoe` | verified | OLMoE-1B-7B-Instruct Q4_K_M | generates | generates |
| `qwen3moe` | experimental | Qwen3-0.9B-A0.6B Q4_K_M | **degenerate** | generates |
| `gpt-oss` | verified | gpt-oss-20b MXFP4 | generates (harmony format) | same |
| `gemma` | experimental | gemma-2b Q4_K_M | **throws** (SDPA shape mismatch) | degenerate too |
| `gemma2` | experimental | gemma-2-2b-it Q4_K_M | **degenerate** | generates |
| `gemma3` | verified | gemma-3-1b-it Q4_K_M | generates | generates |
| `gemma4` | verified | gemma-4-E4B-it Q4_K_M | generates | generates |
| `llama-embed` | experimental | llama-nemotron-embed-1b-v2 Q4_K_M | repeats (embedding model) | degenerate too |
| `exaone4` | experimental | EXAONE-4.0-1.2B Q4_K_M | **degenerate** | generates |
| `plamo3` | experimental | plamo-3-nict-2b-base Q4_K_M | **degenerate** | degenerate too |
| `smollm3` | experimental | SmolLM3-3B Q4_K_M | generates (reasoning preamble) | same |
| `maincoder` | experimental | Maincoder-1B Q4_K_M | generates | generates |
| `mistral3` | experimental | Ministral-3-3B-Instruct-2512 Q4_K_M | generates | generates |
| `muse-glimmer` | experimental | Muse-Glimmer-30B Q4_0 | generates | generates |
| `qwen35` | verified | Qwen3.5-0.8B Q8_0 | generates | generates |
| `qwen35` (Bonsai) | verified | Ternary-Bonsai-27B Q2_g64 | generates | generates |
| `deepseek2-ocr` | experimental | deepseek-ocr-2 Q4_K_M | generates | generates |
| `ernie4_5-moe` | experimental | ERNIE-4.5-21B-A3B Q4_K_M | **degenerate** (blank) | generates |
| `bailingmoe2` | experimental | Ling-mini-2.0 Q2_K | generates | generates |
| `mellum` | experimental | Mellum2-12B-A2.5B-Instruct Q4_K_M | generates | generates |
| `hunyuan-moe` | experimental | — | not tested (no checkpoint) | — |
| `glm4moe` | experimental | — | not tested (smallest GLM-4.5-Air ≈ 40 GiB) | — |
| `exaone-moe` | experimental | — | not tested (smallest ≈ 9 GiB, 32B) | — |
| `minimax-m2` | experimental | — | not tested (smallest ≈ 78 GiB) | — |
| `jais2` | experimental | — | not tested (no checkpoint) | — |

Two caveats on reading this table. `llama-embed` is an *embedding* model, so degenerate
greedy completion is expected of it, not a defect. `gemma` (v1 base) and `plamo3` (base, not
instruct) are degenerate on the reference too, so those rows are checkpoint/prompt artifacts
rather than frontend bugs.

That leaves **5 architectures that generate correctly under llama.cpp but not through the
builder** — `hunyuan-dense`, `qwen3moe`, `gemma2`, `exaone4` and `ernie4_5-moe` (blank output) —
i.e. real conversion defects, plus `gemma`, which throws instead of converting cleanly.
`hunyuan-dense`, `qwen3moe`, `gemma2` and `gemma` were previously misclassified as
`verified_archs()`; they have been moved to `experimental_archs()` (and now emit the one-time
`OPENVINO_WARN`) until the underlying defects are fixed and re-verified.

**`qwen35` is greedy / batch-1 only.** The recurrent conv and delta states are a single
static-shaped block with no batch axis, and `MakeStateful` does not reorder them by `beam_idx` the
way it reorders a KV cache. Beam search or batch > 1 therefore **fails at inference** with a shape
mismatch on the conv window's `Concat` — it does not silently mix state across beams, so no wrong
output can be produced. Prefix caching and PagedAttention are unavailable for the same reason: a
recurrent state cannot be re-derived from a cached prefix, and cannot be paged. Verified
token-for-token against llama.cpp on two real checkpoints (Qwen3.5-0.8B Q8_0, Ternary-Bonsai-27B
Q2_g64), with final-logits agreement within 1.0% / 0.12% of llama.cpp — in line with the noise
already present on the *verified* `qwen3` arch through the same harness.

A packaging gotcha worth knowing: **`Ternary-Bonsai-27B-Q2_0.gguf` is not upstream `Q2_0`** — it
does not load in llama.cpp either. It's packed **g128** (one f16 scale per 128 weights) while
`GGML_TYPE_Q2_0` is **g64** (18 bytes per 64 weights); use `Ternary-Bonsai-27B-Q2_g64.gguf`
instead. The frontend rejects the mispacked file safely (`data runs past EOF`) rather than
dequantizing garbage.

`muse-glimmer`'s row was decided by the *tokenizer*, not the graph: the converted graph reproduces
llama.cpp token-for-token, but GenAI's GGUF tokenizer builder only honored
`tokenizer.ggml.add_bos_token` on the SentencePiece path, silently dropping the leading BOS on the
BPE (`gpt2`) path that this (BOS-sensitive) model uses. Same gap affected `llama3`/`mistral3` the
same way; fixed in `gguf_tokenizer.cpp` by emitting BOS/EOS as a `CombineSegments` segment on every
tokenizer path.

## Adding a new architecture

Support for a new architecture is a combination of:
1. **Ops** — every ggml op in the model's compute graph must have a frontend translator
   (`src/op/<name>.cpp`) and backend admission.
2. **Weights** — every quantization format used by the model's tensors must be handled by
   the weight path (`src/quant/weights.cpp`).
3. **Real-model verification** — run a real `.gguf` end-to-end as above before adding the
   architecture to the Supported table.

For the native builder specifically, see
[`adding_an_architecture.md`](adding_an_architecture.md) — for a same-family arch the change
is usually just adding the name to `experimental_archs()` plus the `arch_uses_neox_rope()`
classification, and promotion to `verified_archs()` should require the GenAI-vs-llama.cpp
comparison above.
