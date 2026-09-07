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
that loads but emits garbage is **not** counted as supported.

## Architectures accepted by the native `.gguf` builder

Everything above is about the **llama.cpp cgraph** path. This section covers the *other*
decoder — the native `.gguf` builder (`DecoderBuilder` in
[`src/builder/arch/decoder_builder.cpp`](../src/builder/arch/decoder_builder.cpp)), which is what
OpenVINO GenAI uses. The two paths share all op
translators but have separate architecture lists.

The builder's accept-list is the union of two sets, both defined in
[`src/builder/arch_registry.cpp`](../src/builder/arch_registry.cpp):

- **`verified_archs()`** — conversion, compilation and decoding checked against a reference
  on a real checkpoint. Numerical architecture regressions are described below.
- **`experimental_archs()`** — expected to work via the builder's GGUF-tensor-table
  auto-detection, but not end-to-end verified. These convert and emit a
  warning through `OPENVINO_WARN` so callers know they are best-effort.

Without additional architecture registrations, names outside these sets are rejected at load
time. External definitions and custom-family catalog entries extend the same registry.

### `verified_archs()` — 23 architectures

| Architecture | Notes |
|---|---|
| `llama` | llama-2 / llama-3 |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm before RoPE |
| `phi3` | fused QKV |
| `minicpm` | NORMAL RoPE; embedding/residual scales and inverse logit scale |
| `olmoe` | full-width QK-norm and MoE |
| `qwen35` | hybrid GatedDeltaNet + full attention, interleaved M-RoPE; greedy / batch 1 only |
| `gpt-oss` | MoE, attention sinks, SWA and OAI gated activation |
| `gemma3` | post-norms and final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings and shared KV |
| `hunyuan-dense` | learned QK-norm after RoPE |
| `qwen3moe` | NEOX RoPE, per-head QK-norm and normalized expert weights |
| `gemma` | GeGLU and multi-query attention |
| `gemma2` | post-norms, SWA, attention and final logit soft-caps |
| `exaone4` | post-norm-only attention and FFN |
| `ernie4_5-moe` | interleaved MoE, biased selection, normalized weights and shared experts |
| `bailingmoe2` | sigmoid routing, biased/grouped expert selection and shared experts |
| `maincoder` | NORMAL RoPE; learned QK-norm after RoPE |
| `mistral3` | dense decoder, NORMAL RoPE |
| `smollm3` | NORMAL RoPE, skipped on every fourth layer |
| `mellum` | MoE with normalized expert weights |
| `muse-glimmer` | RoPE on SWA layers, attention output gate, pre/post norms |
| `deepseek2-ocr` | language backbone: dense lead layers and shared/routed experts |

### `experimental_archs()` — 7 architectures

| Architecture | Remaining validation |
|---|---|
| `llama-embed` | needs embedding-specific numerical and pooling tests; completion is not a suitable test |
| `plamo3` | native builder disagrees with the real checkpoint reference (0/13 matching choices); post-norm tensors without `.weight` are not recognized |
| `hunyuan-moe` | small numerical fixture passes; no real checkpoint validated |
| `glm4moe` | no real checkpoint validated |
| `exaone-moe` | no real checkpoint validated |
| `minimax-m2` | no real checkpoint validated |
| `jais2` | no real checkpoint validated |

The decoder catalog stores each architecture name, RoPE mode, and maturity together.
`arch_uses_neox_rope()` is a derived lookup; `qwen35` uses interleaved multimodal RoPE.
Classify the mode against a reference before registering a new architecture.

### Native architecture regression coverage (2026-09-07)

`GGUFArchitectureAccuracy` checks 23 small, nonzero F32 models against complete logit
vectors produced by the real llama.cpp CPU backend. The cases cover all 13 newly
promoted architectures, Hunyuan MoE, six existing verified architectures, and three Devstral
configurations. The Devstral cases also run through a loaded external library. Each runs
multi-token prefill, one-token decode, and a two-token cache append. The normalized MSE
limit is `1e-5`. F32 inference, F16 KV cache and disabled activation quantization isolate
architecture behavior from optional CPU approximations.

The fixtures include single-head KV for Gemma, nonuniform QK-norm weights, Gemma2/Muse
sliding-window boundaries, SmolLM3's fourth NoPE layer, ERNIE shared experts without
`expert_shared_count`, and Bailing's sigmoid/grouped routing. Configuration tests also
cover Gemma2-27B's attention scale, EXAONE4's 64-layer SWA defaults and ERNIE's interleaved
dense/MoE schedule. Existing
zero-weight conversion fingerprints remain useful structural smoke tests.

Real checkpoints were additionally checked on the prompt `The capital of France is`
through native conversion, `GGUFMakeStateful`, `AdaptToGenAI` and CPU compilation.
The reference's next token is fed back for twelve decode steps, allowing comparison on
identical histories after a token choice differs. The table counts matching greedy
choices across prefill and those twelve steps. Q4_K checks use
`OV_GGUF_Q4_K_ZP_F16=1`; inference uses the precision settings above.

| Promoted architecture | Real checkpoint | Matching choices |
|---|---|---|
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | 13/13 |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | 13/13 |
| `gemma` | gemma-2b Q4_K_M | 12/13 |
| `gemma2` | gemma-2-2b-it Q4_K_M | 13/13 |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | 13/13 |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | 12/13 |
| `bailingmoe2` | Ling-mini-2.0 Q2_K | 12/13 |
| `maincoder` | Maincoder-1B Q4_K_M | 13/13 |
| `mistral3` | Ministral-3-3B-Instruct-2512 Q4_K_M | 13/13 |
| `smollm3` | SmolLM3-3B Q4_K_M | 12/13 |
| `mellum` | Mellum2-12B-A2.5B-Instruct Q4_K_M | 12/13 |
| `muse-glimmer` | Muse-Glimmer-30B Q4_0 | 13/13 |
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | 13/13 |

The real-model check requires the same first prediction and at least 90% matching
choices. It records full-logit errors but does not apply the F32 fixture tolerance to
lossy weight conversions. This is bounded decoder validation, not a quality benchmark
or a guarantee for every checkpoint, context length, quantization or device. In particular,
default integer-zero-point Q4_K and U8 KV-cache approximations can change output.

See [reference generation and reproduction instructions](../tests/test_data/arch_accuracy/README.md).
These references are shipped with the frontend tests; no model download or llama.cpp
build is needed in the regular test run. Real-model hub tests separately exercise
additional checkpoint/quantization combinations in precommit/nightly jobs.

### Devstral text models

Devstral Small 2507 (`llama`) and Devstral Small 2 (`mistral3`) now have real 24B Q4_K_M
checkpoint checks and numerical fixtures through both native and external registration.
Both real models match all 13 reference choices on the tested prompt. Shared fixes handle
YaRN metadata and position-dependent attention temperature. Devstral 2 has small-fixture
coverage; its 123B checkpoint is not verified. Vision and tool orchestration are separate.
See [Devstral support, validation and integration effort](devstral_support.md).

### Historical GenAI audit (before the architecture fixes)

The following table records the earlier generation audit; its set labels and failures
are historical, not the current registry status. Checkpoints marked “not tested” were
not run. This audit motivated the fixes and promotions above.

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

The Qwen3 MoE RoPE mode, Hunyuan QK-norm order, EXAONE4 norm placement, and ERNIE
routing/shared-expert defects from this audit are now covered by numerical regressions.
Gemma's single-KV-head path also passes the new prefill/decode fixture and real-model check.

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
is usually adding a decoder-catalog entry with its name, RoPE mode and experimental maturity.
Promotion to verified maturity should require the GenAI-vs-llama.cpp comparison above.
