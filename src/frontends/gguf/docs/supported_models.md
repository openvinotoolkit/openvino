# GGUF Frontend — Supported Models

The frontend accepts graphs from two paths:

- **Native GGUF:** builds a graph directly from a `.gguf` file using the architecture catalog.
- **llama.cpp cgraph:** converts a graph constructed by llama.cpp through its OpenVINO backend.

The paths share operation converters, but architecture coverage is validated separately.
Validation applies to a tested checkpoint and configuration; it does not guarantee support
for every model, modality, context length or quantization using the same architecture name.

## llama.cpp cgraph path

Validated architectures and model examples:

| Architecture | Tested model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |
| `qwen35`  | Qwen3.5-4B (Q4_K_M)               | Hybrid GatedDeltaNet + full-attention layers; partial-rotary IMROPE; interleaved Q/gate joint projection; f16 KV cache. |

Checkpoint validation for this path covers: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

### Validation

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run passes validation only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage is **not** counted as supported.

## Native GGUF path

Vision/audio projector files (`mmproj-*.gguf`) are covered by
[native multimodal conversion](mmproj.md).

The native architecture catalog in
[`src/builder/arch_registry.cpp`](../src/builder/arch_registry.cpp) has one list of supported
architectures, including the `clip` mmproj handler. All entries use the builder registry
without maturity categories. Architecture extensions add their names to the supported
list of the frontend instance on which they are registered. Derived `ProjectorExtension`
registrations extend the separate [projector list](mmproj.md#supported-projectors).
Checkpoint and numerical coverage are documented separately below.

Without additional architecture registrations, names outside this catalog are rejected at load
time. External definitions and custom-family catalog entries extend the same registry.

### Supported architectures

| Architecture | Notes |
|---|---|
| `bailingmoe2` | sigmoid routing, biased/grouped expert selection and shared experts |
| `clip` | multimodal projector builder; supported projector types and modalities are listed in [multimodal conversion](mmproj.md) |
| `deepseek2-ocr` | language backbone: dense lead layers and shared/routed experts |
| `ernie4_5-moe` | interleaved MoE, biased selection, normalized weights and shared experts |
| `exaone-moe` | dense lead, shared/routed experts, RoPE on sliding-window layers only |
| `exaone4` | post-norm-only attention and FFN |
| `gemma` | GeGLU and multi-query attention |
| `gemma2` | post-norms, SWA, attention and final logit soft-caps |
| `gemma3` | post-norms and final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings, shared KV, and dense plus routed experts (26B-A4B) |
| `glm4moe` | biased sigmoid expert selection, shared experts and optional QK-norm |
| `gpt-oss` | MoE, attention sinks, SWA and OAI gated activation |
| `hunyuan-dense` | learned QK-norm after RoPE |
| `hunyuan-moe` | QK-norm after RoPE, routed and shared experts |
| `jais2` | biased LayerNorm and ungated ReLU-squared FFN |
| `llama` | llama-2 / llama-3 |
| `llama-embed` | per-token embeddings; mean, first-token and last-token pooling; causal or bidirectional attention |
| `maincoder` | NORMAL RoPE; learned QK-norm after RoPE |
| `mamba2` | Mamba 2 mixer, tied or separate output embeddings; stateful greedy decoding with one sequence |
| `mellum` | MoE with normalized expert weights |
| `minicpm` | NORMAL RoPE; embedding/residual scales and inverse logit scale |
| `minimax-m2` | full-width QK-norm, partial RoPE and normalized expert weights |
| `mistral3` | dense decoder, NORMAL RoPE |
| `muse-glimmer` | RoPE on SWA layers, attention output gate, pre/post norms |
| `nemotron_h` | dense hybrid Mamba 2 / attention / ReLU-squared FFN; stateful greedy decoding with one sequence |
| `olmoe` | full-width QK-norm and MoE |
| `phi3` | fused QKV |
| `plamo3` | fused QKV and gate/up, bare post-norm tensors, sliding-window attention |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm before RoPE; also covers Bonsai-8B (Q1_0) |
| `qwen35` | hybrid GatedDeltaNet + full attention, interleaved M-RoPE; batching and beam search with SDPA and PA; also covers Bonsai-27B (Q1_0) and Ternary-Bonsai-27B (Q2_0) |
| `qwen35moe` | hybrid GatedDeltaNet with separate or fused routed experts and a shared expert |
| `qwen3moe` | NEOX RoPE, per-head QK-norm and normalized expert weights |
| `smollm3` | NORMAL RoPE, skipped on every fourth layer |

### Numerical regression coverage

[`GGUFArchitectureAccuracy`](../tests/test_arch_accuracy.cpp) contains 40 small, nonzero F32
fixtures across the decoder families. `GGUFEmbeddingAccuracy` adds five
fixtures for token embeddings, pooling and bidirectional attention.
Additional model variants exercise YaRN and position-dependent attention scaling under
`llama` and `mistral3`. Gemma3 covers distinct global/local RoPE scaling, and Gemma4 covers
mixed-head MQA and dense plus routed experts. The suite does not cover `gpt-oss`; Gemma4 per-layer embeddings and shared KV are covered by `gemma4-ple`. For `qwen35`, `qwen35moe` and `gemma4`,
`GGUFMultimodalBackboneAdaptation` also checks padded batches and beam reordering against
independent requests, and PagedAttention conversion.

Each fixture compares complete last-token logits with llama.cpp CPU through multi-token
prefill, one-token decode and a two-token cache append. The normalized MSE limit is `1e-5`.
F32 inference, F16 KV cache and disabled activation quantization isolate architecture behavior
from optional CPU approximations.

The fixtures include single-head KV for Gemma, nonuniform QK-norm weights, Gemma2/Muse
sliding-window boundaries, SmolLM3's fourth NoPE layer, ERNIE shared experts without
`expert_shared_count`, and Bailing's sigmoid/grouped routing. Configuration tests also
cover Gemma2-27B's attention scale, EXAONE4's 64-layer SWA defaults and ERNIE's interleaved
dense/MoE schedule. Zero-weight conversion fingerprints provide structural smoke coverage.

### Mamba 2 execution

`GGML_OP_SSM_SCAN` uses the existing `SelectiveSSM` operation. `GGUFMakeStateful` normalizes
convolution caches for OpenVINO's recurrent fusions. Stateful execution supports greedy
decoding with one sequence and resettable convolution/SSM caches.

CPU tests cover prefill, decode and state reset through the stateful frontend path.
GenAI paged integration is not included. Beam search and prefix-cache reuse are not tested.
Mamba 1 and `nemotron_h_moe` are not supported.

Regenerate the Mamba references with `gen_arch_accuracy.py --oracle <oracle>
--architectures mamba2 mamba2-tied nemotron_h`, using
`architecture_oracle.cpp` built against llama.cpp `476c01efe88aad7880a8132d5d3a415f2ca75139`.

### Reference-checked checkpoints

The following checkpoint comparisons use the prompt `The capital of France is`, native
conversion, `GGUFMakeStateful`, `AdaptToGenAI` and CPU compilation.
The reference's next token is fed back for twelve decode steps, allowing comparison on
identical histories after a token choice differs. The table counts matching greedy
choices across prefill and those twelve steps. Q4_K checks use
`OV_GGUF_Q4_K_ZP_F16=1`; inference uses the precision settings above.

| Architecture | Real checkpoint | Matching choices |
|---|---|---|
| `bailingmoe2` | Ling-mini-2.0 Q2_K | 12/13 |
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | 13/13 |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | 12/13 |
| `exaone-moe` | K-EXAONE-236B-A23B Q2_K | 11/13; staged CPU chat passes separate functional checks |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | 13/13 |
| `gemma` | gemma-2b Q4_K_M | 12/13 |
| `gemma2` | gemma-2-2b-it Q4_K_M | 13/13 |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | 13/13 |
| `hunyuan-moe` | Hunyuan-A13B-Instruct Q2_K | 13/13 |
| `glm4moe` | ArliAI GLM-4.5-Air REAP50 Creative Q2_K | 12/13 |
| `jais2` | Jais-2-8B-Chat Q4_K_M, represented-weight F32 reference | 13/13 |
| `llama` | Devstral Small 2507, 24B Q4_K_M | 13/13 |
| `maincoder` | Maincoder-1B Q4_K_M | 13/13 |
| `mamba2` | Mamba2-2.7B Q8_0 | 12/13 |
| `mellum` | Mellum2-12B-A2.5B-Instruct Q4_K_M | 12/13 |
| `minimax-m2` | MiniMax-M2.1-REAP-50 Q2_K | 12/13 |
| `mistral3` | Devstral Small 2, 24B Q4_K_M (text) | 13/13 |
| `mistral3` | Ministral-3-3B-Instruct-2512 Q4_K_M | 13/13 |
| `muse-glimmer` | Muse-Glimmer-30B Q4_0 | 13/13 |
| `nemotron_h` | NVIDIA Nemotron-H-8B-Reasoning-128K Q4_K_M | 12/13 |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | 13/13 |
| `plamo3` | plamo-3-nict-2b-base Q4_K_M | 12/13 |
| `qwen35moe` | Qwen3.6-35B-A3B Q4_K_M / Q4_0 | 13/13 / 12/13 |
| `smollm3` | SmolLM3-3B Q4_K_M | 12/13 |

K-EXAONE-236B-A23B Q2_K passes functional checks for staged OpenVINO CPU chat.
Its raw-completion real-checkpoint comparison gives
11/13 matching choices, with the same first prediction. Steps 4 and 5 differ, so this
checkpoint has not passed the token-agreement threshold. The two F32 architecture fixtures,
including the NextN variant, pass. Replaying the same histories with llama.cpp and
F16 KV caches reproduces the recorded logits byte-for-byte with one and four CPU
threads. An F32-cache replay changes choices at steps 3, 4 and 5, demonstrating
precision sensitivity; this does not establish the cause of the OpenVINO difference.
A diagnostic llama.cpp run that streams each quantized weight row through ggml
dequantization and F32 matmul gives 12/13 against the original reference, differing
at step 4. Disabling weight repacking and the alternative GEMM alone gives 11/13,
differing at steps 3 and 4. These reference-side arithmetic changes demonstrate
variation in token rankings; OpenVINO's exact discrepancy remains unresolved.

A separate CPU chat check of the same checkpoint through OpenVINO completed three
single-turn prompts with coherent answers: Paris as France's capital, 19 pencils
remaining after `3 * 8 - 5`, and a two-sentence explanation of blue-light scattering.
All reached end-of-turn. The validation runner executed the frontend graph one
layer at a time because the reverted u2 expert path expands to F32 during CPU
compilation. CPU llama.cpp proposed draft continuations; every accepted token,
including end-of-turn, was checked against OpenVINO's greedy logits, and draft
mismatches were replaced with OpenVINO predictions. The runner passed the existing
EXAONE prefill/decode fixture with maximum NMSE below 2e-7. This establishes bounded
chat functionality for staged CPU execution; it does not validate a single compiled
full-model deployment or resolve the 11/13 raw-completion comparison.

Functional validation does not require identical wording: the chat check produced
correct answers despite differences from the llama.cpp draft. Numerical changes can
alter greedy token rankings without making the resulting answer incorrect. The three
prompts establish basic chat functionality, not general answer quality.

The raw-completion reference check requires the same first prediction and at least 90% matching
choices. It records full-logit errors but does not apply the F32 fixture tolerance to
lossy weight conversions. This is bounded decoder validation, not a quality benchmark
or a guarantee for every checkpoint, context length, quantization or device. In particular,
the default requantization of Q4_K, Q4_1 (u4) and Q5_K (u8) weights to integer zero points,
and the U8 KV cache, can change output. `OV_GGUF_Q4_K_ZP_F16=1` keeps the exact fractional zero
points of Q4_K and Q4_1 weights, at the cost of a slower matmul.

Native Q8_0 preserves weight codes and scales. These comparisons disable OpenVINO's dynamic
activation quantization (`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`) and use F32 inference.
llama.cpp's Q8 CPU path quantizes activations. Running llama.cpp with F32 arithmetic on
the same decoded weights, while keeping OpenVINO's settings unchanged, gives 13/13 matching
choices for Mamba2-130M and Mamba2-2.7B. The 130M checkpoint has model-hub smoke coverage but
falls below the agreement threshold against Q8 CPU arithmetic with OpenVINO dynamic
quantization disabled (11/13). Enabling it with group size 32 gives 13/13 matching choices
for both checkpoints against llama.cpp Q8 CPU in stateful execution. Full logits still
differ; their mean normalized error increases despite the improved token agreement.

Jais-2-8B-Chat gives 11/13 choices against llama.cpp's native quantized CPU arithmetic,
with the same first prediction. Comparing against F32 arithmetic on the exact same decoded
Q4_K_M weights gives 13/13 choices and maximum normalized logit error of 6e-6. The quantized
CPU mismatch remains a validation limitation.

Two-bit expert weights retain u2 storage in the frontend. The CPU compressed expert
matmul path does not currently support u2, so large Q2 MoE checkpoints can expand during
compilation. Repacked tensors have independent buffers so replacement does not retain an
entire model's original compressed allocation. Large MoE checkpoints still require substantial
RAM and swap for compilation and CPU weight reorders. The tested MiniMax-M2.1 REAP-50
Q2_K case peaked at about 222 GiB combined process RAM and swap on the validation machine;
it does not fit a 64 GiB nightly runner.

See [reference generation and reproduction instructions](../tests/test_data/arch_accuracy/README.md).
These references are shipped with the frontend tests; no model download or llama.cpp
build is needed in the regular test run. Real-model hub tests separately exercise
additional checkpoint/quantization combinations in precommit/nightly jobs.

The `llama-embed` checkpoint Llama-Nemotron-Embed-1B-v2 Q4_K_M matches per-token
embeddings against llama.cpp using F32 arithmetic on the represented weights (normalized MSE
`3e-6`; an F32 copy gives `2e-6`). This comparison isolates architecture behavior from
llama.cpp quantized activation arithmetic. Embeddings are returned as `[tokens, width]`,
pooled outputs as `[1, width]`; L2 normalization is left to the caller.

### Runtime limitations

- **Recurrent states.** `AdaptToGenAI` gives Gated-DeltaNet states (`qwen35`, `qwen35moe`) a
  dynamic batch dimension, reorders them by `beam_idx` and masks left padding out of their causal
  convolution. Mamba 2 and `nemotron_h` states stay at one sequence. GenAI must recognize
  `gguf_recurrent_states` metadata to reset, rather than trim, SDPA state.
- **Multimodal models:** a tested language backbone does not establish support for its
  vision or audio components, preprocessing or full application pipeline.
- **Ternary Bonsai packaging:** `Ternary-Bonsai-27B-Q2_0.gguf` uses g128 packing that does
  not match upstream `Q2_0` (g64). Use `Ternary-Bonsai-27B-Q2_g64.gguf`. The frontend
  rejects the mismatched file.

## Extending support

See [adding an architecture](adding_an_architecture.md) for catalog entries and shared decoder
features, and [porting a model from llama.cpp](porting_a_llama_cpp_model.md) for custom builders
and runtime extensions. New registrations require numerical fixtures and real-checkpoint
reference comparisons; successful conversion alone is insufficient.
