# GGUF frontend supported models

The frontend converts graphs from two paths that share operation converters but are validated
separately:

- **Native GGUF:** builds the graph from a `.gguf` file through the architecture catalog.
- **llama.cpp cgraph:** converts graphs built by llama.cpp's OpenVINO backend.

Validation applies to the tested checkpoint and configuration, not to every model, modality,
context length or quantization sharing an architecture name. Acceptance limits are defined in
[testing.md](testing.md#acceptance). Projector files (`mmproj-*.gguf`) are covered in [mmproj.md](mmproj.md).

## Native GGUF path

[`arch_registry.cpp`](../src/builder/arch_registry.cpp) holds one catalog; other names are rejected at
`load()` unless an [extension](extensions.md) registers them on that frontend. `clip` selects the
[mmproj coordinator](mmproj.md).

Tested models list real checkpoints and their quantization. `—` means no checkpoint is listed.

| Architecture | Tested model | Notes |
|---|---|---|
| `afmoe` | Trinity-Nano-Preview BF16 | Dense lead, routed/shared experts, sigmoid gating, pre/post norms, three sliding layers then one global NoPE layer; F32 oracle fixture |
| `bailingmoe2` | Ling-mini-2.0 Q2_K | Sigmoid routing, biased/grouped selection, shared experts |
| `cohere2` | c4ai-command-r7b-12-2024 Q4_0 | Parallel LayerNorm attention/FFN; RoPE on local layers |
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | Language backbone: dense lead layers, shared/routed experts |
| `dflash` | Qwen3.6-35B-A3B Q4_K_M / DFlash BF16 | Context-only cache commits, noncausal noise block, optional post-norms/scales and shared V |
| `eagle3` | Qwen3-8B Q8_0 / EAGLE3 F16 | Target feature fusion, optional pre-FC/residual norms and vocabulary mapping |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | Interleaved MoE, biased selection, normalized weights, shared experts |
| `exaone-moe` | K-EXAONE-236B-A23B Q2_K | Dense lead, shared/routed experts, RoPE on SWA layers only; below greedy acceptance threshold (11/13) |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | Post-norm-only attention and FFN |
| `gemma` | gemma-2b Q4_K_M | GeGLU, multi-query attention |
| `gemma2` | gemma-2-2b-it Q4_K_M | Post-norms, SWA, attention and final soft-caps |
| `gemma3` | gemma-3-1b-it Q4_0 | Post-norms, final soft-cap, separate global/local RoPE scaling |
| `gemma3n` | Gemma3n E2B Q4_0 | AltUp, Laurel, sparse FFN, per-layer embeddings and shared KV |
| `gemma4` | gemma-4-E2B-it Q4_0 | SWA, per-layer embeddings, shared KV, dense plus routed experts |
| `glm4moe` | GLM-4.5-Air REAP50 Q2_K | Biased sigmoid selection, shared experts, optional QK-norm |
| `gpt-oss` | gpt-oss-20b Q4_0 | MoE, attention sinks, SWA, OAI gated activation; **no numerical fixture**; CPU expands MXFP4 experts to F32 at compile time (about 120 GiB for 20B) |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | Learned QK-norm after RoPE |
| `hunyuan-moe` | Hunyuan-A13B-Instruct Q2_K | QK-norm after RoPE, routed and shared experts |
| `jais2` | Jais-2-8B-Chat Q4_K_M | Biased LayerNorm, ungated ReLU² FFN; 13/13 against represented F32, 11/13 against quantized CPU |
| `lfm2` | LFM2-350M / LFM2.5-1.2B-Instruct Q4_0 | LFM2 / LFM2.5 dense hybrid short convolution and GQA, QK-norm, tied head |
| `lfm2.audio_vocoder` | LFM2.5-Audio-1.5B Q4_0 ([component parity](mmproj.md#lfm2-audio-components)) | Standalone speech component; accepts the existing publisher architecture string |
| `lfm2moe` | LFM2.5-8B-A1B Q4_0 | LFM2 / LFM2.5 MoE hybrid, dense lead, biased normalized expert routing |
| `lfm3` | LFM3-1.7B / LFM3-VL-2B / LFM3-Audio-2B / LFM3-Omni-2B Q4_0 | Dense hybrid convolution/GQA, per-head output gates, paired optional post norms, tied output, mixed full/sliding RoPE |
| `llama` | Devstral Small 2507 24B Q4_K_M | llama-2 / llama-3, YaRN |
| `llama-embed` | Llama-Nemotron-Embed-1B-v2 Q4_K_M | Per-token embeddings; mean/first/last pooling; causal or bidirectional |
| `maincoder` | Maincoder-1B Q4_K_M | NORMAL RoPE, QK-norm after RoPE |
| `mamba2` | Mamba2-2.7B Q8_0 | Mamba 2 mixer, tied or separate output; one-sequence greedy decoding; 12/13 against Q8 CPU, 13/13 against represented F32 |
| `mellum` | Mellum2-12B-A2.5B-Instruct Q4_K_M | MoE with normalized expert weights |
| `minicpm` | MiniCPM-2B-sft Q4_0 | Embedding/residual scales, inverse logit scale |
| `minimax-m2` | MiniMax-M2.1-REAP-50 Q2_K | Full-width QK-norm, partial RoPE, normalized experts; see [memory limits](quantization.md#memory-and-packaging-limits) |
| `mistral3` | Devstral Small 2 24B; Ministral-3-3B-Instruct-2512 Q4_K_M | Dense, NORMAL RoPE, attention temperature |
| `muse-glimmer` | Muse-Glimmer-30B Q4_0 | RoPE on SWA layers, attention output gate, pre/post norms |
| `nemotron_h` | Nemotron-H-8B-Reasoning-128K Q4_K_M | Hybrid Mamba 2 / attention / ReLU² FFN; one-sequence greedy decoding |
| `olmoe` | OLMoE-1B-7B-0924-Instruct Q4_0 | Full-width QK-norm, MoE |
| `phi3` | Phi-3-mini-4k-instruct Q4_0 | Fused QKV |
| `plamo3` | plamo-3-nict-2b-base Q4_K_M | Fused QKV and gate/up, bare post-norms, SWA |
| `qwen2` | Qwen2.5-0.5B-Instruct Q4_0 | qwen2 / qwen2.5 |
| `qwen3` | Qwen3-0.6B Q4_0 | QK-norm before RoPE |
| `qwen3tts` | Qwen3TTS represented F32 | Speech decoder; projector and codec are separate components |
| `qwen35` | Qwen3.5-4B Q4_0; Ternary-Bonsai-2-27B PQ2_0 | Hybrid GatedDeltaNet + attention, interleaved M-RoPE, folded Hadamard weights; batching and beam search with SDPA and PA |
| `qwen35moe` | Qwen3.6-35B-A3B Q4_K_M / Q4_0 | Hybrid GatedDeltaNet, separate or fused routed experts, shared expert |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | NEOX RoPE, per-head QK-norm, normalized experts |
| `qwen3vl` | Qwen3-VL-2B Q4_0 | Interleaved multimodal RoPE and DeepStack inputs |
| `qwen3vlmoe` | Qwen3-Omni-30B-A3B Q4_K_M | Multimodal RoPE and routed experts |
| `smollm3` | SmolLM3-3B Q4_K_M | NORMAL RoPE, skipped on every fourth layer |

### Numerical regression coverage

[`GGUFArchitectureAccuracy`](../tests/test_arch_accuracy.cpp) runs small nonzero F32 decoder
fixtures, including Devstral YaRN/temperature, Gemma4 MQA/MoE/PLE and Qwen3.5 mixed and fused
variants; `GGUFEmbeddingAccuracy` covers embedding fixtures. Each compares
complete logits with llama.cpp CPU through prefill, decode and a two-token cache append. Fixtures
cover sliding-window boundaries, SmolLM3's NoPE layer, ERNIE without `expert_shared_count` and
Bailing grouped routing; see the [fixture README](../tests/test_data/arch_accuracy/README.md).
`GGUFMultimodalBackboneAdaptation` checks padded batches, beam reordering and PagedAttention for
`qwen35`, `qwen35moe` and `gemma4`. Zero-weight fingerprints add structural coverage.

### Runtime limitations

- **Recurrent states:** Gated-DeltaNet states support batching and beams through `AdaptToGenAI`;
  Mamba 2 and `nemotron_h` support one sequence with greedy decoding and resettable caches, without
  GenAI paged integration; beam search and prefix-cache reuse are untested. Mamba 1 and
  `nemotron_h_moe` are not supported. See [runtime.md](runtime.md#stateful-and-genai-conversion).
- **Multimodal:** a tested language backbone does not cover its vision or audio encoders,
  preprocessing or application pipeline.
- **Model variants:** the `lfm3` builder supports causal autoregressive backbones; Decision/D1 variants are unsupported.
- **Quantization and memory:** see [quantization.md](quantization.md).

## llama.cpp cgraph path

Manually validated with the backend's stateful execution:

| Architecture | Tested model | Notes |
|---|---|---|
| `llama` | TinyLlama-1.1B-Chat v1.0 Q4_K_M | Dense, GQA |
| `qwen2` | Qwen2.5-0.5B-Instruct Q8_0 | Dense |
| `qwen3` | Qwen3-0.6B Q8_0 | QK-norm |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M, Qwen3-4B Q4_K_M | `mul_mat_id` |
| `olmoe` | OLMoE-1B-7B-0924-Instruct Q4_0 | MoE |
| `gemma3` | gemma-3 (checkpoint not recorded) | Mixed SWA/global RoPE |
| `gemma4` | gemma-4-E4B-it Q4_K_M | Per-op RoPE, F16 KV cache |
| `qwen35` | Qwen3.5-4B Q4_K_M | GatedDeltaNet, partial-rotary IMROPE, interleaved Q/gate; F16 KV cache |

These runs record no llama.cpp revision. [llama.cpp compatibility CI](../../../../.github/workflows/job_gguf_llamacpp_validation.yml)
pins the backend revision and runs its operator tests plus state scenarios on generated
`llama`, `qwen2` and `qwen3` models only. A run passes when the output is coherent and consistent
with the ggml CPU backend:

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

Checkpoints used `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K` and `Q8_0`; other formats are covered by unit tests only.

## LFM2 and LFM2.5 coverage

Dense and MoE language models use the `lfm2` and `lfm2moe` GGUF architecture names,
respectively. The builder detects each convolution or attention layer from its weights;
it supports arbitrary layer ordering, separate or fused QKV, per-head QK normalization,
dense leading layers, softmax/sigmoid expert routing with selection bias, tied output
weights, and attention sliding windows. No model-size-specific layer schedule is hardcoded.
LFM2.5 8B-A1B and LFM2 24B-A2B use the same MoE builder; the 24B checkpoint has not been
replayed. LFM2-VL and LFM2.5-VL language backbones use `lfm2`; their separate SigLIP vision
projector is covered in [mmproj.md](mmproj.md).

Six nonzero F32 llama.cpp fixtures cover mixed layer order, tied heads, a sliding-window
boundary, fused QKV, dense-leading MoE, and all-routed sigmoid MoE. They compare full logits
through prefill, cached one/multiple-token decode, and state reset with NMSE below `1e-5`.
Batching, left padding, beam reordering, injected embeddings and structural conversion to
PagedAttention/PagedCausalConv1D are covered separately. This does not establish a GenAI
multimodal application or audio generation pipeline.

Real CPU reference-token replay uses llama.cpp `03fa73cb27f5c251b9528489b18d303b1366aca4`,
F32 inference, F16 KV, and disabled activation quantization. The prompt is
`The capital of France is`. Quantized-CPU logit NMSE ranges are `2.32e-4..6.20e-3` for
LFM2-350M, `3.99e-4..1.45e-3` for LFM2.5-1.2B-Instruct, and `2.98e-4..8.17e-2` for
LFM2.5-8B-A1B. These pass the real-checkpoint greedy-choice gate above, not the F32
fixture NMSE gate. Checkpoint revisions are `8fdc9d526b7ed346b19257551b05816c7912ecc2`,
`8ed288026e23958ad9dfa92d53ed773a8eee7125`, and
`49c14831707011e64d70b2ebd8462ba08d608434`, respectively, from LiquidAI's GGUF repositories.

The standalone `lfm2.audio_vocoder` architecture loads LFM2.5 Audio depthformer and
codec components; the existing publisher architecture string is also accepted.
It consumes hidden states and codec codes rather than text tokens; see
[audio component inputs and outputs](mmproj.md#lfm2-audio-components).

Noncausal LFM2 embedding/encoder/ColBERT variants need centered convolution and their
embedding/pooling output contracts and are rejected explicitly. LFM2.5-Audio's `lfm2a`
conformer projector and LFM2.5 speech components are supported as described in
[mmproj.md](mmproj.md#lfm2-audio-components). GenAI waveform preprocessing and generation
require the `gguf-frontend-mmproj` integration. Older Mimi-based LFM2 speech output
requires a separate codec. Other sizes, quantizations, long contexts,
and downstream fine-tunes require their own checkpoint validation.

## Block drafters

| Architecture | Features |
|---|---|
| `dspark` | Noncausal block attention, captured target features, log-SNR conditioning, optional Markov/confidence outputs |

DSpark accepts a fixed draft block in `inp_tokens` and captured target-layer features in
`dspark_ctx_feat` (`[1,1,context_rows,capture_count*embedding_width]`). Positions and cache-write
indices cover context plus draft rows; callers provide a fully open attention mask for the
noncausal round. Outputs include `logits`, `dspark.hidden`, per-layer caches and, when present,
`dspark.corrected_logits` and raw `dspark.confidence_score`. `dspark_prev_tokens` supplies the
sampled predecessor for each auxiliary-head row. Sequential sampling and target-layer capture
belong to the caller. Log-SNR uses Prism's round-1 inference convention: the first draft
position is at `max_log_snr`, and the remaining masked positions are at `min_log_snr`.
Normalization `(log_snr - min_log_snr) / (max_log_snr - min_log_snr) * 1000` maps these
endpoints to 1000 and 0, independent of the bounds. Both bounds must still be present,
finite and strictly ordered, as required by the model's conditioning metadata.
Hidden-correction checkpoints and generic GenAI decoder adaptation are not supported.
Draft blocks up to 4096 tokens are supported.

## 2026 family validation

Native checkpoint validation covers Cohere2, dense Qwen3VL, Gemma3n, EAGLE3 and the limited Qwen-backed DFlash path.
The independent small F32 fixtures use llama.cpp CPU revision
`098ce526b546b5a190327c1da82929cee051b9ac` and compare full logits through prefill,
cached decode and cache append. Gemma3n's fixture uses a scoped `2e-5` NMSE limit
and requires matching greedy predictions at every step; its final-step NMSE is
`1.65684e-5`. Other decoder fixtures retain `1e-5`. Gemma3n uses standard SDPA and
GenAI PagedAttention with the existing plugins. Its SDPA/PA full-logit regression
checks retain `1e-5`; Cohere2 also checks prefill, decode and multi-token append.

Original Cohere2 Q4_0, Qwen3-VL-2B Q4_0 and Gemma3n E2B Q4_0 each match all
13 checked greedy predictions. This meets the real-checkpoint gate: the first prediction
must match and at least 90% of the 13 choices must agree. Full-logit NMSE is recorded
separately; Gemma3n's maximum is `0.001477`, which does not use the synthetic-fixture
NMSE gate. Cohere2's repeated GenAI SDPA and PA outputs now agree. The original combined
Gemma3n projector passes the vision feature gate for one previously tested input and repeated
image generation; other tested inputs and the unsupported audio branch are described in
[mmproj.md](mmproj.md#added-2026-projector-paths).

An additional replay against llama.cpp `604cc430239c76ff636c443e3400bf552b750421`
matches 13/13 choices for Cohere2 and Gemma3n and 12/13 for Qwen3-VL. This is
teacher-forced agreement, not identical free-running generation or strict logit parity.

A real Llama-Nemotron-Embed-1B-v2 F16 checkpoint at revision
`effc8deb9688852dbf4ba458301999f7133f8eed` passes per-token embedding comparison
against the same pinned CPU oracle.

Drafts may omit token embeddings and the output head. Conversion then exposes
`gguf.target_token_embedding` and `gguf.target_lm_head` weight inputs; the GenAI
consumer clones target weight subgraphs before compilation. EAGLE3's reduced-vocabulary
`d2t` auxiliary constant contains absolute target token IDs; omitted tables use identity
mapping. GenAI must convert these IDs to the offsets expected by its sampler. DFlash
Markov, LiLiCorr, selector and convolution variants are unsupported.
The real Qwen3-8B Q8_0/EAGLE3 F16 pair passes repeated PA generation, two-request
continuous batching and target-only greedy agreement, with 23 of 68 draft tokens accepted
in each tested repetition. The Qwen3.6-35B-A3B Q4_K_M/DFlash BF16 pair passes repeated
single-request PA generation with prefix caching disabled, accepting 4 of 65 draft tokens
per repetition. These checks use the companion GenAI mapping and metrics fixes.
DeepSeek-V4 and multiple active DFlash requests are deferred.

### Quantized support and reference accuracy

The enabled families use the frontend's [implemented weight formats](quantization.md).
Quantized decoder and projector files are accepted; F32/F16 files establish the
reference graph behavior, rather than restricting which formats can be loaded.
The CPU checkpoint results below use llama.cpp revision
`098ce526b546b5a190327c1da82929cee051b9ac`. F32 materialization preserves the source
represented values in a separate GGUF file with `general.file_type=0`.
Known quantized numerical differences remain recorded. Model support depends on
floating graph correctness, integration checks and validated output quality;
quantized replay divergence alone does not disable a format. Passing conversion
alone does not establish numerical parity or generated-output quality.

AFMoE (Trinity-Nano-Preview Q4_K_S) and Qwen3-Omni-30B-A3B-Instruct Q4_K_M
are supported on CPU with GenAI PagedAttention. WWB validates their quantized text
quality against original-weight FP16 references using the same quantized file in
llama.cpp and GenAI; the token-replay differences below are numerical diagnostics,
with no material additional GenAI quality loss in the measured prompts.

| Quantized checkpoint | llama.cpp vs original FP16 | GenAI matched-input decoder vs same FP16 | GenAI default inputs vs same FP16 | GenAI matched minus llama.cpp |
|---|---:|---:|---:|---:|
| Trinity-Nano-Preview Q4_K_S (AFMoE) | 0.9174 | 0.9272 | 0.9275 | +0.0098 |
| Qwen3-Omni-30B-A3B-Instruct Q4_K_M | 0.9844 | 0.9832 | 0.9832 | −0.0011 |

These WWB mean semantic cosine scores use 20 prompts per model and at most 128
generated tokens, greedy sampling without penalties, CPU with four threads, F16 KV,
F32 inference hint, dynamic quantization disabled and `OV_GGUF_Q4_K_ZP_F16=1`.
They measure reference-answer agreement, not percent accuracy. The lowest AFMoE
score is 0.7204 on a flawed French translation shared by both quantized backends.
Omni's lowest GenAI score is 0.9133; reviewed low-score outputs remain coherent.
Image/audio/TTS, long-context and GPU quality are outside this WWB result.
In a separate CPU PA smoke test, Omni repeatedly identified a cat in a box and
transcribed a short speech clip; this is not a feature-parity or broader quality test.

References come from original publisher BF16 GGUFs converted to F16, with only
BF16/F32 input tensors; they are not expansions of quantized weights. Publisher
revisions are `arcee-ai/Trinity-Nano-Preview-GGUF@2aa08593b79242d224da0215fb36924dcc0f87ea`
and `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF@6e35a28f4a19b18730f8949b0c579c6429649ab8`.
Reference answers use the pinned llama.cpp CPU revision above. Scoring uses WWB
from GenAI `0169aa93d98811cdec1e4ea52e56396b9fbc4166` and
`sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`.

The matched-input runs use identical reference token IDs and EOG sets. AFMoE's
GenAI tokenizer segments ` $` and `-sentence` differently on two prompts; those
two prompts were regenerated with actual default string inputs, reusing 18
verified unchanged rows. Omni's GenAI default stop set omits llama.cpp's
`<|fim_pad|>`, `<|repo_name|>` and `<|file_sep|>` markers. All 20 Omni prompt
encodings match and none of those markers was sampled, so audited default-input
reuse gives the same score. These integration differences remain recorded. See
[WWB validation](../../../../.claude/skills/ov-gguf/references/wwb.md) for the method.

| Family | Floating-point reference evidence | Quantized accuracy observations |
|---|---|---|
| AFMoE | Original-weight F16 reference used by WWB; represented-F32 decoder matches 13/13 in the arithmetic diagnostic | Q4_K_S text quality supported by WWB above; token replay matches 11/13; Q4_0/DQ32 matches 7/13 |
| Qwen3-ASR | Original BF16 decoder matches 13/13; F32 `qwen3a` projector feature parity passes | Q8_0 decoder with CPU DQ32 matches 12/13, with DQ disabled 10/13; original BF16/Q8_0 projectors differ from llama.cpp |
| MiniCPM-V 4.7 | F32 `minicpmv4_7` projector passes 448×448 and 224×112 feature checks; existing `qwen35` decoder matches 13/13 | Original mixed BF16/F32 projector exceeds the feature parity gate |
| Qwen3TTS | Represented-F32 decoder matches 13/13; F32 speaker/predictor/codec projector checks pass | Original Q4_K_M decoder matches 8/13; original BF16/Q8_0 projectors differ from llama.cpp |
| Qwen3VL MoE / Qwen3-Omni | Original-weight F16 Omni reference used by WWB; represented mixed-F16 decoder matches 13/13; F32 combined projector feature parity passes | Omni Q4_K_M text quality supported by WWB above; post-merge token replay matches 11/13, below the 90% gate, with maximum logit NMSE `0.111346`; original Q8_0 projector differs from llama.cpp |

ASR's F32 projector feature NMSE is `4.60e-9` / `4.74e-8` for 100 / 200 frames.
Both the original BF16 decoder and Q8_0 decoder with CPU DQ32, paired with this projector, pass two pinned LibriSpeech clips
with aggregate WER `3.7%`, exact repeats, first-output cancellation, one terminal
callback, and reset after cancellation through the GenAI ASR pipeline.
MiniCPM's F32 projector feature NMSE is `7.65e-7` / `3.61e-6`.
Qwen3TTS's F32 path passes 14 full predictor logits, all 15 sampled codes, exact
feedback, and two 138,240-sample forced-code waveform chunks (NMSE `2.31e-6` /
`1.23e-6`). Its speaker encoder passes 100/200-frame feature checks. Standard
codec convolutions preserve llama.cpp's F16 operand rounding through serialization
and CPU compilation. GenAI passes two identical finite 3,840-sample outputs at
24 kHz using three generated tokens and a zero speaker vector. Speech quality,
voice cloning and longer generation are outside this validation; the current
Text2SpeechPipeline has no public cancellation callback.

Qwen3-Omni mixed-F16 preserves the source F32 router/norm tensors. The decoder
passes all 13 choices against its own llama.cpp CPU reference, with full-logit
NMSE up to `1.37e-2` recorded separately. The runtime keeps 144 expert tensors
in F16 and expands only selected experts, peaking at 27.2 GiB for this replay.
Repeated native SDPA and PA text outputs agree. Image and audio generation are validated
with PA and 16-token prefill chunks; unchunked SDPA image prefill exceeded the
RAM guard. See [projector and scheduler limits](mmproj.md#projector-reference-accuracy-and-runtime-limits).

### Deferred families

| Family | Remaining evidence |
|---|---|
| Qwen3Next | Canonical 80B-A3B Q4_K_M frontend attempt stopped at 12.56 GB available RAM. A complete first-layer audit keeps experts compressed, but source plus executor expert copies estimate 111 GiB before globals and temporary allocations. No completed full decoder or API result |
