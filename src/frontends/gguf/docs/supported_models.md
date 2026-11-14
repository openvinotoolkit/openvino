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
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | Language backbone: dense lead layers, shared/routed experts |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | Interleaved MoE, biased selection, normalized weights, shared experts |
| `exaone-moe` | K-EXAONE-236B-A23B Q2_K | Dense lead, shared/routed experts, RoPE on SWA layers only |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | Post-norm-only attention and FFN |
| `gemma` | gemma-2b Q4_K_M | GeGLU, multi-query attention |
| `gemma2` | gemma-2-2b-it Q4_K_M | Post-norms, SWA, attention and final soft-caps |
| `gemma3` | gemma-3-1b-it Q4_0 | Post-norms, final soft-cap, separate global/local RoPE scaling |
| `gemma4` | gemma-4-E2B-it Q4_0 | SWA, per-layer embeddings, shared KV, dense plus routed experts |
| `glm4moe` | GLM-4.5-Air REAP50 Q2_K | Biased sigmoid selection, shared experts, optional QK-norm |
| `gpt-oss` | gpt-oss-20b Q4_0 | MoE, attention sinks, SWA, OAI gated activation; **no numerical fixture**; CPU expands MXFP4 experts to F32 at compile time (about 120 GiB for 20B) |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | Learned QK-norm after RoPE |
| `hunyuan-moe` | Hunyuan-A13B-Instruct Q2_K | QK-norm after RoPE, routed and shared experts |
| `jais2` | Jais-2-8B-Chat Q4_K_M | Biased LayerNorm, ungated ReLU² FFN |
| `lfm2` | LFM2-350M / LFM2.5-1.2B-Instruct Q4_0 | LFM2 / LFM2.5 dense hybrid short convolution and GQA, QK-norm, tied head |
| `lfm2.audio_vocoder` | LFM2.5-Audio-1.5B Q4_0 ([component parity](mmproj.md#lfm2-audio-components)) | Standalone speech component; accepts the existing publisher architecture string |
| `lfm2moe` | LFM2.5-8B-A1B Q4_0 | LFM2 / LFM2.5 MoE hybrid, dense lead, biased normalized expert routing |
| `lfm3` | LFM3-1.7B / LFM3-VL-2B / LFM3-Audio-2B / LFM3-Omni-2B Q4_0 | Dense hybrid convolution/GQA, per-head output gates, paired optional post norms, tied output, mixed full/sliding RoPE |
| `llama` | Devstral Small 2507 24B Q4_K_M | llama-2 / llama-3, YaRN |
| `llama-embed` | Llama-Nemotron-Embed-1B-v2 Q4_K_M | Per-token embeddings; mean/first/last pooling; causal or bidirectional |
| `maincoder` | Maincoder-1B Q4_K_M | NORMAL RoPE, QK-norm after RoPE |
| `mamba2` | Mamba2-2.7B Q8_0 | Mamba 2 mixer, tied or separate output; one-sequence greedy decoding |
| `mellum` | Mellum2-12B-A2.5B-Instruct Q4_K_M | MoE with normalized expert weights |
| `minicpm` | MiniCPM-2B-sft Q4_0 | Embedding/residual scales, inverse logit scale |
| `minimax-m2` | MiniMax-M2.1-REAP-50 Q2_K | Full-width QK-norm, partial RoPE, normalized experts |
| `mistral3` | Devstral Small 2 24B; Ministral-3-3B-Instruct-2512 Q4_K_M | Dense, NORMAL RoPE, attention temperature |
| `muse-glimmer` | Muse-Glimmer-30B Q4_0 | RoPE on SWA layers, attention output gate, pre/post norms |
| `nemotron_h` | Nemotron-H-8B-Reasoning-128K Q4_K_M | Hybrid Mamba 2 / attention / ReLU² FFN; one-sequence greedy decoding |
| `olmoe` | OLMoE-1B-7B-0924-Instruct Q4_0 | Full-width QK-norm, MoE |
| `phi3` | Phi-3-mini-4k-instruct Q4_0 | Fused QKV |
| `plamo3` | plamo-3-nict-2b-base Q4_K_M | Fused QKV and gate/up, bare post-norms, SWA |
| `qwen2` | Qwen2.5-0.5B-Instruct Q4_0 | qwen2 / qwen2.5 |
| `qwen3` | Qwen3-0.6B Q4_0 | QK-norm before RoPE |
| `qwen35` | Qwen3.5-4B Q4_0; Ternary-Bonsai-2-27B PQ2_0; Ternary-Bonsai-2-4B/9B PQ2_0-MTP-Q8_0 | Hybrid GatedDeltaNet + attention, interleaved M-RoPE, folded Hadamard weights, embedded NextN MTP draft; batching and beam search with SDPA and PA |
| `qwen35moe` | Qwen3.6-35B-A3B Q4_K_M / Q4_0 | Hybrid GatedDeltaNet, separate or fused routed experts, shared expert |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | NEOX RoPE, per-head QK-norm, normalized experts |
| `smollm3` | SmolLM3-3B Q4_K_M | NORMAL RoPE, skipped on every fourth layer |

### Numerical regression coverage

[`GGUFArchitectureAccuracy`](../tests/test_arch_accuracy.cpp) runs 40 small nonzero F32 decoder
fixtures (every decoder above except `gpt-oss`, plus Devstral YaRN/temperature, Gemma4 MQA/MoE/PLE,
Qwen3.5 mixed and fused variants) and `GGUFEmbeddingAccuracy` five embedding fixtures. Each compares
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
