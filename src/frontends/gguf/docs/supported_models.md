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
| `afmoe` | Trinity-Nano-Preview BF16 | Dense lead, routed/shared experts, sigmoid gating, pre/post norms, three sliding layers then one global NoPE layer |
| `bailingmoe2` | Ling-mini-2.0 Q2_K | Sigmoid routing, biased/grouped selection, shared experts |
| `cohere2` | c4ai-command-r7b-12-2024 Q4_0 | Parallel LayerNorm attention/FFN; RoPE on local layers |
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | Language backbone: dense lead layers, shared/routed experts |
| `dflash` | Qwen3.6-35B-A3B Q4_K_M / DFlash BF16 | Context-only cache commits, noncausal noise block, optional post-norms/scales and shared V |
| `eagle3` | Qwen3-8B Q8_0 / EAGLE3 F16 | Target feature fusion, optional pre-FC/residual norms and vocabulary mapping |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | Interleaved MoE, biased selection, normalized weights, shared experts |
| `exaone-moe` | K-EXAONE-236B-A23B Q2_K | Dense lead, shared/routed experts, RoPE on SWA layers only |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | Post-norm-only attention and FFN |
| `gemma` | gemma-2b Q4_K_M | GeGLU, multi-query attention |
| `gemma2` | gemma-2-2b-it Q4_K_M | Post-norms, SWA, attention and final soft-caps |
| `gemma3` | gemma-3-1b-it Q4_0 | Post-norms, final soft-cap, separate global/local RoPE scaling |
| `gemma3n` | Gemma3n E2B Q4_0 | AltUp, Laurel, sparse FFN, per-layer embeddings and shared KV |
| `gemma4` | gemma-4-E2B-it Q4_0 | SWA, per-layer embeddings, shared KV, dense plus routed experts |
| `glm4moe` | GLM-4.5-Air REAP50 Q2_K | Biased sigmoid selection, shared experts, optional QK-norm |
| `gpt-oss` | gpt-oss-20b Q4_0 | MoE, attention sinks, SWA, OAI gated activation; CPU expands MXFP4 experts to F32 at compile time (about 120 GiB for 20B) |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | Learned QK-norm after RoPE |
| `hunyuan-moe` | Hunyuan-A13B-Instruct Q2_K | QK-norm after RoPE, routed and shared experts |
| `jais2` | Jais-2-8B-Chat Q4_K_M | Biased LayerNorm, ungated ReLU² FFN |
| `lfm2` | LFM2-350M / LFM2.5-1.2B-Instruct Q4_0; LFM2.5-Encoder-230M/350M, -Embedding-350M, -ColBERT-350M Q8_0 | LFM2 / LFM2.5 dense hybrid short convolution and GQA, QK-norm, tied head; noncausal encoders with pooling and `dense_2` projection |
| `lfm2.audio_vocoder` | LFM2.5-Audio-1.5B Q4_0 ([audio components](mmproj.md#lfm2-audio-components)) | Standalone speech component; accepts the existing publisher architecture string |
| `lfm2moe` | LFM2.5-8B-A1B Q4_0 | LFM2 / LFM2.5 MoE hybrid, dense lead, biased normalized expert routing |
| `lfm3` | LFM3-1.7B / LFM3-VL-2B / LFM3-Audio-2B / LFM3-Omni-2B / D1-2B Q4_0 | Dense hybrid convolution/GQA, per-head output gates, paired optional post norms, tied output, mixed full/sliding RoPE; stateless D1 whole-prompt logits |
| `llama` | Devstral Small 2507 24B Q4_K_M | llama-2 / llama-3, YaRN |
| `llama-embed` | Llama-Nemotron-Embed-1B-v2 Q4_K_M | Per-token embeddings; mean/first/last pooling; causal or bidirectional |
| `maincoder` | Maincoder-1B Q4_K_M | NORMAL RoPE, QK-norm after RoPE |
| `mamba2` | Mamba2-2.7B Q8_0 | Mamba 2 mixer, tied or separate output; one-sequence greedy decoding |
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

### Runtime limitations

- **Recurrent states:** Gated-DeltaNet states support batching and beams through `AdaptToGenAI`;
  Mamba 2 and `nemotron_h` support one sequence with greedy decoding and resettable caches, without
  GenAI paged integration; beam search and prefix-cache reuse are untested. Mamba 1 and
  `nemotron_h_moe` are not supported. See [runtime.md](runtime.md#stateful-and-genai-conversion).
- **Multimodal:** a tested language backbone does not cover its vision or audio encoders,
  preprocessing or application pipeline.
- **Model variants:** the `lfm3` builder supports causal autoregressive backbones and stateless D1
  (`lfm3.attention.causal=false`) whole-prompt logits; the LFM2 Decision head
  (`lfm2.decision.block_count`) is unsupported. D1 has no GenAI adaptation; its iterative denoising
  sampler is left to the application.
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
`llama`, `qwen2` and `qwen3` models only.

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

The standalone `lfm2.audio_vocoder` architecture loads LFM2.5 Audio depthformer and
codec components; the existing publisher architecture string is also accepted.
It consumes hidden states and codec codes rather than text tokens; see
[audio component inputs and outputs](mmproj.md#lfm2-audio-components).

Noncausal LFM2.5 encoder, embedding and ColBERT checkpoints (`lfm2.attention.causal=false`)
convert to stateless models with one `embeddings` output: centered short convolutions,
bidirectional attention, then mean/CLS/last pooling (`lfm2.pooling_type`) and the optional
`dense_2` projection, matching llama.cpp. Without pooling they return per-token rows, one model
call per sequence; inputs are `inp_tokens` and `inp_pos` (positions `0..T-1`). Noncausal
attention is accepted only for `lfm2`, not `lfm2moe`. Five F32 fixtures cover per-token, mean-with-biased-projection, CLS, last and
ColBERT outputs. LFM3 D1 (`lfm3` with `lfm3.attention.causal=false`) reads a whole prompt with
bidirectional full-attention layers and causal sliding-window layers. Its inputs are
`inp_tokens`, `inp_pos`, `self_kq_mask_swa` and `inp_out_ids`; it returns logits for the rows
selected by `inp_out_ids` (pass `0..T-1` for every token) and has no KV or convolution state.

Real checkpoints were compared with llama.cpp `604cc430239c76ff636c443e3400bf552b750421` CPU on
identical tokens. Q8_0 NMSE is `1.29e-4` for LFM2.5-ColBERT-350M, `3.12e-4` (CLS) for LFM2.5-Embedding-350M, `1.77e-3` for
LFM2.5-Encoder-230M and `8.94e-4` for LFM2.5-Encoder-350M; the same weights represented in F32
give `6.7e-8`, `2.3e-7`, `1.2e-6` and `4.4e-7`. D1-2B Q4_0 matches every top-1 token for six prompt
positions with NMSE `3.0e-4..3.8e-3`, and `<1e-10` represented in F32. Revisions are
`bc240003aba07253e261a8aaf0d2c9683318a967`, `a80de9c5b941d429104f0038292a0ef5a860e486`,
`bb26e78c7164b386028928ebfcecebcc42467689`, `6559103d6c7136150cf989a4eda1462e5ef7b0c4` and
`4a181094d1b6addfaa3a0956f78e7ab2aa406a0c` from LiquidAI's GGUF repositories.

LFM2.5-Audio's `lfm2a`
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
