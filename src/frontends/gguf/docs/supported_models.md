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
| `inkling2` | — | Inkling 2 Small backbone; clipped NoPE attention, relative bias, short convolutions, depth mixing and SITU MoE; F32 CPU oracle fixture |
| `inkling2_loop` | Inkling-2-Mini Q8_0 | Shared-weight gated loop with independent per-pass state; fixed/varying-pass F32 CPU oracle fixtures |
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
| `qwen35` | Qwen3.5-4B Q4_0; Ternary-Bonsai-2-27B PQ2_0; Ternary-Bonsai-2-4B/9B PQ2_0-MTP-Q8_0 | Hybrid GatedDeltaNet + attention, interleaved M-RoPE, folded Hadamard weights, embedded NextN MTP draft; batching and beam search with SDPA and PA |
| `qwen35moe` | Qwen3.6-35B-A3B Q4_K_M / Q4_0 | Hybrid GatedDeltaNet, separate or fused routed experts, shared expert |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | NEOX RoPE, per-head QK-norm, normalized experts |
| `qwen3vl` | Qwen3-VL-2B Q4_0 | Interleaved multimodal RoPE and DeepStack inputs |
| `qwen3vlmoe` | Qwen3-Omni-30B-A3B Q4_K_M | Multimodal RoPE and routed experts |
| `qwen4exp` | HF tiny-random Qwen3.8-Flash-Next (F32); no real checkpoint replayed (smallest GGUF is 83 GB plus a 51B-parameter n-gram table) | Qwen3.8-Flash-Next text model: GatedDeltaNet + block-sparse attention (QSA), hyper-connection streams, hashed n-gram embeddings, shared-expert MoE; see [coverage](#qwen38-flash-next-coverage) |
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

The cgraph converters for `LIGHTNING_INDEXER`, `TOP_K_MASK`, `DSV4_HC_PRE` (plain and gated) and
`DSV4_HC_POST`, used by `qwen4exp`, are tested against ggml CPU only.

These runs record no llama.cpp revision. [llama.cpp compatibility CI](../../../../.github/workflows/job_gguf_llamacpp_validation.yml)
pins the backend revision and runs its operator tests plus state scenarios on generated
`llama`, `qwen2` and `qwen3` models only.

Checkpoints used `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K` and `Q8_0`; other formats are covered by unit tests only.

## Qwen3.8-Flash-Next coverage

`qwen4exp` is the llama.cpp architecture of Qwen3.8-Flash-Next (`qwen4_exp` in Transformers), the
preview of the Qwen4 architecture. The builder converts the text model:

- Gated DeltaNet layers with a sigmoid output gate and q/k normalization `x / sqrt(sum(x^2) + eps)`;
  packed (`attn_qkv_gate`, `ssm_beta_alpha`) and separate projections.
- Qwen Sparse Attention layers as opset17 `QSAPrepare` -> `SparseAttentionIndexer` ->
  `SparseAttention`: blocks of `attention.compress_ratios` tokens, a budget of
  `attention.indexer.top_k` tokens and the query's incomplete block, with a partially rotated
  RMS-normalized index query. The graph follows the form that the plugin fusions of
  [#39309](https://github.com/openvinotoolkit/openvino/pull/39309) and
  [#39342](https://github.com/openvinotoolkit/openvino/pull/39342) match.
- Four hyper-connection residual streams (fused or separate `down`/`inject` projections) in place
  of every norm, including the output mixer.
- Hashed n-gram embeddings (PLE) before one layer, with a gated, dilated causal convolution.
- Shared-expert MoE.
- NextN (MTP) blocks. Unsloth publishes them as separate `mtp-*.gguf` heads; see
  [Qwen3.8-Flash-Next MTP](runtime.md#qwen38-flash-next-mtp).

Unsloth's release uses i-quants in every file below `UD-Q4_K_XL`, and the frontend does not convert
i-quants. `UD-Q5_K_XL`, `UD-Q6_K_XL`, `Q8_0` and `BF16` convert. `UD-Q4_K_XL` stores its n-gram
table as `IQ4_NL`; pair it with the `Q8_0` or `Q5_1` n-gram file.

Running the sparse operations requires plugin support. The GPU kernels are merged
([#38902](https://github.com/openvinotoolkit/openvino/pull/38902)), but the passes that lower
the core operations to them are still open: the indexer fusion in
[#39342](https://github.com/openvinotoolkit/openvino/pull/39342) and the dense attention fusion in
[#39309](https://github.com/openvinotoolkit/openvino/pull/39309). CPU support is in
[#39201](https://github.com/openvinotoolkit/openvino/pull/39201). Without plugin support the
qwen4exp tests skip. Validation, all on CPU with the kernels and fusions of #39309:

- The F32 `qwen4exp` fixture matches llama.cpp `0ec4027a2` logits exactly through prefill and
  cached decode, with two-token QSA blocks that make block selection active, plus state reset; a
  variant without `output.weight` covers the tied output head.
- [`optimum-intel-internal-testing/tiny-random-qwen38-flash-next`](https://huggingface.co/optimum-intel-internal-testing/tiny-random-qwen38-flash-next)
  converted to F32 GGUF: 13/13 greedy choices with NMSE at most `1e-6` for a 27-token prompt
  (six complete blocks against a two-block budget), and identical 12-token greedy output through
  GenAI `LLMPipeline` (`GGUF_READER="FRONTEND"`). Its layer types read `qwen_sparse_attention`,
  which the pinned llama.cpp converter maps to dense layers; it was converted as `full_attention`,
  the label of the real checkpoint.
- Left-padded batches with any pad id equal single-sequence runs; beam search runs.
- `EMBEDS_TO_LOGITS` moves the n-gram lookup into a separate model with the optimum-intel IO
  (`input_ids`, `ngram_history` -> `ngram_embeddings`, `new_ngram_history`) and matches token input.
- MTP: the `qwen4exp-mtp` fixture matches llama.cpp `0ec4027a2` for the draft and for the
  `mtp_target` hidden states through prefill and cached decode (NMSE below `1e-5`). The real Unsloth
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` header, filled with zero weights, converts, compiles and runs
  on CPU.

Limitations: PagedAttention conversion is not supported yet; no real checkpoint has been run
(memory); the vision projector is not covered; the llama.cpp cgraph path has converters for the new
operations but no end-to-end backend run.
## Inkling 2

Inkling 2 Small and Mini use `inkling2` and `inkling2_loop`. They share one builder: clipped
NoPE attention with RMS-normalized V, a learned relative-position bias, a query-dependent
attention sink, static and input-dependent short convolutions, depth-wise residual mixing,
sigmoid-routed SITU experts and padded-vocabulary trimming. Mini unrolls the metadata loop
count into one graph; the passes share weights and keep independent attention and convolution
states.

Design decisions:
- The relative bias is gathered from a per-token table by clamped query-key distance. Key
  positions are kept in an extra i32 state, so cached decode and reset need no extra inputs.
- Sliding-window masks are halved before the bias is added, so masked logits stay finite in
  F16.
- Q/K/relative clipping uses `ReduceL2` and keeps llama.cpp's `1e-6` denominator epsilon only
  for thresholds below 1. For larger thresholds, including the released models' 8 and 22.6,
  it changes the clip factor by at most `1e-6 / threshold`, and dropping it keeps the chain
  out of F32 on GPU.
- Input-dependent convolution taps are extra one-hot channels of the same `SSM_CONV`, reusing
  the regular convolution state.
- Experts are emitted as a direct expert-major `GatherMatmul`.

Limitations:
- Inference uses the stateful SDPA model on CPU and GPU, single-sequence and greedy.
  `SDPAToPagedAttention` does not support the model, so GenAI must use the SDPA attention
  backend.
- GPU decode is slow because the GPU plugin does not yet fuse SDPA with an attention sink.
- GenAI raw-text input requires the Inkling pretokenizer and a Tokenizers build providing
  `BPETokenizerIgnoreMerges`; otherwise supply pretokenized input. The published chat template
  is not supported, so set `GenerationConfig.apply_chat_template` to `false`.
- `IQ4_NL` is not supported; use a format such as `Q8_0` or `BF16`.
- Speculative/MTP components are not supported.

The `inkling2v` image encoder is described in [mmproj.md](mmproj.md#inkling-2-vision).

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
