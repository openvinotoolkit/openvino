# Native GGUF multimodal conversion (experimental)

Reference: llama.cpp `16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb`.
Muse Glimmer uses `03fa73cb27f5c251b9528489b18d303b1366aca4`, which adds its encoder.
The reference is used only by offline fixture generators and validation executables.
It is not a production dependency.

This implementation is partial. Conversion, encoder accuracy and GenAI generation
are separate qualification levels. A catalog entry does not certify every checkpoint,
preprocessing variant or associated language backbone.

## Priority: registered language backbones

Further work prioritizes mmproj checkpoints paired with language architectures in
the [native catalog](supported_models.md). Eligibility is determined by the language
file's exact `general.architecture`, not the model's brand or embedding width.
Verified language architectures take priority over experimental registrations.

| Model family | Eligible language architecture | Frontend conversion gap | Accuracy / GenAI gap |
|---|---|---|---|
| Gemma3 | `gemma3` | Initial vision path implemented | Broader checkpoint qualification; original Q4_K_M run failed; companion GenAI integration required |
| Qwen3.5 / 3.6 / 3.8 | `qwen35`, `qwen35moe` (experimental) | `qwen3vl_merger` implemented | Real encoder comparison, preprocessing, multimodal positions, DeepStack assembly and cached generation |
| MiniCPM-V/o resampler variants | Checkpoint-dependent `minicpm`, `llama`, `qwen2` | `resampler` implemented | Real checkpoint comparisons, tiling, query placement and GenAI adapter; audio qualified separately |
| MiniCPM-V 4.6 | `qwen35` | `minicpmv4_6` implemented; window/downsample merger oracle passes | Real checkpoints and full pipeline validation |
| Gemma4 | `gemma4` | `gemma4v`, `gemma4a`, `gemma4uv`, `gemma4ua` implemented | E2B Q8 vision/audio pass against dequantized F32 reference; other checkpoints, preprocessing and generation pending |
| Pixtral / Mistral multimodal | Checkpoint-dependent `llama`, `mistral3` | `pixtral` implemented, including patch merger and row separators | Ministral 3 Q8 passes against dequantized F32 reference; preprocessing and GenAI validation pending |
| DeepSeek-OCR / OCR2 | `deepseek2-ocr` | `deepseekocr`, `deepseekocr2` implemented; SAM and projector oracles pass | Real checkpoints, crops/layout preprocessing and OCR generation |
| Phi4 multimodal | `phi3` | `phi4` implemented; dynamic-resolution oracle passes | Real checkpoints, preprocessing and GenAI validation |
| Eligible LLaVA variants | Checkpoint-dependent `llama`, `qwen2` | `mlp` implemented | Checkpoint-specific feature selection, image assembly and generation |
| Eligible InternVL / Idefics / audio variants | Require an exact registered backbone | Several projectors implemented below | Pair-by-pair encoder and processor equivalence, then GenAI integration |

`qwen2vl`, `qwen3vl`, `llama4` and `gemma3n` are distinct,
unregistered language architectures. Their support does not follow from `qwen35`,
`qwen2`, `qwen3`, `llama` or `gemma3`. Projector conversion already present for
such families remains available, but their full pipelines are outside this priority.
MiniCPM numeric version identifiers describe reference dispatch; they are not a
promise of support for every similarly named marketing release.

## Implemented conversion

| Modality | Projectors | F32 CPU oracle | Real-checkpoint encoder | GenAI adapter |
|---|---|---|---|---|
| Vision | `gemma3` | Pass, separate/fused QKV and legacy FFN names | Pass, Gemma3 4B F16 projector | Gemma3, experimental |
| Vision | `muse-glimmer` | Pass, sparse/global attention, two-axis RoPE, resized positions and channel-outer pixel shuffle | Represented-weight F32 and faithful Q4_K decoding pass; default Q4_K requantization fails strict accuracy | Pending |
| Vision | `mlp` (including normalized variant) | Pass | Pending | Pending |
| Vision | `idefics3`, `janus_pro`, `internvl` | Pass | Pending | Pending |
| Vision | `resampler` | Pass, rectangular grids, explicit queries and legacy defaults | Pending | Pending |
| Vision | `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | Pass, rectangular grids, temporal merging, window ordering and DeepStack | Pending | Pending |
| Vision | `pixtral` | Pass, rectangular grids, patch merger, row separators | Pass, Ministral 3 Q8 against dequantized F32 reference | Pending |
| Vision | `phi4`, `minicpmv4_6` | Pass, resized positions / window attention and two-stage merging | Pending | Pending |
| Vision | `gemma4v`, `gemma4uv` | Pass, two-axis positions, clipping, pooling / patch normalization | E2B `gemma4v` Q8 passes against dequantized F32 reference; unified pending | Pending |
| Vision | `deepseekocr`, `deepseekocr2` | Pass, SAM local/global attention, position resize, tile rows, GQA, queries, overview separators | Pending | Pending |
| Audio | `gemma4a`, `gemma4ua` | Pass, context boundaries, causal convolution / waveform frames | E2B `gemma4a` Q8 passes against dequantized F32 reference; unified pending | Pending |
| Audio | `qwen2a`, `ultravox`, `voxtral`, `musicflamingo`, `meralion`, `glma` | Pass | Pending | Pending |

The legacy `qwen2.5o` name resolves to Qwen2.5 VL for vision and Qwen2 audio.
The global `clip.projector_type` key takes precedence over modality-specific keys.
Unknown projector/modality combinations fail conversion; a combined file never
silently drops an unsupported modality. InternVL's large RMS-normalized encoder
variant is implemented but has no dedicated numerical fixture yet.

## Graph boundary

Preprocessing belongs to the caller. All activation inputs are F32; indices are I32.

| Family | Inputs |
|---|---|
| Fixed-resolution vision | `vision.pixel_values [1,3,S,S]`, normalized NCHW |
| MiniCPM resampler | `vision.pixel_values [1,3,H,W]`; `vision.position_ids [1,1,1,T]`, learned position indices |
| MiniCPM resampler | `vision.position_h` and `vision.position_w [1,1,T,1]`, F32 patch coordinates for sinusoidal positions |
| Qwen vision | `vision.pixel_values [2,3,H,W]`, temporal pair; repeat a still image twice |
| Qwen vision | `vision.patch_indices [1,1,1,T]`, patch ordering; `vision.position_ids [1,1,1,4*T]`, four position sections |
| Qwen2.5 vision | Also `vision.attention_mask [1,1,T,T]`, additive window mask, and `vision.output_indices [1,1,1,T/4]`, inverse window order |
| Pixtral / Phi4 / Gemma4 vision | Dynamic `vision.pixel_values [1,3,H,W]`; Pixtral/Gemma4 also take I32 `vision.position_x`, `vision.position_y [1,1,1,T]` |
| Muse Glimmer | Dynamic pixels, one-based `position_x`/`position_y`, window `patch_indices`, inverse `output_indices`, pixel-shuffle `merge_indices`, and additive `attention_mask`; names are vision-qualified. `vision.window_size` records the learned position-grid side. |
| MiniCPM-V 4.6 | Pixels and learned position IDs; `window_indices`, `inverse_window_indices`, additive `attention_mask`, and four `vit_merger.indices.N` / `merger.indices.N` row selectors, all vision-qualified |
| DeepSeek-OCR / OCR2 | `vision.pixel_values [B,3,H,W]`, local/global relative-position index matrices, and `vision.output_indices` for reference token ordering |
| Gemma4 audio | `audio.features [1,1,mel,frames]`, `position_embeddings [1,1,13,D]`, additive `attention_mask [1,1,T,T]`, and I32 `relative_indices [1,1,T,T]` |
| Gemma4 unified audio | `audio.waveform_frames [1,1,T,640]`, raw 16 kHz waveform split into 640-sample frames |
| Whisper-derived audio | `audio.features [1,mel,1,frames]`, plus `audio.position_ids [1,1,1,ceil(frames/2)]` |

Qwen spatial dimensions must be divisible by patch size times merge size. Patch
indices apply spatial grouping and window permutation before the encoder; positions
must follow that order. Output indices restore the reference spatial ordering.
Audio feature extraction, chunking, valid lengths and token placement remain the
caller's responsibility; the learned position table bounds supported lengths.
MiniCPM resampler dimensions must be divisible by patch size; `T` is the patch
count. The caller supplies the reference's learned-position buckets and raster
coordinates. Accepted `clip.minicpmv_version` values are 2, 3, 4, 5, 6 and 100045;
missing/zero selects 2. Missing/zero `clip.minicpmv_query_num` selects 96 queries
for version 2 and 64 otherwise. Explicit positive query counts must match the
query tensor. Resolved version and query count are retained in runtime metadata.

Gemma4 vision takes pixels in [0,1] and performs the reference's `2*x-1` scaling
inside the encoder. Unified vision uses an effective patch size of metadata patch
size times projector scale factor. Pixtral's merge size defaults to one; Gemma4's
pooling/patch scale defaults to three.

OCR inputs contain independent square tiles (a tall reference tile row becomes a
batch). `output_indices` selects from flattened projected tiles followed by learned
newline (OCR1 only) and view-separator embeddings. OCR1 also takes
`position_indices`: select the original position table for an unchanged grid, or
the appended resized table otherwise. The resized table preserves the pinned
reference's legacy interpolation layout and byte offset for the class position.
OCR2 takes `query_indices` into the concatenated 144/256-query tables,
`query_output_indices`, actual `position_ids`, and the reference image/query mask.
OCR2 currently processes one tile per invocation. The caller decides overview
versus tile-row ordering; no separator is silently discarded.

Gemma4 audio preserves the reference's causal 12-position horizon, relative-score
shift and softcap. The current graph expresses the mask over the full subsampled
sequence, so attention memory grows quadratically with feature length. The caller
supplies masks and position indices; large recordings should be qualified separately.

Raw outputs are `vision.embeddings` and/or `audio.embeddings`, shape `[1,1,T,D]`.
Qwen3 packs primary and DeepStack features along D, in reference order.
`AdaptMmprojToGenAI` selects one reachable modality, removes unrelated inputs,
exposes `[1,T,D]`, removes modality prefixes from inputs and splits auxiliary outputs
as `image_features` and `deepstack_features.N` (or `audio_features`).
Run this pass on separate clones to retain both independently executable encoders.

Runtime metadata is stored under `gguf_mmproj`, including source `clip.*` keys,
resolved projector, merge size, output layout and feature roles. Numeric values use
strings, numeric arrays use comma-separated strings. String arrays use concatenated
`length:value` entries, identified by a companion `.encoding` value. This preserves
empty strings and delimiters. Metadata survives IR serialization and adaptation.
External cgraph decoders return an empty mmproj metadata map by default.

## GenAI integration

The separately delivered companion GenAI change accepts a language GGUF and
projector GGUF in memory. This OpenVINO branch alone does not add `mmproj_path`
to an installed GenAI package; the matching GenAI integration is required:

```python
pipe = openvino_genai.VLMPipeline(
    "language.gguf", "CPU", mmproj_path="mmproj.gguf",
    INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0)
result = pipe.generate("Describe the image.", images=[image],
                       max_new_tokens=20, do_sample=False)
```

The current adapter requires a Gemma3 pair. It constructs the tokenizer from GGUF
metadata, extracts a shared-weight text lookup model, and exposes `inputs_embeds`
while preserving state. Token embeddings receive Gemma scaling exactly once; image
embeddings compensate for that scaling before assembly. Image tokens use a
bidirectional mask within their image group. End-of-turn tokens stop generation.
No temporary IR or Hugging Face configuration directory is required.
With `enable_save_ov_model=True`, adapted models are saved as
`<language.gguf>.vlm.xml`, `<language.gguf>.embeddings.xml`, and
`<mmproj.gguf>.vision.xml`, retaining their runtime metadata.

Embedding-mode adaptation is batch-one SDPA. M-RoPE input positions are independent
sections `[4,1,T]`; the caller must supply actual spatial/temporal positions. Existing
text-mode conversion retains its previous interface. Auxiliary token lookups retain
`input_ids` when required by the language graph.

Audio generation, other GenAI family adapters, video preprocessing, beam search,
PagedAttention and GPU qualification are not implemented/qualified by this route.

## Reproduction and evidence

Recorded checkpoint results, including failed configurations, are in
[validation.json](../tests/test_data/mmproj_accuracy/validation.json).

Build `ov_gguf_frontend_tests` and `ov_gguf_architecture_library_tests`. Run both
without filters to include decoder architecture, quantization, extension and op
coverage gates. `tests/gen_mmproj_accuracy.py` regenerates small nonzero fixtures
using `tests/mmproj_oracle.cpp` linked against the pinned CPU-only libmtmd/libggml.
`tests/gen_muse_mmproj_accuracy.py` generates the Muse Glimmer static and dynamic-grid oracle checks.
`tests/gen_mmproj_supported_accuracy.py` adds 14 fixtures for the nine newly added
projector types, reusing compiled raw/adapted models across shapes and back again.
Gemma4 fixtures also cover one-sided clipping bounds and their reference defaults.
Pass `--geometry-oracle /path/to/mmproj_ops_oracle` to regenerate the standalone
window/relative-position expectations as well. Both oracle sources are in `tests/`.
`GGUF_ORACLE_DUMP=<directory>` enables intermediate reference tensor dumps.
`tests/gen_arch_accuracy.py` also includes Gemma3 global-vs-local RoPE scaling.
All synthetic numerical tests require normalized MSE below `1e-5`.

```sh
PYTHONPATH=<llama.cpp>/gguf-py python3 tests/gen_mmproj_accuracy.py --oracle /path/to/mmproj_oracle
python3 tests/validate_mmproj.py mmproj.gguf --oracle /path/to/mmproj_oracle --report encoder.json
```

Real checkpoint: `ggml-org/gemma-3-4b-it-GGUF`, revision
`d0976223747697cb51e056d85c532013931fe52e`.
The F16 mmproj SHA256 is
`8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb`.
Its 896x896 normalized-image encoder comparison passed at NMSE `2.97797e-6`.

The companion GenAI `tests/python_tests/validate_gguf_mmproj.py` compares first-token
and subsequent argmax choices against llama.cpp on identical generated-token
histories, then checks request reset. The solid-red-image/text run on the original
Q4_K_M language checkpoint **failed**: text 8/9 matching choices, image 17/20 with a
first-token mismatch. It must not be marked verified.

An F16 copy produced by the pinned `llama-quantize --allow-requantize ... F16`
from that same Q4_K_M checkpoint passed both 20-token cases at 100%, including
first tokens and request reset. This is a dequantized checkpoint, not the original
publisher's F16 weights; the result does not qualify Q4_K_M execution.
The same dequantized F16 pair also passed a natural-image comparison using the
reference repository's `tools/mtmd/test-1.jpeg` at 100% for 20 choices. A separate
13-step raw-decoder comparison passed first-token/choice checks with recorded
logit NMSE at or below approximately `1e-6`.
The optional `--chat` check passed a cached image-chat follow-up at 19/20 matching
choices (95%), including the first token, initial chat cache reset, and reset after
finishing chat. These are limited checkpoint/prompt results, not exhaustive media
or conversation qualification.

Further checkpoint evidence is recorded in
[supported_backbones_validation.json](../tests/test_data/mmproj_accuracy/supported_backbones_validation.json).
Gemma4 E2B vision/audio and Ministral 3 vision **pass with the published Q8_0 files
on OpenVINO against a llama.cpp F32 reference** (NMSE `2.95e-7`, `2.71e-8`,
`4.72e-7`, respectively). The reference copies are dequantized from those same
Q8_0 files, preserving their represented weights. Acceptance requires finite
normalized MSE strictly below `1e-5`; the report records the threshold and both
checkpoint hashes. These are encoder comparisons on synthetic media/features,
not GenAI generation or qualification against original publisher-F32 weights.

Use this F32 reference for quantized encoder conversion acceptance. Comparisons
against llama.cpp's quantized execution are separate diagnostics: its Q8 CPU
matmuls quantize activations, whereas this OpenVINO validation disables dynamic
activation quantization. Those comparisons exceed `1e-5` and are retained under
`quantized_execution_diagnostics`, without determining frontend acceptance.
OpenVINO Q8 versus OpenVINO dequantized F32 output NMSE is zero for Pixtral and
below `7e-13` for both Gemma4 branches in the isolation runs. To measure error
introduced when originally quantizing a checkpoint, use the publisher's matching
unquantized weights instead and qualify an appropriate threshold separately.

The pinned `llama-quantize` rejects `clip` files; use the offline
`tests/dequantize_mmproj.py` with the pinned gguf-py for the F32 copies:

```sh
PYTHONPATH=<llama.cpp>/gguf-py python3 tests/dequantize_mmproj.py mmproj-Q8_0.gguf mmproj-dequant-F32.gguf
python3 tests/validate_mmproj.py mmproj-Q8_0.gguf --reference-model mmproj-dequant-F32.gguf --oracle /path/to/mmproj_oracle --nmse-threshold 1e-5 --report encoder.json
```

Omitting `--reference-model` runs the same checkpoint in both runtimes, preserving
the previous diagnostic behavior. `validate_mmproj.py` also accepts `--width`,
`--height` and `--modality audio`, and extracts only the requested branch of
combined models. Remaining real-checkpoint gaps include Phi4, MiniCPM-V 4.6,
OCR/OCR2 and the Gemma4 unified variants.

## Remaining reference coverage

The following reference projector names still require builders, numerical fixtures
and checkpoint validation. This is a broader inventory; the registered-backbone
matrix above determines implementation priority:

`ldp`, `ldpv2`, `adapter`, `step3vl`, `gemma3nv`, `llama4`, `qwen3a`, `lfm2`, `kimivl`, `paddleocr`, `lightonocr`, `cogvlm`, `dots_ocr`, `lfm2a`, `glm4v`, `youtuvl`, `yasa2`, `kimik25`, `nemotron_v2_vl`, `exaone4_5`, `hunyuanvl`, `granite_speech`, `mimovl`, `granite4_vision`.

`gemma3na` is a named enum without a graph-builder dispatch in the pinned reference
and is an explicit exclusion. `mlp_norm` is an internal tensor-detected variant of
`mlp`, not a separately accepted projector metadata string.

Remaining delivery work includes other convolutional/Conformer topologies outside
the selected backbone scope; shared-family GenAI adapters;
preprocessing and embedding/logit equivalence across real checkpoints; audio/mixed
requests and broader multi-turn reference validation. Broad frontend parity is not complete.
