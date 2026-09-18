# Native GGUF multimodal conversion (experimental)

Reference: llama.cpp `16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb`.
The reference is used only by offline fixture generators and validation executables.
It is not a production dependency.

This implementation is partial. Conversion, encoder accuracy and GenAI generation
are separate qualification levels. A catalog entry does not certify every checkpoint,
preprocessing variant or associated language backbone.

## Implemented conversion

| Modality | Projectors | F32 CPU oracle | Real-checkpoint encoder | GenAI adapter |
|---|---|---|---|---|
| Vision | `gemma3` | Pass, separate/fused QKV and legacy FFN names | Pass, Gemma3 4B F16 projector | Gemma3, experimental |
| Vision | `mlp` (including normalized variant) | Pass | Pending | Pending |
| Vision | `idefics3`, `janus_pro`, `internvl` | Pass | Pending | Pending |
| Vision | `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | Pass, rectangular grids, temporal merging, window ordering and DeepStack | Pending | Pending |
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
| Qwen vision | `vision.pixel_values [2,3,H,W]`, temporal pair; repeat a still image twice |
| Qwen vision | `vision.patch_indices [1,1,1,T]`, patch ordering; `vision.position_ids [1,1,1,4*T]`, four position sections |
| Qwen2.5 vision | Also `vision.attention_mask [1,1,T,T]`, additive window mask, and `vision.output_indices [1,1,1,T/4]`, inverse window order |
| Audio | `audio.features [1,mel,1,frames]`, plus `audio.position_ids [1,1,1,ceil(frames/2)]` |

Qwen spatial dimensions must be divisible by patch size times merge size. Patch
indices apply spatial grouping and window permutation before the encoder; positions
must follow that order. Output indices restore the reference spatial ordering.
Audio feature extraction, chunking, valid lengths and token placement remain the
caller's responsibility; the learned position table bounds supported lengths.

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

The companion GenAI change accepts a language GGUF and projector GGUF in memory:

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

## Remaining reference coverage

The following reference projector names still require builders, numerical fixtures
and checkpoint validation:

`ldp`, `ldpv2`, `resampler`, `adapter`, `step3vl`, `gemma3nv`, `gemma4v`, `gemma4a`, `gemma4uv`, `gemma4ua`, `phi4`, `pixtral`, `llama4`, `qwen3a`, `lfm2`, `kimivl`, `paddleocr`, `lightonocr`, `cogvlm`, `dots_ocr`, `deepseekocr`, `deepseekocr2`, `lfm2a`, `glm4v`, `youtuvl`, `yasa2`, `kimik25`, `nemotron_v2_vl`, `exaone4_5`, `hunyuanvl`, `minicpmv4_6`, `granite_speech`, `mimovl`, `granite4_vision`.

`gemma3na` is a named enum without a graph-builder dispatch in the pinned reference
and is an explicit exclusion. `mlp_norm` is an internal tensor-detected variant of
`mlp`, not a separately accepted projector metadata string.

Remaining delivery work includes MiniCPM version dispatch, Gemma4 vision/audio,
OCR/resampler/convolutional and Conformer topologies; shared-family GenAI adapters;
preprocessing and embedding/logit equivalence across real checkpoints; audio/mixed
requests and broader multi-turn reference validation. Broad frontend parity is not complete.
