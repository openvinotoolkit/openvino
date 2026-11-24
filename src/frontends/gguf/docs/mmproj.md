# Native GGUF multimodal conversion

A llama.cpp multimodal projector file (`general.architecture = clip`, usually named
`mmproj-*.gguf`) converts to an `ov::Model` with vision/audio encoders and projectors, and
optional speech-generation components. The language model is a separate `.gguf` converted by
the regular decoder path.

## Supported projectors

`clip` selects the mmproj coordinator, and a separate projector registry selects each vision or
audio branch. Built-in entries come from `projector_catalog` in
[`mmproj_builder.cpp`](../src/builder/arch/mmproj_builder.cpp) and from the LFM3 builder.

| Modality | Projector | Graph features |
|---|---|---|
| Vision | `gemma3`, `idefics3`, `janus_pro` | SigLIP; fused QKV and legacy FFN names |
| Vision | `mlp` | CLIP feature layers; normalized variant detected from tensors |
| Vision | `internvl` | InternViT, class token, pixel-shuffle merge |
| Vision | `resampler` | MiniCPM-V resampler, `clip.minicpmv_version` 2-6 and 100045 |
| Vision | `gemma3nv` | MobileNet stem, UIR, attention and multiscale feature branches; vision-only support |
| Vision | `minicpmv4_6` | window attention, two-stage merging |
| Vision | `minicpmv4_7` | window-attention and direct 2x2 merging; packed-patch GenAI layout |
| Vision | `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | temporal pairs, M-RoPE, windows (2.5), DeepStack (3) |
| Vision | `pixtral` | 2D RoPE, patch merger, row separators |
| Vision | `phi4` | dynamic resolution, resized positions |
| Vision | `lfm2` | SigLIP, resized positions, pixel unshuffle, optional projector LayerNorm |
| Vision | `lfm3vl` | temporal packed patches and streaming vision |
| Vision | `gemma4v`, `gemma4uv` | two-axis positions, per-tensor clipping, pooling; unified patch normalization |
| Vision | `muse-glimmer` | sparse/global windows, two-axis RoPE, pixel shuffle |
| Vision | `deepseekocr`, `deepseekocr2` | SAM local/global attention, tiles, queries, view separators |
| Audio | `gemma4a` | causal convolution, 12-position relative attention with softcap |
| Audio | `gemma4ua` | raw 640-sample waveform frames |
| Audio | `qwen3a` | Qwen3-ASR audio encoder |
| Audio | `qwen3tts_spkenc` | speaker features |
| Audio | `lfm2a`, `lfm3a` | LFM audio encoders |
| Generated audio | `qwen3tts_gen` | code predictor and streaming codec |
| Generated audio | `lfm3a_gen` | depthformer, feedback and codec components |
| Audio | `qwen2a`, `ultravox`, `voxtral`, `musicflamingo`, `meralion`, `glma` | Whisper-derived encoders |

The built-in `qwen3vl_merger` projector supports BF16 and Q8_0 weights.

InternViT-6B (width 3200, 45 layers) uses RMS norms like the reference, but has no fixture.
The LFM2.5 vocoder is a separate `lfm2.audio_vocoder` architecture.

Not implemented: `ldp`, `ldpv2`, `adapter`, `step3vl`, `llama4`,
`kimivl`, `paddleocr`, `lightonocr`, `cogvlm`, `dots_ocr`, `glm4v`, `youtuvl`, `yasa2`,
`kimik25`, `nemotron_v2_vl`, `exaone4_5`, `hunyuanvl`, `granite_speech`, `mimovl`,
`granite4_vision`. An unsupported projector fails conversion of the whole file.

## Validation coverage

- Every catalog projector has a small numerical fixture checked against the pinned llama.cpp CPU
  encoder, for both raw and GenAI-adapted features. Dynamic fixtures reuse one compiled model
  across several input sizes; see the [fixture README](../tests/test_data/mmproj_accuracy/README.md).
- [Precommit](../../../../tests/model_hub_tests/gguf/gguf_mmproj_precommit) and
  [nightly](../../../../tests/model_hub_tests/gguf/gguf_mmproj_nightly) lists run real
  checkpoints with generated inputs. Per-checkpoint relaxations and skips are recorded in those
  lists.
- Encoder tests do not validate media preprocessing, media insertion or complete VLM/audio
  generation. A catalog entry is not a claim of coverage for every checkpoint, resolution,
  quantization or device.

See [testing](testing.md) for commands and acceptance criteria.

## Files with vision and audio encoders

A projector file holds at most one vision and one audio encoder; Gemma4 E2B/E4B/12B files carry
both. Conversion builds the file's supported encoders into one model:

1. `clip.has_vision_encoder` and `clip.has_audio_encoder` select the encoders. A file with
   neither fails conversion unless the file declares supported speech-generation components.
2. Each encoder takes its projector from the global `clip.projector_type`, or else from
   `clip.vision.projector_type` / `clip.audio.projector_type`. A combined file therefore uses the
   per-modality keys; only the legacy `qwen2.5o` resolves to a different projector per modality
   (Qwen2.5 VL vision, Qwen2 audio). An unsupported projector fails conversion of the whole file,
   except for the combined Gemma3n `gemma3nv` vision / `gemma3na` audio pair. As in the reference,
   this pair builds only vision when no registered extension handles its audio branch.
3. The encoders become disconnected branches of the same graph. Vision reads `v.*` tensors and
   `vision.*` inputs and produces `vision.embeddings`; audio reads `a.*` tensors and `audio.*`
   inputs and produces `audio.embeddings`.
4. `gguf_mmproj` rt_info records each encoder's `<modality>.projector` and `<modality>.merge`.
   The omitted Gemma3n audio branch retains `audio.projector=gemma3na` and
   `audio.supported=false`; both fields survive serialization, and GenAI adaptation skips the
   unsupported audio branch while returning the vision encoder.

Running the combined model requires the inputs of both branches. `AdaptMmprojToGenAI` (below)
turns it into one encoder; apply it to a clone per modality to obtain both.
Declared speech-generation components are built into the same model.

## Graph boundary

The converted graph mirrors the llama.cpp encoder graph:

- Preprocessing belongs to the caller: resize/crop, normalization, mel extraction and frame
  padding follow the reference processor of each model.
- Inputs are named `<modality>.<name>`. Besides pixels or audio features, many encoders take
  index inputs that reorder patches the way the reference does (spatial grouping and windows
  before the encoder, the inverse order after it), positions and additive attention masks.
- Activations are F32 and indices I32. Dynamic encoders accept different spatial sizes;
  fixed-resolution encoders keep the size recorded in metadata.
- Outputs are `vision.embeddings` and/or `audio.embeddings` with shape `[1,1,T,D]`. Projectors
  with auxiliary features (for example DeepStack) pack them after the primary features along `D`.
- Streaming encoders and speech components expose their recurrent state as explicit inputs and
  outputs.

The exact inputs of each projector are defined by its builder.
[`checkpoint_inputs`](../tests/mmproj_fixtures.py) shows how tests construct them; it creates
random pixels/features and is not a media preprocessor.

## Runtime metadata

Model rt_info `gguf_mmproj` keeps every source `clip.*` key and adds derived values such as
`<modality>.projector`, `<modality>.merge`, `vision.patch_size` and `vision.genai_layout`. Values
are strings: numeric arrays are comma-separated, and string arrays use `length:value` entries
flagged by a `<key>.encoding` companion. The metadata survives IR serialization and adaptation.

## Using the encoders and a language model

[`adapt_to_genai()`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) returns every GenAI
encoder of a converted mmproj model by name:

| Name | Model |
|---|---|
| `vision_embeddings` | Vision encoder in the projector's GenAI layout (below), or native pixel inputs when the projector has none |
| `vision_embeddings_pos`, `vision_embeddings_merger` | Additional models of the `split_merger` layout |
| `audio_embeddings` | Audio encoder; Gemma4 encoders take `input_features [1,frames,width]` (plus `input_features_mask [1,frames]` for `gemma4a`, which also returns an all-true `attention_mask`) |
| `audio_generation_<component>` | Independent speech-generation components |

It uses
[`AdaptMmprojToGenAI`](../include/openvino/frontend/gguf/adapt_mmproj_to_genai.hpp), which keeps one
modality of a converted mmproj model and rewrites it in place. To use it directly, run it on a
separate clone for each modality:

| Mode | Result |
|---|---|
| `Layout::EMBEDDINGS` (default) | Native inputs without the modality prefix; `[1,T,D]` outputs `image_features` (plus `deepstack_features.N`) or `audio_features` |
| `Modality::VISION`, `Layout::VISION_ENCODERS` | optimum-intel export layout, so OpenVINO GenAI can reuse its existing vision encoders; `get_vision_models()` returns the models |
| `Modality::AUDIO_GENERATION` | Independent speech components from `get_audio_generation_models()` |

`VISION_ENCODERS` follows the projector's `vision.genai_layout`:

| Layout | Interface |
|---|---|
| `image` | Native pixel inputs are kept |
| `padded_patches` | Raster patches followed by padding, with per-patch (x, y) positions |
| `windowed_patches` | Flattened patches and `image_grid_thw`; window, merge and position indices are derived in the graph |
| `split_merger` | Separate patch-embedding, position-table and merger models |
| `packed_patches` | One packed pixel strip with positions, a patch grid and merge/window indices |
| `temporal_packed_patches` | `[groups,rows,cols,patch features]` pixels; raster positions are derived in the graph and groups are encoded independently |
| `batched_image` | `pixel_values [B,3,H,W]`; each image is encoded independently into `last_hidden_state [B,T,D]` (`mlp`, `internvl`) |

Layout construction runs on a clone and replaces the model only on success, so an invalid
layout leaves the native model available for the `EMBEDDINGS` fallback. A
[ProjectorExtension](extensions.md#extend-mmproj-with-a-projector-component) can declare a
built-in layout or provide its own adapter callback.

`temporal_packed_patches` requires `vision.temporal_patch_size` to match the number of frame
inputs (one or two). The LFM3 builder declares two only when `v.patch_embd.weight.1` exists;
otherwise it declares one. The graph encodes groups sequentially using `TensorIterator`, so
the target device must support that operation (CPU/GPU; NPU is not supported by this layout).
It does not rely on GenAI looping over groups externally.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) in `EMBEDS_TO_LOGITS`
mode prepares the language model for media injection. The token lookup moves to
`get_embedding_model()` and the language model takes `inputs_embeds [B,T,D]`. Depending on the
model, it also takes `token_type_ids` for bidirectional attention within an image, `position_ids
[4,B,T]` for interleaved M-RoPE, or per-layer inputs computed by
`get_per_layer_embedding_model()`.

Adapt the language model in that mode, as described in
[runtime.md](runtime.md#stateful-and-genai-conversion). In C++, with a `FrontEndManager manager`:

```cpp
#include <openvino/frontend/gguf/adapt_to_genai.hpp>

using namespace ov::frontend::gguf;
auto frontend = manager.load_by_framework("gguf");
auto encoders = adapt_to_genai(frontend->convert(frontend->load("mmproj.gguf")));
auto decoder = adapt_to_genai(frontend->convert(frontend->load("language.gguf")),
                              {pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS});
auto language = decoder.at("language");
auto token_lookup = decoder.at("text_embeddings");
```

`text_embeddings_per_layer` is present only for models with that branch. The consumer inserts the encoder
features into `inputs_embeds` at the media token positions and supplies masks, positions and any
per-layer inputs. These passes expose graph contracts; they do not construct tokenizers, media
preprocessing or a complete GenAI pipeline. GenAI uses an F16 KV cache for GGUF multimodal models
to match llama.cpp, unless cache precision is set explicitly.

`gemma3nv` exposes `pixel_values`; `qwen3a` and `qwen3tts_spkenc` expose `features` in the
native audio layout. These adapted graphs require consumer support: GenAI's GGUF VLM integration
does not yet accept these projector pairings, and its existing Qwen3 audio encoder uses a different
input interface.

## Speech generation components

Builders declare speech components in `ModelContract::component_groups["audio_generation"]`,
naming the primary model and each component's outputs and recurrent-state prefix. Vision and
audio groups describe their encoders the same way, so adaptation also works after loading an
unadapted IR. The adapter keeps only the selected component's state and prunes other branches.
For LFM3, `lfm3a_gen` provides `depthformer`, `feedback` and `codec` models; sampling and
waveform reconstruction belong to the caller.
Qwen3TTS `qwen3tts_gen` provides `predictor` and `codec` models. The codec keeps its
`gen_audio.state_in.*` and `gen_audio.state_out.*` tensors as explicit streaming inputs and outputs.

### LFM2 audio components

LFM2.5 Audio uses the `lfm2a` encoder for audio input and the standalone `lfm2.audio_vocoder`
GGUF (depthformer, feedback and codec) for speech output; the publisher's architecture string is
normalized on load. The encoder takes `audio.features` `[1,1,mel_bins,frames]` and
`audio.pos_emb` `[1,1,2*ceil(frames/8)-1,encoder_width]`; its adapter exposes
`audio_features` `[1,tokens,hidden]`. The vocoder exposes `audio.depth_logits`, `audio.feedback`,
`audio.codec_features` and `audio.istft_window`.

The Q4_0 encoder matches represented-F32 references with NMSE `3.81e-10` and `2.35e-9`.
The vocoder is checked against pinned publisher PyTorch fixtures; Q4_0 depthformer and
detokenizer represented-F32 comparisons have NMSE `2.62e-12` and `1.32e-9`. GenAI supplies
waveform preprocessing and speech reconstruction in the `gguf-frontend-mmproj` integration.
Older Mimi-based LFM2 speech output is not supported.

## Known limitations

### Gemma4 image-window parity gap

GenAI-adapted Gemma4 masks follow optimum-intel: tokens of one image attend to the whole image
even beyond the sliding window, while llama.cpp clips older image patches at the window boundary.
Outputs can diverge when an image crosses that boundary; text-only requests are unaffected.
Closing the gap needs an explicit serializable image-window policy in the PagedAttention
operation, not a runtime-metadata switch.

### Bidirectional image prefix-cache gap

GenAI PagedAttention prefix caching can reuse attention states across changed image context, so
multi-image and restarted image chats can differ from SDPA. Disable
`SchedulerConfig.enable_prefix_caching` for affected requests until prefix reuse respects
complete image groups.

### Quantized language models

Close greedy choices can diverge between engines because of different quantized arithmetic.
Q4_K_M conversion has a known accuracy loss relative to llama.cpp, so Q4_K_M differences alone
do not indicate an mmproj integration regression.

## Projector extensions

Use [ProjectorExtension](extensions.md#extend-mmproj-with-a-projector-component) to add or replace one
vision/audio branch under the `clip` coordinator; the [examples](../examples/architecture_extension/README.md)
build and run one. A different whole-model format needs an `ArchitectureExtension`.
