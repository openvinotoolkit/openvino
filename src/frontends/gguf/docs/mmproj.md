# Native GGUF multimodal conversion

A llama.cpp multimodal projector file (`general.architecture = clip`, usually named
`mmproj-*.gguf`) converts to an `ov::Model` that runs its vision and/or audio encoder and
projector. The language model is a separate `.gguf` converted by the regular decoder path.

## Supported projectors

Projector support is separate from the architecture list: `clip` selects the mmproj coordinator,
and the per-frontend projector registry selects each vision/audio branch. Built-in entries come
from `projector_catalog` in
[`mmproj_builder.cpp`](../src/builder/arch/mmproj_builder.cpp). Every entry has a small
numerical fixture checked against the llama.cpp CPU encoder (see
[the fixture README](../tests/test_data/mmproj_accuracy/README.md)).

| Modality | Projector | Graph features |
|---|---|---|
| Vision | `gemma3`, `idefics3`, `janus_pro` | SigLIP; fused QKV and legacy FFN names |
| Vision | `mlp` | CLIP feature layers; normalized variant detected from tensors |
| Vision | `internvl` | InternViT, class token, pixel-shuffle merge |
| Vision | `resampler` | MiniCPM-V resampler, `clip.minicpmv_version` 2-6 and 100045 |
| Vision | `minicpmv4_6` | window attention, two-stage merging |
| Vision | `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | temporal pairs, M-RoPE, windows (2.5), DeepStack (3) |
| Vision | `pixtral` | 2D RoPE, patch merger, row separators |
| Vision | `phi4` | dynamic resolution, resized positions |
| Vision | `gemma4v`, `gemma4uv` | two-axis positions, per-tensor clipping, pooling; unified patch normalization |
| Vision | `muse-glimmer` | sparse/global windows, two-axis RoPE, pixel shuffle |
| Vision | `deepseekocr`, `deepseekocr2` | SAM local/global attention, tiles, queries, view separators |
| Audio | `gemma4a` | causal convolution, 12-position relative attention with softcap |
| Audio | `gemma4ua` | raw 640-sample waveform frames |
| Audio | `qwen2a`, `ultravox`, `voxtral`, `musicflamingo`, `meralion`, `glma` | Whisper-derived encoders |

InternViT-6B (width 3200, 45 layers) uses RMS norms like the reference, but has no fixture.

Not implemented: `ldp`, `ldpv2`, `adapter`, `step3vl`, `gemma3nv`, `llama4`, `qwen3a`, `lfm2`,
`kimivl`, `paddleocr`, `lightonocr`, `cogvlm`, `dots_ocr`, `lfm2a`, `glm4v`, `youtuvl`, `yasa2`,
`kimik25`, `nemotron_v2_vl`, `exaone4_5`, `hunyuanvl`, `granite_speech`, `mimovl`,
`granite4_vision`. `gemma3na` has no graph builder in the reference.

## Validation coverage

| Evidence | Scope and limits |
|---|---|
| Offline numerical fixtures | Every catalog projector has a small CPU encoder reference; tests compare raw and GenAI-adapted features |
| Dynamic fixtures | Multiple sizes reuse one compiled model; variants are listed in the [fixture README](../tests/test_data/mmproj_accuracy/README.md) |
| Real-checkpoint lists | [Precommit](../../../../tests/model_hub_tests/gguf/gguf_mmproj_precommit) and [nightly](../../../../tests/model_hub_tests/gguf/gguf_mmproj_nightly) select checkpoints, including combined Gemma4 and Qwen2.5-Omni files |
| Qwen2-VL real checkpoint | CPU NMSE limit is explicitly relaxed to `2e-4`; the list records reference patch-embedding arithmetic as the reason |
| DeepSeek-OCR real checkpoint | Skipped: model-hub input preparation does not yet implement SAM tiles/views; offline OCR fixtures remain covered |
| Full media application | Encoder fixtures and generated-input checkpoint tests do not validate host preprocessing, media insertion, or complete VLM/audio generation |

See [testing](testing.md) for commands and acceptance. A catalog entry is not a claim of coverage
for every checkpoint, resolution, quantization, or device.

## Files with vision and audio encoders

A projector file holds at most one vision and one audio encoder; Gemma4 E2B/E4B/12B files carry
both. Conversion builds every encoder the file declares into one model:

1. `clip.has_vision_encoder` and `clip.has_audio_encoder` select the encoders. A file with
   neither fails conversion.
2. Each encoder takes its projector from the global `clip.projector_type`, or else from
   `clip.vision.projector_type` / `clip.audio.projector_type`. A combined file therefore uses the
   per-modality keys; only the legacy `qwen2.5o` resolves to a different projector per modality
   (Qwen2.5 VL vision, Qwen2 audio). An unsupported projector fails conversion of the whole file,
   so no declared modality is dropped silently.
3. The encoders become disconnected branches of the same graph. Vision reads `v.*` tensors and
   `vision.*` inputs and produces `vision.embeddings`; audio reads `a.*` tensors and `audio.*`
   inputs and produces `audio.embeddings`.
4. `gguf_mmproj` rt_info records each encoder's `<modality>.projector` and `<modality>.merge`.

Running the combined model requires the inputs of both branches. `AdaptMmprojToGenAI` (below)
turns it into one encoder; apply it to a clone per modality to obtain both.

## Graph boundary

Preprocessing belongs to the caller. Activations are F32 and indices I32, except the
explicit F32 positional inputs below. Dynamic encoders accept different spatial sizes;
fixed-resolution encoders retain the size recorded in metadata.

| Family | Inputs |
|---|---|
| Fixed-resolution vision | `vision.pixel_values [1,3,S,S]`, normalized NCHW |
| MiniCPM resampler | `vision.pixel_values [1,3,H,W]`, learned `vision.position_ids [1,1,1,T]`, F32 `vision.position_h` / `vision.position_w [1,1,T,1]` |
| Qwen vision | `vision.pixel_values [2,3,H,W]` (a temporal pair; repeat a still image), `vision.patch_indices [1,1,1,T]`, `vision.position_ids [1,1,1,4*T]` |
| Qwen2.5 vision | also `vision.attention_mask [1,1,T,T]` (additive window mask) and `vision.output_indices [1,1,1,T/4]` |
| Pixtral / Phi4 / Gemma4 vision | `vision.pixel_values [1,3,H,W]`; Pixtral and Gemma4 also `vision.position_x`, `vision.position_y [1,1,1,T]` |
| Muse Glimmer | pixels; `patch_indices` in window order with each window padded to `window_size²` slots; one-based `position_x` / `position_y` per slot; `output_indices`; pixel-shuffle `merge_indices`; additive key mask `window_mask [W,1,1,window_size²]` that hides the padding |
| MiniCPM-V 4.6 | pixels, position IDs, `window_indices`, `inverse_window_indices`, `attention_mask`, `vit_merger.indices.N`, `merger.indices.N` |
| DeepSeek-OCR / OCR2 | `vision.pixel_values [B,3,H,W]` (independent square tiles), local/global relative-position indices, `vision.output_indices`; OCR1 `position_indices`, OCR2 `query_indices`, `query_output_indices`, `position_ids`, `attention_mask` |
| Gemma4 audio | `audio.features [1,1,mel,frames]` |
| Gemma4 unified audio | `audio.waveform_frames [1,1,T,640]`, 16 kHz waveform in 640-sample frames |
| Whisper-derived audio | `audio.features [1,mel,1,frames]`, `audio.position_ids [1,1,1,ceil(frames/2)]` |

Index inputs reorder patches the way the reference does: spatial grouping and windows before
the encoder, the inverse order after it. Qwen sizes must be divisible by patch size times merge
size. Gemma4 vision takes pixels in [0,1] and applies the reference's `2*x-1` inside the graph;
unified vision uses patch size times projector scale factor. Gemma4 audio builds its
12-position causal horizon and relative positions in the graph as `[T,T]` tensors, so memory
grows quadratically with the number of frames. Missing/zero `clip.minicpmv_version` selects 2; missing/zero `clip.minicpmv_query_num`
selects 96 queries for version 2 and 64 otherwise, and an explicit count must match the query
tensor.

Outputs are `vision.embeddings` and/or `audio.embeddings`, `[1,1,T,D]`. `qwen3vl_merger` packs
the DeepStack features after the primary ones along `D`.

## Runtime metadata

Model rt_info `gguf_mmproj` keeps every source `clip.*` key and adds `<modality>.projector`,
`<modality>.merge`, `vision.auxiliary_count`, the effective `vision.patch_size`,
`vision.window_size` (Muse Glimmer) and
`vision.minicpmv_version` / `vision.query_count` (resampler). Values are strings: numeric
arrays are comma-separated, and string arrays are `length:value` entries flagged by a
`<key>.encoding` companion. The metadata survives IR serialization and adaptation; supplied
cgraph decoders return an empty map.

## Preprocessing real media

`GGUFMMProjAccuracy` and `GGUFMMProjDynamicAccuracy` run every projector offline from committed
inputs and llama.cpp outputs; see [testing.md](testing.md).

For real files, [model-hub commands](testing.md#real-checkpoints) reproduce encoder comparisons
with generated inputs. [`checkpoint_inputs`](../tests/mmproj_fixtures.py) illustrates supported
index construction; it creates random pixels/features and is not a media preprocessor.
For real Gemma3 images, reproduce the reference resize/crop and RGB channel normalization from
`clip.vision.image_mean` / `clip.vision.image_std` before supplying NCHW pixels. Whisper-derived
audio needs the reference mel extraction and frame padding before `audio.features`; Gemma4 unified
audio instead takes consecutive 640-sample frames from a 16 kHz waveform, without mel extraction.

## Using the encoders and a language model

[`AdaptMmprojToGenAI`](../include/openvino/frontend/gguf/adapt_mmproj_to_genai.hpp) keeps one
modality, drops the other branch's inputs and prefixes, and exposes `[1,T,D]` outputs named
`image_features` and `deepstack_features.N`, or `audio_features`. It rewrites the model in place,
so run it on a separate clone for each modality of a combined file.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) in `EMBEDS_TO_LOGITS`
mode prepares the language model for media injection:

- the token lookup moves to `get_embedding_model()`, sharing its weights, and the language model
  takes `inputs_embeds [B,T,D]`; token-embedding scaling is applied once;
- Gemma4 E2B/E4B use a separate `get_per_layer_embedding_model()` whose output feeds
  `per_layer_inputs [B,T,layers,width]`; it is not a second output of `get_embedding_model()`;
- Gemma3 and Gemma4 outside the E2B/E4B embedding widths take `token_type_ids [B,T]`: image tokens attend
  bidirectionally within their image, in every Gemma3 layer and in Gemma4 sliding-window layers;
- interleaved M-RoPE models take `position_ids [4,B,T]`: GenAI's sequence, time, height and width
  sections;
- the per-layer lookup reads image, video and audio placeholders as the padding token, as HF and
  llama.cpp do;
- Gemma4 image attention remains causal at the E2B/E4B embedding widths (1536/2560), matching
  llama.cpp; other variants allow bidirectional attention within an image in sliding-window layers.

[`genai_vision_models`](../include/openvino/frontend/gguf/genai_vision.hpp) returns the vision
encoder with inputs compatible with the optimum-intel export layout, allowing OpenVINO GenAI
to reuse its existing encoders. GGUF preprocessing uses llama.cpp geometry and token limits:

| Projector | Models | Inputs |
|---|---|---|
| `gemma3` | `vision_embeddings` | `pixel_values [1,3,S,S]` |
| `gemma4v`, `gemma4uv` | `vision_embeddings` | `pixel_values [1,P,patch*patch*3]` with patches in raster order, then padding; `image_position_ids [1,P,2]` as (x, y), -1 for padding |
| `muse-glimmer` | `vision_embeddings` | `pixel_values [rows*cols,3*patch*patch]`, `image_grid_thw [1,3]`; window, merge and position indices are derived in the graph |
| `qwen3vl_merger` | `vision_embeddings`, `vision_embeddings_pos`, `vision_embeddings_merger` | flattened patches `hidden_states`; position-table indices `input [4,N]`; `hidden_states`, `attention_mask [1,N,N]`, `rotary_pos_emb [N,head/2]` |

The adapted graphs preserve llama.cpp computation: Gemma4 retains GELU_QUICK unless the
GGUF specifies GELU. Muse Glimmer GGUF files collapse HF's two-frame patch kernel,
so the layout takes one frame per patch.

GenAI selects these settings from the GGUF metadata while keeping the existing exported-model
processor defaults. GGUF video inputs keep all supplied frames unless the caller provides
sampled frame indices. Numerical acceptance uses the pinned llama.cpp CPU reference; an
optimum-intel comparison checks compatibility and does not replace that reference.

GenAI uses an F16 KV cache for GGUF multimodal models to match llama.cpp's default cache
precision; explicit cache precision properties take precedence.

Gemma4 sliding-window layers apply the window to image tokens as well as text tokens.
The GenAI adaptation preserves this behavior through paged attention conversion.

Use language checkpoints containing Q4_0 weights for quantized generation accuracy tests;
inspect their tensor types because files named Q4_0 can also contain Q5_K/Q6_K weights.
Q4_0 can still diverge on close greedy choices because the CPU engines use different
quantized arithmetic. Keep the original llama.cpp comparison and its accuracy thresholds;
an exact F16 expansion can help distinguish arithmetic differences from integration errors.
Q4_K_M conversion currently has an expected accuracy loss relative to llama.cpp; it retains the existing
conversion until the plugin-side issue is resolved. Q4_K_M differences therefore do not
establish an mmproj integration regression.

Register `GenAIExtension` before converting the language model, as described in
[runtime.md](runtime.md#stateful-and-genai-conversion). In C++, with a `FrontEndManager manager`
and an already converted combined `mmproj` model:

```cpp
#include <openvino/frontend/gguf/adapt_mmproj_to_genai.hpp>
#include <openvino/frontend/gguf/extension/genai.hpp>

using namespace ov::frontend::gguf::pass;
auto vision = mmproj->clone();
auto audio = mmproj->clone();
AdaptMmprojToGenAI(AdaptMmprojToGenAI::Modality::VISION).run_on_model(vision);
AdaptMmprojToGenAI(AdaptMmprojToGenAI::Modality::AUDIO).run_on_model(audio);
auto frontend = manager.load_by_framework("gguf");
auto genai = std::make_shared<ov::frontend::gguf::GenAIExtension>(
    ov::frontend::gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
frontend->add_extension(genai);
auto language = frontend->convert(frontend->load("language.gguf"));
auto token_lookup = genai->get_embedding_model();
auto per_layer_lookup = genai->get_per_layer_embedding_model();
```

`per_layer_lookup` is null for models without that branch. Compile the selected encoders and
lookup models; the consumer inserts their raw features into `inputs_embeds` at the model's media
token positions and supplies masks, positions, and any per-layer inputs. Keep model-specific
placeholder handling and DeepStack delivery in the consumer. These passes expose graph contracts;
they do not construct tokenizers, media preprocessing, or a complete GenAI pipeline.

## Projector extensions

Use [ProjectorExtension](extensions.md#extend-mmproj-with-a-projector-component) to add or replace one
vision/audio branch under the `clip` coordinator; the [examples](../examples/architecture_extension/README.md)
build and run one. A different whole-model format needs an `ArchitectureExtension`.
