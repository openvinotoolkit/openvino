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

Preprocessing belongs to the caller. Activations are F32 and indices I32; spatial sizes stay
dynamic, so one compiled model serves different image sizes.

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
`<modality>.merge`, `vision.auxiliary_count`, `vision.window_size` (Muse Glimmer) and
`vision.minicpmv_version` / `vision.query_count` (resampler). Values are strings: numeric
arrays are comma-separated, and string arrays are `length:value` entries flagged by a
`<key>.encoding` companion. The metadata survives IR serialization and adaptation; supplied
cgraph decoders return an empty map.

## Using the encoders and a language model

[`AdaptMmprojToGenAI`](../include/openvino/frontend/gguf/adapt_mmproj_to_genai.hpp) keeps one
modality, drops the other branch's inputs and prefixes, and exposes `[1,T,D]` outputs named
`image_features` and `deepstack_features.N`, or `audio_features`. It rewrites the model in place,
so run it on a separate clone for each modality of a combined file.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) in `EMBEDS_TO_LOGITS`
mode prepares the language model for media injection:

- the token lookup moves to `get_embedding_model()`, sharing its weights, and the language model
  takes `inputs_embeds [B,T,D]`; token-embedding scaling is applied once;
- Gemma4 E2B/E4B per-layer token embeddings become a second embedding output and the
  `per_layer_inputs` language input;
- Gemma3 and Gemma4 without per-layer embeddings take `token_type_ids [B,T]`: image tokens attend
  bidirectionally within their image, in every Gemma3 layer and in Gemma4 sliding-window layers;
- interleaved M-RoPE models take `position_ids [4,B,T]`: GenAI's sequence, time, height and width
  sections.

## Projector extensions

Start with the [extension guide](extensions.md#extend-mmproj-with-a-projector-component)
for component examples, registry selection, replacement and shared-library packaging.

`ProjectorExtension` derives from `ArchitectureExtension`, so it uses the same
`frontend.add_extension(...)` and shared-library loading paths. It adds a component
registration to the projector registry, not another whole-model `clip` handler.
Whole-model `ArchitectureExtension` remains available for a different file format or
an implementation that needs control of the complete graph.

A `ProjectorDefinition` contains a unique handler id, the GGUF architecture name,
modality (`vision` or `audio`), resolved projector type, a branch-building callback,
and an optional metadata predicate. The callback receives the coordinator's
`GgufGraphContext` and returns `ProjectorResult`: one output and optional string
metadata under its own modality prefix. It must not call `finish()` or register the
branch's primary output; the coordinator names it `vision.embeddings` or
`audio.embeddings`, builds every declared modality, and finishes the shared graph.
Any extra branch inputs must have unique names.

```cpp
#include <openvino/frontend/gguf/extension/projector.hpp>

using namespace ov::frontend::gguf;

ProjectorDefinition definition{
    "clip.vision.my-projector", "clip", "vision", "my-projector",
    [](GgufGraphContext& graph) {
        // Reuse a built-in encoder/projector topology with compatible metadata and weights.
        return build_builtin_projector(graph, "vision", "gemma3");
    },
    {}};
frontend.add_extension(std::make_shared<ProjectorExtension>(definition));
```

A new topology instead uses `graph.metadata()`, `graph.tensors()`, `graph.add_input()`
and `graph.node()` to construct its branch. It does not need to implement the other
modality, metadata preservation, whole-model selection, or final graph assembly.
`build_builtin_projector` reuses a complete built-in encoder/projector branch;
it does not expose replacement of an individual internal encoder layer.

The coordinator first resolves the global projector type, falling back to the
per-modality key when it is empty. Legacy `qwen2.5o` resolves to the respective vision
and audio types before registry lookup. Selection requires architecture, modality
and resolved type to match, followed by the optional predicate. An unknown type
fails conversion; two matching component handlers are an ambiguity error.

Duplicate handler ids are rejected. To override an existing component, use its id
with `RegistrationMode::Replace`, for example `clip.vision.gemma3`. Other projector
registrations remain active. A replacement id must already exist. Registration is
isolated to the frontend instance. Loaded input models retain the registry snapshot
from `load()`; register projector extensions before loading the file.

The code-level lists are `ArchRegistry::supported_archs()` and
`ArchRegistry::projectors().supported_projectors()`. The latter returns definitions
with ids, architecture names, modalities and types, including extension entries.
For a shared-library plugin, return the `ProjectorExtension` through
`OPENVINO_CREATE_EXTENSIONS`, as with an architecture extension; see the
[plugin instructions](porting_a_llama_cpp_model.md#build-and-load-an-external-plugin).
The buildable [mmproj plugin example](../examples/architecture_extension/mmproj_extension.cpp)
exports a Gemma3 alias and a custom audio projection branch. The example CMake project
builds it as `gguf_mmproj_extension`; its [README](../examples/architecture_extension/README.md)
includes a fixture generator and CPU runner commands.

`clip` is the architecture label of the supported llama.cpp mmproj formats, not a
promise that future formats use it. A projector registration for another architecture
also needs a whole-model coordinator registered for that architecture. Listing
`clip` as supported does not imply support for every projector type.
