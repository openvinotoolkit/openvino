# Native GGUF multimodal conversion

A llama.cpp multimodal projector file (`general.architecture = clip`, usually named
`mmproj-*.gguf`) converts to an `ov::Model` with vision/audio encoders and projectors, and
optional speech-generation components. The language model is a separate `.gguf` converted by
the regular decoder path.

## Supported projectors

`clip` selects the mmproj coordinator, and a separate projector registry selects each vision or
audio branch. Built-in entries come from `projector_catalog` in
[`mmproj_builder.cpp`](../src/builder/arch/mmproj_builder.cpp) and from the LFM3 builder.

| Modality | Projectors |
|---|---|
| Vision | `gemma3`, `idefics3`, `janus_pro`, `mlp`, `internvl`, `resampler`, `minicpmv4_6`, `minicpmv4_7`, `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger`, `pixtral`, `phi4`, `lfm2`, `lfm3vl`, `gemma4v`, `gemma4uv`, `muse-glimmer`, `deepseekocr`, `deepseekocr2` |
| Audio | `gemma4a`, `gemma4ua`, `lfm2a`, `lfm3a`, `qwen2a`, `ultravox`, `voxtral`, `musicflamingo`, `meralion`, `glma` |
| Speech generation | `lfm3a_gen`; the LFM2.5 vocoder is a separate `lfm2.audio_vocoder` architecture |

Not implemented: `ldp`, `ldpv2`, `adapter`, `step3vl`, `gemma3nv`, `llama4`, `qwen3a`,
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

One file may carry at most one vision and one audio encoder (for example Gemma4 and Qwen2.5-Omni).
`clip.has_vision_encoder` / `clip.has_audio_encoder` select the encoders, and each takes its
projector from `clip.projector_type` or the per-modality `clip.<modality>.projector_type`. The
encoders become disconnected branches of one model: vision reads `v.*` tensors and `vision.*`
inputs and produces `vision.embeddings`; audio uses `a.*`, `audio.*` and `audio.embeddings`.
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

[`AdaptMmprojToGenAI`](../include/openvino/frontend/gguf/adapt_mmproj_to_genai.hpp) keeps one
modality of a converted mmproj model and rewrites it in place, so run it on a separate clone for
each modality:

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

Register `GenAIExtension` before converting the language model, as described in
[runtime.md](runtime.md#stateful-and-genai-conversion). In C++, with a `FrontEndManager manager`
and an already converted combined `mmproj` model:

```cpp
#include <openvino/frontend/gguf/adapt_mmproj_to_genai.hpp>
#include <openvino/frontend/gguf/extension/genai.hpp>

using namespace ov::frontend::gguf::pass;
auto vision = mmproj->clone();
auto audio = mmproj->clone();
AdaptMmprojToGenAI vision_adapter(AdaptMmprojToGenAI::Modality::VISION,
                                 AdaptMmprojToGenAI::Layout::VISION_ENCODERS);
vision_adapter.run_on_model(vision);
auto vision_models = vision_adapter.get_vision_models();
AdaptMmprojToGenAI(AdaptMmprojToGenAI::Modality::AUDIO).run_on_model(audio);
auto frontend = manager.load_by_framework("gguf");
auto genai = std::make_shared<ov::frontend::gguf::GenAIExtension>(
    ov::frontend::gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
frontend->add_extension(genai);
auto language = frontend->convert(frontend->load("language.gguf"));
auto token_lookup = genai->get_embedding_model();
auto per_layer_lookup = genai->get_per_layer_embedding_model();
```

`per_layer_lookup` is null for models without that branch. The consumer inserts the encoder
features into `inputs_embeds` at the media token positions and supplies masks, positions and any
per-layer inputs. These passes expose graph contracts; they do not construct tokenizers, media
preprocessing or a complete GenAI pipeline. GenAI uses an F16 KV cache for GGUF multimodal models
to match llama.cpp, unless cache precision is set explicitly.

## Speech generation components

Builders declare speech components in `ModelContract::component_groups["audio_generation"]`,
naming the primary model and each component's outputs and recurrent-state prefix. Vision and
audio groups describe their encoders the same way, so adaptation also works after loading an
unadapted IR. The adapter keeps only the selected component's state and prunes other branches.
For LFM3, `lfm3a_gen` provides `depthformer`, `feedback` and `codec` models; sampling and
waveform reconstruction belong to the caller.

### LFM2 audio components

LFM2.5 Audio uses the `lfm2a` encoder for audio input and the standalone `lfm2.audio_vocoder`
GGUF (depthformer, feedback and codec) for speech output; the publisher's architecture string is
normalized on load. The vocoder is checked against pinned publisher PyTorch fixtures. GenAI
supplies waveform preprocessing and speech reconstruction in the `gguf-frontend-mmproj`
integration. Older Mimi-based LFM2 speech output is not supported.

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

### Legacy export layouts

- Older MiniCPM-V `resampler` and `minicpmv4_6` exports store the FFN up and down weights under
  swapped names. As in llama.cpp, the swap is detected from tensor shapes and undone.
- Published MERaLiON-2 projectors store a per-frame `mm.a.norm_pre` (encoder width). llama.cpp,
  and therefore this frontend, normalizes the stacked frames as one vector and repeats the
  per-frame weight and bias across them. Hugging Face MERaLiON-2 normalizes each frame before
  stacking, so features differ from the original model, not from llama.cpp.

## Projector extensions

Use [ProjectorExtension](extensions.md#extend-mmproj-with-a-projector-component) to add or
replace one vision/audio branch under the `clip` coordinator; the
[examples](../examples/architecture_extension/README.md) build and run one. A different
whole-model format needs an `ArchitectureExtension`.
