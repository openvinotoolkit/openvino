# Native GGUF multimodal conversion (experimental)

A llama.cpp multimodal projector file (`general.architecture = clip`, usually named
`mmproj-*.gguf`) converts to an `ov::Model` that runs its vision and/or audio encoder and
projector. The language model is a separate `.gguf` converted by the regular decoder path.
Conversion, encoder accuracy and end-to-end generation are separate qualification levels:
a supported projector does not certify every checkpoint, preprocessing variant or language
pairing.

## Supported projectors

The catalog is `projector_catalog` in
[`mmproj_builder.cpp`](../src/builder/arch/mmproj_builder.cpp). Every entry has a small
numerical fixture checked against the llama.cpp CPU encoder (see
[the fixture README](../tests/test_data/mmproj_accuracy/README.md)). A real-checkpoint
encoder check means a downloaded projector matched llama.cpp at normalized MSE below `1e-5`.

| Modality | Projector | Graph features | Real-checkpoint encoder check |
|---|---|---|---|
| Vision | `gemma3`, `idefics3`, `janus_pro` | SigLIP; fused QKV and legacy FFN names | `gemma3`: gemma-3-4b F16 |
| Vision | `mlp` | CLIP feature layers; normalized variant detected from tensors | — |
| Vision | `internvl` | InternViT, class token, pixel-shuffle merge | — |
| Vision | `resampler` | MiniCPM-V resampler, `clip.minicpmv_version` 2-6 and 100045 | — |
| Vision | `minicpmv4_6` | window attention, two-stage merging | — |
| Vision | `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | temporal pairs, M-RoPE, windows (2.5), DeepStack (3) | `qwen3vl_merger`: Qwen3.5 0.8B/2B/4B/9B, Qwen3.6 35B, Qwen3.8 27B |
| Vision | `pixtral` | 2D RoPE, patch merger, row separators | Ministral-3 3B Q8_0 |
| Vision | `phi4` | dynamic resolution, resized positions | — |
| Vision | `gemma4v`, `gemma4uv` | two-axis positions, per-tensor clipping, pooling; unified patch normalization | E2B, E4B, 26B, 31B (`gemma4v`); 12B (`gemma4uv`) |
| Vision | `muse-glimmer` | sparse/global windows, two-axis RoPE, pixel shuffle | Muse Glimmer 30B Q4_K |
| Vision | `deepseekocr`, `deepseekocr2` | SAM local/global attention, tiles, queries, view separators | — |
| Audio | `gemma4a` | causal convolution, 12-position relative attention with softcap | E2B, E4B |
| Audio | `gemma4ua` | raw 640-sample waveform frames | 12B |
| Audio | `qwen2a`, `ultravox`, `voxtral`, `musicflamingo`, `meralion`, `glma` | Whisper-derived encoders | — |

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

Embedding mode is validated with one sequence, with SDPA and PagedAttention. The GenAI branch
`gguf-frontend-mmproj` uses both passes in `VLMPipeline(language.gguf, device,
mmproj_path=mmproj.gguf)` for Gemma3, Qwen3.5/3.6 with `qwen3vl_merger`, Gemma4 vision/audio and
Muse Glimmer.

## Validation status

Last full run: 2026-10-01, CPU, llama.cpp `03fa73cb27f5c251b9528489b18d303b1366aca4` as the
oracle. Encoder checks use the represented-weight F32 projector as reference. GenAI checks
compare greedy choices on identical histories with the original publisher BF16 weights; they
pass with the first token and at least 90% of choices (13 for language, up to 20 per medium).
They use F32 inference, F16 KV cache, no dynamic activation quantization and
`OV_GGUF_Q4_K_ZP_F16=1`.

- **Encoders:** all real-checkpoint checks in the table above pass, at normalized MSE between
  `2e-13` and `5.4e-6`.
- **Language (GenAI `LLMPipeline`, plus chat, reset, batch-of-two, beam and streaming checks):**
  Qwen3.5-4B Q4_0 (12/13) and Q4_K_M (13/13) with PA and SDPA, Qwen3.6-35B Q4_K_M PA (12/13),
  Gemma4 E4B Q4_K_M PA and Muse Glimmer Q4_K_M PA (13/13) pass. Gemma4 12B Q4_0 (10/13) and
  Gemma4 26B Q4_0 (7/13) miss the first token. llama.cpp replaying the same reference history on
  the same Q4_0 files scores the same 10/13 and 7/13 and also misses the first token; OV matches
  its choices at 12/13 and 11/13 steps, differing only where its top two are within 0.21-0.51
  logits.
- **Multimodal (GenAI `VLMPipeline`):** Qwen3.5-4B passes text, image and video for Q4_K_M with
  PA and SDPA, and for Q4_0 with PA; Gemma4 E2B Q4_K_M passes text, image, video, audio and mixed input with PA.
  In Gemma4 E2B chat, the audio follow-up turn misses only the first token (19/20); llama.cpp
  running the same Q4_K_M file on that history makes the same choice, so quantization, not
  history handling, causes the difference.
  Gemma4 12B Q4_0 PA fails mixed input (17/20, first token differs) and video (17/20) against
  the publisher reference. Against llama.cpp with the same Q4_0 files it passes all five inputs:
  20/20 except mixed (18/20), where OV takes llama.cpp's runner-up at margins of 0.28 and 0.53
  logits. Muse Glimmer Q4_K_M passes against the same quantized file in llama.cpp; no publisher
  reference is available for it.

The Gemma4 26B difference starts at layer 8, where two experts swap places at the top-8 cutoff
for a near tie. Checkpoint matrices for Qwen3.5 0.8B-9B, Qwen3.6, Qwen3.8 and all Gemma4 sizes
are summarized in the
[architecture fixture README](../tests/test_data/arch_accuracy/README.md#q8_0_c-requantization-accuracy-check).
End-to-end GenAI generation, including long contexts, also runs on an Intel Arc A770 GPU (16 GB)
for the models that fit its memory. NPU and concurrent requests are not qualified.

The model hub tests in [`tests/model_hub_tests/gguf`](../../../../tests/model_hub_tests/gguf)
also convert downloaded projector files for `idefics3`, `internvl`, `pixtral`, the Qwen VL mergers,
`gemma3`, Gemma4 vision and audio, `voxtral`, `ultravox` and Qwen2.5 Omni, and compare every
encoder with llama.cpp built at the pinned revision during the test session.

The GenAI checks are run with GenAI's `tests/python_tests/validate_gguf_llm.py` (language) and
`tests/python_tests/validate_gguf_mmproj.py` (media). Keep their reports outside the source tree.
