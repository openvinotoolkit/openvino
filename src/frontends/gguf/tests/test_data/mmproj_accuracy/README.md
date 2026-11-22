# Multimodal projector accuracy references

Each NPZ holds a small random F32 `clip` GGUF (`model`, uint8 bytes), the encoder inputs and
the embeddings that the real llama.cpp CPU encoder produces for them. OpenVINO does not
participate in reference generation. The regular frontend test run consumes these files; no
llama.cpp build or model download is needed unless they are regenerated.

`gen_mmproj_accuracy.py` generates both suites:

| Suite | Layout | Families |
|---|---|---|
| `GGUFMMProjAccuracy` | `inputs`, `embeddings` and family-specific index inputs | `gemma3` (+`_fused`, `_legacy`), `idefics3`, `janus_pro`, `mlp` (+`_norm`, `_feature`), `internvl` (+`_qknorm`), `resampler` (+`_v2`, `_v4`), `qwen2vl_merger`, `qwen2.5vl_merger` (+`_window_video`), `qwen3vl_merger`, `qwen2a`, `ultravox`, `voxtral` (+`_odd`), `musicflamingo`, `meralion`, `glma` |
| `GGUFMMProjDynamicAccuracy` | two input sizes as `0.*` and `1.*`; one compiled model must serve both | `muse-glimmer`, `pixtral` (+`_merge`), `phi4`, `gemma4v` (+`_one_sided`), `gemma4uv` (+`_low_contrast`), `gemma4ua`, `gemma4a`, `minicpmv4_6`, `deepseekocr` (+`_resize`, `_overview`), `deepseekocr2` (+`_overview`); `qwen2.5vl_merger_grids` and `resampler_grids` rerun single-grid fixture models |

Suffixes select fixture variants, for example fused QKV, legacy FFN names, concatenated CLIP
feature layers, whole-tensor QK norms, an odd frame count, one-sided clipping bounds or OCR
overview separators. Both suites compare the raw encoder and its `AdaptMmprojToGenAI` form
with normalized MSE below `1e-5`.

The standalone op tests in `test_ops.cpp` read `../mmproj_*.npy`, `../vision_rope_expected.npy`
and `../multimodal_imrope_expected.npy`: window partition, SAM relative positions,
vision/interleaved RoPE, antialiased bilinear resize and 2D im2col.

## Reference revision

Base fixtures and op expectations come from llama.cpp
[`03fa73cb27f5c251b9528489b18d303b1366aca4`](https://github.com/ggml-org/llama.cpp/commit/03fa73cb27f5c251b9528489b18d303b1366aca4),
which has aligned-corner position interpolation, the Muse Glimmer encoder, and a `ggml_clamp`
that no longer clamps its source in place. The Gemma4 fixtures clip only the Q input, so
clipping that leaks into the shared K/V input fails them.

## Regenerating

Build the oracles against a CPU-only llama.cpp checkout (`GGML_OPENVINO=OFF`), with `LLAMA_SRC`
and `LLAMA_BUILD` pointing to the checkout and its build directory. Run from
`src/frontends/gguf/tests`:

```sh
c++ -std=c++17 mmproj_oracle.cpp -I "$LLAMA_SRC/tools/mtmd" -I "$LLAMA_SRC/include" \
    -I "$LLAMA_SRC/ggml/include" -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lmtmd -lllama -lggml -lggml-base -o mmproj_oracle
c++ -std=c++17 mmproj_ops_oracle.cpp -I "$LLAMA_SRC/ggml/include" -L "$LLAMA_BUILD/bin" \
    -Wl,-rpath,"$LLAMA_BUILD/bin" -lggml -lggml-base -lggml-cpu -o mmproj_ops_oracle
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_mmproj_accuracy.py --oracle ./mmproj_oracle \
    --ops-oracle ./mmproj_ops_oracle
```

`--families` regenerates a subset. `mmproj_oracle` takes
`model vision|audio width height input.f32 output.f32 [second_frame.f32]`;
`GGUF_ORACLE_DUMP=<directory>` writes its intermediate F32 tensors, and
`GGUF_ORACLE_OVERVIEW=1` adds the OCR view separator. `mmproj_fixtures.py` holds the helpers
shared by the generator and the model hub tests.

## Real checkpoints

`tests/model_hub_tests/gguf/test_gguf_mmproj.py` compares downloaded projector files with
`mmproj_oracle`, which it builds at the revision above, on an F32 copy of the same weights.
See [testing](../../../docs/testing.md#real-checkpoints) for pytest commands, dependencies,
oracle caches, and local-file lists, and [coverage limits](../../../docs/mmproj.md#validation-coverage)
for checkpoint tolerance overrides and skipped cases.

`lfm2` and `lfm2_no_norm` use the same pinned oracle revision above and the two-grid
generator seed 2718. They cover SigLIP, resized positions, spatial unshuffle and the
optional input LayerNorm at 16x8 and 8x24 pixels, through both native and embedding-adapted IO.

`lfm2a` uses the same pinned llama.cpp CPU oracle, with 49 and 101 feature frames.
It exercises depthwise subsampling, relative attention shift and temporal convolution.
The dynamic test returns to the first shape to check request reuse.

`minicpmv4_7` and `minicpmv4_7_4x` are extracted from PR #39000. The pinned
llama.cpp CPU references at `098ce526b546b5a190327c1da82929cee051b9ac`
cover the window-attention and direct 2x2 merger paths,
with unequal image grids and packed-patch GenAI input adaptation.
