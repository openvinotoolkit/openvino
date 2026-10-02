# Multimodal projector accuracy references

Each NPZ holds a small random F32 `clip` GGUF (`model`, uint8 bytes), the encoder inputs and
the embeddings that the real llama.cpp CPU encoder produces for them. OpenVINO does not
participate in reference generation. The regular frontend test run consumes these files; no
llama.cpp build or model download is needed unless they are regenerated.

| Suite | Generator | Layout | Families |
|---|---|---|---|
| `GGUFMMProjAccuracy` | `gen_mmproj_accuracy.py` | `inputs`, `embeddings` and family-specific index inputs | `gemma3` (+`_fused`, `_legacy`), `idefics3`, `janus_pro`, `mlp` (+`_norm`, `_feature`), `internvl` (+`_qknorm`), `resampler` (+`_v2`, `_v4`), `qwen2vl_merger`, `qwen2.5vl_merger` (+`_window_video`), `qwen3vl_merger`, `qwen2a`, `ultravox`, `voxtral` (+`_odd`), `musicflamingo`, `meralion`, `glma` |
| `GGUFMMProjDynamicAccuracy` | `gen_mmproj_dynamic_accuracy.py` | two input sizes as `0.*` and `1.*`; one compiled model must serve both | `muse-glimmer`, `pixtral` (+`_merge`), `phi4`, `gemma4v` (+`_one_sided`), `gemma4uv` (+`_low_contrast`), `gemma4ua`, `gemma4a`, `minicpmv4_6`, `deepseekocr` (+`_resize`, `_overview`), `deepseekocr2` (+`_overview`) |

Suffixes select fixture variants, for example fused QKV, legacy FFN names, concatenated CLIP
feature layers, whole-tensor QK norms, an odd frame count, one-sided clipping bounds or OCR
overview separators. Both suites compare the raw encoder and its `AdaptMmprojToGenAI` form
with normalized MSE below `1e-5`.

The standalone op tests in `test_ops.cpp` read `../mmproj_*.npy`, `../vision_rope_expected.npy`
and `../multimodal_imrope_expected.npy`: window partition, SAM relative positions,
vision/interleaved RoPE, antialiased bilinear resize and 2D im2col.

## Reference revisions

The fixtures were generated with llama.cpp
[`16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb`](https://github.com/ggml-org/llama.cpp/commit/16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb),
except `qwen3vl_merger`, `muse-glimmer`, `gemma4uv_low_contrast`, `gemma4v` (+`_one_sided`),
`gemma4a`, `mlp_feature` and `internvl_qknorm`, which were generated with
[`03fa73cb27f5c251b9528489b18d303b1366aca4`](https://github.com/ggml-org/llama.cpp/commit/03fa73cb27f5c251b9528489b18d303b1366aca4)
(aligned-corner position interpolation, the Muse Glimmer encoder, and a `ggml_clamp` that no
longer clamps its source in place). The Gemma4 fixtures clip only the Q input, so clipping that
leaks into the shared K/V input fails them. Regenerating every family at `03fa73cb` reproduces
the stored arrays, except `deepseekocr`, whose embeddings differ at normalized MSE `7e-7`.

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
export PYTHONPATH="$LLAMA_SRC/gguf-py"
python3 gen_mmproj_accuracy.py --oracle ./mmproj_oracle --qwen3-oracle <03fa73cb mmproj_oracle>
python3 gen_mmproj_dynamic_accuracy.py --oracle ./mmproj_oracle --ops-oracle ./mmproj_ops_oracle
```

`--families` regenerates a subset. `mmproj_oracle` takes
`model vision|audio width height input.f32 output.f32 [second_frame.f32]`;
`GGUF_ORACLE_DUMP=<directory>` writes its intermediate F32 tensors, and
`GGUF_ORACLE_OVERVIEW=1` adds the OCR view separator. `mmproj_fixtures.py` holds the helpers
shared by the generators and `validate_mmproj.py`.

## Real checkpoints

`validate_mmproj.py` compares a real projector with `mmproj_oracle` on synthetic normalized
inputs and writes a JSON report. For a quantized projector, compare with an F32 copy of the
same represented weights made by `dequantize_mmproj.py`; the pinned `llama-quantize` rejects
`clip` files.

```sh
python3 dequantize_mmproj.py mmproj-Q8_0.gguf mmproj-F32.gguf
python3 validate_mmproj.py mmproj-Q8_0.gguf --reference-model mmproj-F32.gguf \
    --oracle ./mmproj_oracle --report encoder.json [--modality audio] [--width W --height H]
```

Without `--reference-model`, both runtimes execute the same quantized file; llama.cpp then
quantizes activations in its Q8 matmuls, so that comparison is a diagnostic, not acceptance.
Keep the reports outside the source tree.
