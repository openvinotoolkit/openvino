# LFM3 CPU validation

These results distinguish frontend conversion, represented-F32 arithmetic,
quantized CPU parity and GenAI API integration. Failed comparisons retain the
existing acceptance gate; they are not promoted to passes by diagnostics.

## Inputs

- llama.cpp CPU reference: `604cc430239c76ff636c443e3400bf552b750421`, with OpenVINO disabled.
- Decoder snapshots and filenames: [supported models](supported_models.md#lfm3).
- Projectors: matching VL, Audio and Omni Q8_0 files from those publisher snapshots.
- Inference: CPU F32, activation quantization disabled. Decoder reference KV is F16;
  streaming media and speech state is F32. No GPU results are claimed.
- Scope: LFM3-1.7B, LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B. Decision/D1 is excluded.

## Frontend checks

| Check | Result |
|---|---|
| Focused decoder, media, speech and combined Omni tests | 15/15 pass, no skips |
| Required unfiltered frontend unit binary, architecture fixtures enforced | 706 pass, 1 optional real-embedding checkpoint skip, 0 failures; operation-coverage gate passes |
| Independent extension-library tests | 2/2 pass |
| Affected GenAI regressions, including LFM2/Gemma audio, media ordering and SDPA | 104/104 pass, no skips |
| Final GenAI real-checkpoint public APIs and saved IR | 7/7 and 3/3 pass, no skips |
| Four Q4_0 decoders, prefill plus 12 reference-token steps each | 52/52 greedy choices match |
| Quantized decoder full-logit NMSE | `5.1e-5..3.175e-3`; the greedy-choice gate passes, strict full-logit parity is not established |
| Q8_0 vision/audio, 12 embedding cases and associated states | 156/180 metrics fail NMSE `<1e-5`; all 12 embedding cases fail |
| Same media weights expanded to represented F32 for the reference | 180/180 metrics pass NMSE `<1e-5` |
| Q8_0 Audio and Omni depthformer | 4/4 depth comparisons fail NMSE `<1e-5`; range `2.265e-4..2.600e-4` |
| Q8_0 Audio and Omni feedback, codec spectra and codec states | 132/132 metrics pass NMSE `<1e-5`; feedback is exact |
| Audio speech reference expanded to represented F32 | 68/68 metrics pass NMSE `<1e-5` |

The media cases cover square and rectangular still images, paired video frames,
273-frame audio and streaming 256+17-frame audio across the attention window.
Speech fixtures cover masked/revealed depthformer logits, valid FSQ boundary
codes, streaming codec state, reset and window boundaries. Combined Omni fixtures
compare every native component output/state and independently adapted models,
including codec IR reload.

A bounded activation-quantization diagnostic matches the pinned Q8_0 quantizer
exactly on identical inputs. It improves an identical-input speech projection to
NMSE `1.82e-13`, but does not recover complete quantized depthformer or vision
parity. Small upstream F32 differences cross activation rounding thresholds.
No prototype quantization graph is included in production. Represented-F32
results isolate this arithmetic gap and do not waive the original failed gates.

The synthetic codec spectrum also retains a stricter elementwise diagnostic:
12 values exceed `1e-4`, maximum `1.31e-4`, while spectral NMSE is `1.63e-10`.
See [speech fixture notes](../tests/lfm3_speech_README.md) for the reference contract,
reproduction commands and retained numerical limits.

## Reproduction and evidence

Frontend fixtures and independent oracle sources are committed under
[`tests`](../tests). Follow [testing](testing.md) and run the unfiltered frontend
unit binary with `OV_GGUF_REQUIRE_ARCH_FIXTURES=1` and
`OV_GGUF_Q4_K_ZP_F16` unset. Quantized checkpoint runs set the latter variable
before process startup and retain the original numerical gates.

The [machine-readable validation summary](lfm3_validation.json) records final
counts, tested source snapshots and loaded library hashes.

Full local logs, XML, checkpoint inventories, model/library hashes and diagnostic
reports are retained outside the repositories in
`/home/vmaxim/.cache/ov-validation/lfm3-20261028/manifest.json` and its artifact directory.
Independent correctness review found one malformed RGB input boundary defect; it
was fixed with rejection checks before preprocessing and inference. Final shared
regressions also cover Qwen3.5 generated full-width gates and Gemma audio injection.

These local results do not establish GitHub Actions packaging or GPU accuracy.
Independent comparison of a complete sampled text/codebook/feedback/waveform
trajectory remains unestablished; the native generation references are text-only.
