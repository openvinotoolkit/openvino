# GGUF frontend and GenAI acceptance — September 2026

**Full support is not achieved.** This report covers the requested Qwen3.5
0.8B/2B/4B/9B, Qwen3.6 35B-A3B, Qwen3.8 27B, Gemma4
E2B/E4B/12B/26B-A4B/31B and Muse Glimmer 30B checkpoints. Both Q4_0 and
Q4_K_M files were obtained for every size. Qwen3.8 UD-Q4_K_M was additionally
checked and must not be confused with the plain Q4_K_M variant.

All 24 requested language files convert, and 19 pass the bounded native
reference-accuracy check. All 12 projectors convert, with encoder accuracy
exceptions listed below. No pair is qualified for all GenAI scenarios.

The OpenVINO branch `mvafin/gguf/mmproj-support` was rebased onto
`upstream/master` at `ae5eba017d` and pushed. Implementation through `6e39da06cc`
includes Qwen MoE and hybrid PagedAttention adaptation, Gemma4 shared-weight and
MoE fixes, interpolation accuracy, and the Muse Glimmer vision builder.
GenAI was tested on companion branch `mvafin/gguf/mmproj-acceptance` at
`0943b301`, including the GGUF recurrent-state history-reset fix.

The [machine-readable results](../tests/test_data/mmproj_accuracy/acceptance_2026_09.json)
contain checkpoint repositories, immutable revisions, SHA256 hashes, numerical
metrics, scenario errors and local artifact paths. The preexisting Muse Q4_0 file
has a recorded SHA256 but no recovered publisher revision. Individual runs retain
the drivers' manual source markers; affected model runs were repeated after fixes.
These runs span branch development rather than a single commit. Incremental
builds retained older embedded version strings, so those strings alone do not
identify the tested source.

A final rebase incorporated four newer upstream commits after the initial
`a9955dc014` baseline. They change AUTO device telemetry, CMake source grouping
and documentation. The GGUF frontend, core/inference code and CPU execution code
are identical to the tested pre-rebase tree at `de4259fd56`; per-run source hashes
are retained from that tree.

## Method and limits

All runs used CPU, F32 inference, F16 KV cache, four inference threads and disabled
dynamic activation quantization. GenAI was forced to use `GGUF_READER=FRONTEND`.
PA means PagedAttention with the default scheduler's prefix caching enabled.
Earlier raw Qwen reports mistakenly labeled this default as disabled; the
consolidated JSON annotates that metadata correction and preserves measured values.

Language acceptance uses llama.cpp CPU at
`03fa73cb27f5c251b9528489b18d303b1366aca4`, the same quantized checkpoint, the prompt
`The capital of France is`, and 13 next-token choices on identical reference
histories. Passing requires the same first token and at least 12/13 choices.
Native frontend checks apply stateful/GenAI adaptation directly and retain
full-logit normalized MSE. GenAI compares token choices; it does not expose a full-logit error measurement in this harness.

Tokenizer, greedy generation, streaming equality, chat/reset, batch-two versus
individual generation, and beam-search output checks are bounded smoke tests.
A successful greedy call does not certify response quality. The single-prompt
accuracy result is not broad language-model qualification. Media preprocessing,
embedding assembly, end-to-end image/audio/video/mixed generation, long context,
GPU/NPU, speculative decoding and concurrent/cancelled requests remain unqualified.
Performance was not measured.

## Language reference accuracy

Each cell gives status (`P` pass, `F` fail) and matching choices out of 13.
`F*` means the first token mismatched, even where the overall match count reaches 12. `Unsupported` means
conversion failed before numerical comparison. Functional failures appear below.

| Checkpoint | Native frontend | GenAI SDPA | GenAI PA |
|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | P 13/13 | P 13/13 | P 12/13 |
| Qwen3.5-0.8B-Q4_K_M | P 13/13 | P 13/13 | F 11/13 |
| Qwen3.5-2B-Q4_0 | P 13/13 | P 13/13 | F 6/13 |
| Qwen3.5-2B-Q4_K_M | F 11/13 | F 11/13 | F 7/13 |
| Qwen3.5-4B-Q4_0 | P 13/13 | P 13/13 | P 13/13 |
| Qwen3.5-4B-Q4_K_M | P 13/13 | P 13/13 | F 7/13 |
| Qwen3.5-9B-Q4_0 | P 13/13 | P 13/13 | P 13/13 |
| Qwen3.5-9B-Q4_K_M | P 13/13 | P 13/13 | F 5/13 |
| Qwen3.6-35B-A3B-Q4_0 | P 12/13 | P 12/13 | F 8/13 |
| Qwen3.6-35B-A3B-Q4_K_M | P 12/13 | P 12/13 | F 11/13 |
| Qwen3.8-27B-Q4_0 | P 13/13 | P 13/13 | F 11/13 |
| Qwen3.8-27B-Q4_K_M | P 13/13 | P 13/13 | F 11/13 |
| Qwen3.8-27B-UD-Q4_K_M | Unsupported | Unsupported | Unsupported |
| Gemma4-E2B-Q4_0 | P 12/13 | P 12/13 | P 12/13 |
| Gemma4-E2B-Q4_K_M | P 12/13 | P 12/13 | P 12/13 |
| Gemma4-E4B-Q4_0 | P 13/13 | P 13/13 | P 13/13 |
| Gemma4-E4B-Q4_K_M | F 10/13 | F 10/13 | F 10/13 |
| Gemma4-12B-Q4_0 | F* 11/13 | F* 11/13 | F* 11/13 |
| Gemma4-12B-Q4_K_M | P 13/13 | P 13/13 | P 13/13 |
| Gemma4-26B-A4B-Q4_0 | F* 12/13 | F* 12/13 | F* 11/13 |
| Gemma4-26B-A4B-Q4_K_M | F* 10/13 | F* 10/13 | F* 10/13 |
| Gemma4-31B-Q4_0 | P 12/13 | P 12/13 | P 12/13 |
| Gemma4-31B-Q4_K_M | P 12/13 | P 12/13 | P 12/13 |
| Muse Glimmer-30B-Q4_0 | P 13/13 | P 13/13 | P 13/13 |
| Muse Glimmer-30B-Q4_K_M | P 13/13 | P 13/13 | P 13/13 |

## GenAI generation gaps

- **All pairs:** GGUF `mmproj_path` loading is absent in the tested GenAI revision.
  `VLMPipeline` looks for `<language.gguf>/openvino_language_model.xml` and fails.
  Image, audio, video and mixed requests are blocked at loading; encoder success
  does not qualify these scenarios. The older Gemma3 prototype was removed before
  this revision.
- **SDPA:** batch-two and beam-search checks fail across the tested families, with
  batch-one recurrent layouts or attention-mask shape mismatches. Single-request
  text generation and streaming may still work.
- **PA prefix caching:** several Qwen checkpoints lose reference-choice accuracy.
  Qwen3.5 2B Q4_0 gives 6/13 with the default prefix cache, versus 13/13 with prefix
  caching explicitly disabled and 13/13 with a fresh pipeline for each history.
  These are diagnostic alternatives, not a fix or qualification of prefix caching.
- **Chat:** Qwen3.8 and Gemma4 26B encounter unsupported Jinja template syntax.
  Where chat/reset smoke tests pass, conversational logits have not been compared
  against the reference.
- **Additional Qwen3.8 UD variant:** UD-Q4_K_M contains IQ4_XS (GGML type 23), which
  the frontend cannot convert. The plain Q4_K_M file is a separate matrix entry.

The JSON records each backend's scenario status and exact exception. Passing
batch/beam/streaming smoke checks does not override a failed reference-accuracy
check.

## Encoder conversion and accuracy

All 12 downloaded projector files convert, including Muse Glimmer. Gemma4 E2B,
E4B and 12B files expose both vision and audio branches. The downloaded Gemma4
26B/31B files contain vision only; missing source audio is not a conversion failure.

Encoder comparisons use deterministic synthetic images/features against
represented-weight F32 copies of the same checkpoint in llama.cpp CPU. They do
not compare against the publisher's original unquantized model. Acceptance is
finite normalized MSE strictly below `1e-5`; alternative precision modes are
reported separately.

| Encoder | Tested shapes | Default accuracy |
|---|---|---|
| Qwen3.5 0.8B, 4B, 9B; Qwen3.6 35B; Qwen3.8 27B | Vision 96×64 and 192×224 | Pass |
| Qwen3.5 2B | Vision 96×64 and 192×224 | Fail: NMSE 2.52e-5 and 1.32e-5 |
| Gemma4 E2B/E4B | Vision 144×96; audio 101/104 frames × 128 mel bins | Pass |
| Gemma4 12B | Vision 144×96; waveform 9×640 samples | Pass |
| Gemma4 26B/31B | Vision 144×96 | Pass |
| Muse Glimmer | Vision 56×84 and 504×56 | Fail: NMSE 0.0449 and 0.1159 |

Most encoder reference runs use llama.cpp
`16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb`; Muse uses the newer language-oracle
revision above, which implements that encoder. F32 Muse source versus F32
reference passes at `4.95e-7`. With `OV_GGUF_Q4_K_ZP_F16=1`, the quantized Muse
source passes both grids at `2.39e-6` and `5.41e-6`.

Default Q4_K conversion performs additional lossy u4 requantization. The optional
F16 zero-point mode avoids that approximation. It also makes the Qwen3.5 2B
Q4_K_M native language check pass, but Gemma4 E4B Q4_K_M still gives 10/13 and 26B Q4_K_M still fails:
not every failure is explained by this approximation. These diagnostics leave
default-mode failures visible in the matrix.

Qwen3.5 2B vision still exceeds the strict threshold with an F32 source copy.
Rounding its input through F16 reduces the error below threshold, consistent with
the reference convolution's F16 activation arithmetic. That changed-input
experiment does not qualify the original-input comparison.

## Regression checks and reproduction

After rebuilding on the final upstream rebase, the OpenVINO frontend suite passed **403 tests with no failures or skips**,
including synthetic architecture accuracy, architecture fingerprints, operation,
quantization, adaptation, dynamic encoder and combined-modality checks. The
extension-library test and GenAI recurrent-cache classification regression each
passed. Fingerprint conversion/rejection checks do not mean every enumerated
architecture is supported.

Build the frontend library and test executable together after source changes:

```sh
cmake --build build-mmproj --target openvino_gguf_frontend ov_gguf_frontend_tests pyopenvino -j 8
bin/intel64/Release/ov_gguf_frontend_tests
bin/intel64/Release/ov_gguf_architecture_library_tests
```

Generate a CPU reference with `tests/architecture_oracle.cpp` at the recorded
llama.cpp revision. `llama.bin` contains vocabulary size followed by the 13 F32
logit rows; `llama.bin.tokens` holds the token schedule. Run the GenAI harness
against builds from the recorded branches:

```sh
python3 src/frontends/gguf/tests/validate_genai.py language.gguf \
  --backend PA --reference llama.bin --report PA.json --mmproj mmproj.gguf
python3 src/frontends/gguf/tests/validate_genai.py language.gguf \
  --backend SDPA --reference llama.bin --report SDPA.json --mmproj mmproj.gguf
```

For native checks, set `OV_GGUF_ACCURACY_DATA` to a directory containing the
checkpoint as `llama.gguf`, `llama.bin` and `llama.bin.tokens`, and select
`Architectures/GGUFArchitectureAccuracy.PrefillAndCachedDecodeMatchLlamaCPU/llama`.
The parameter name `llama` selects fixture filenames; the actual architecture
comes from the checkpoint metadata. Use `tests/validate_mmproj.py` and
`tests/dequantize_mmproj.py` for encoder comparisons as described in
[mmproj.md](mmproj.md#reproduction-and-evidence).

The remaining work is GenAI multimodal loading and family adapters, correct cached
positions/state restoration, SDPA batch/beam layouts, template compatibility,
IQ4_XS format coverage, and
resolving the recorded numerical failures. Further acceptance must cover real
media preprocessing and assembly, cached multimodal histories, resets and the
other currently unqualified scenarios before claiming complete model support.
