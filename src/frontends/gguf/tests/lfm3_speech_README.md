# LFM3 speech reference fixture

`lfm3_speech.npz` contains a nonzero synthetic GGUF plus outputs from the real
llama.cpp CPU implementation at `604cc430239c76ff636c443e3400bf552b750421`.
The oracle invokes `clip_encode()` on the LFM3 depthformer and streaming codec;
expected features are never calculated by NumPy.

The depthformer has three bidirectional layers, twelve tied 2017-entry tables,
four query heads and two KV heads with head size two. Nine independent frames
exercise all masked slots, partially revealed slots, feedback, and IDs 0, 2015,
and 2016. The codec uses the reference's required 512-channel state, eight
alternating convolution/attention layers, nonuniform sandwich norms, sigmoid
head gates, and the `[8,7,6,6]` FSQ table. Query and KV widths differ. Large
codec matrices repeat a random 4093-element pattern to compress the fixture;
this prime period keeps channels and the convolution's three projections
distinct. Norms and attention/gate weights remain independent random values.

Codec calls contain one, two, and six frames (six STFT positions per frame).
The final chunk crosses the 30-position attention window. Additional calls
reset state and decode two frames, then reset and decode all nine frames.
The archive keeps the thirteen nonzero state outputs for every call.

Spectral assertions use the established mmproj component gate in
[`docs/testing.md`](../docs/testing.md): NMSE below `1e-5`, with exact shape and
finite-value checks. Depth logits, feedback, and logical recurrent states keep
the existing elementwise helper. The initial spectral elementwise diagnostic
failed twelve values on the reset two-frame clip: maximum absolute difference
`1.3113e-4`, NMSE `1.6276e-10`. Stateless and plain stateful execution had identical
outputs. Intermediate CPU-reference bisection showed accumulated F32 differences
amplified by layer four's FFN post norm; the same reference norm input agrees at
`7.15e-7` maximum absolute error. The diagnostic failure and metrics remain in
`speech-summary.json` under the task's validation artifact directory. No existing
acceptance gate was changed.

Real Q8 checkpoint validation uses the same pinned CPU oracle, F32 inference,
disabled OpenVINO dynamic quantization, and F32 KV states. Each report contains
68 comparisons: two depth outputs, feedback, five spectra, and sixty state
arrays including the five position states. The required NMSE gate remains
`1e-5`.

| Checkpoint | Masked depth NMSE | Partly revealed depth NMSE | Result |
| --- | ---: | ---: | --- |
| LFM3-Audio-2B-Q8_0 | `2.26475e-4` | `2.33621e-4` | Both depth checks fail |
| LFM3-Omni-2B-Q8_0 | `2.41716e-4` | `2.60034e-4` | Both depth checks fail |
| Audio Q8 weights expanded to represented F32 | `1.26399e-12` | `1.48760e-12` | All 68 checks pass |

Original Q8 feedback is exact. Both checkpoints' spectra have maximum NMSE
`8.38669e-8`; recurrent states have maximum NMSE `2.67943e-7`. The represented
F32 diagnostic expands Q8 generation weights exactly, retains original F16
weights, and omits encoder branches. It checks graph math and layout; the
original Q8 depth failures remain unresolved, and no represented F32 Omni
check was run.

An artifact-only activation Q8 prototype patches 37 constant-weight depth
MatMuls and explicitly emulates the stored FP16 per-block scale. Its projection
agrees with the CPU oracle at NMSE `6.66357e-13`, but full depth still fails at
`1.64416e-4` and `1.82022e-4`. Feeding the identical reference layer-zero norm
output to its query projection gives NMSE `1.81619e-13`, maximum absolute error
`3.33786e-6`. This supports activation rounding threshold changes from small
upstream F32 differences as an explanation for error growth. It does not
establish full Q8 equivalence, and no production quantization patch was made.
The real reports and diagnostic records are retained beside
`speech-summary.json` in the task's validation artifact directory.

Build the oracle against the pinned CPU-only `mtmd`, `ggml`, and `ggml-base`
libraries, with include directories `tools/mtmd`, `include`, and `ggml/include`:

```sh
c++ -std=c++17 -O2 lfm3_speech_oracle.cpp \
  -I"$LFM3_REFERENCE_SRC/tools/mtmd" -I"$LFM3_REFERENCE_SRC/include" \
  -I"$LFM3_REFERENCE_SRC/ggml/include" -L"$LFM3_REFERENCE_BUILD/bin" \
  -Wl,-rpath,"$LFM3_REFERENCE_BUILD/bin" -lmtmd -lggml -lggml-base \
  -o lfm3_speech_oracle
PYTHONPATH="$LFM3_REFERENCE_SRC/gguf-py" python3 gen_lfm3_speech_fixtures.py \
  --oracle ./lfm3_speech_oracle
ov_gguf_frontend_tests --gtest_filter='GGUFLfm3Speech.*'
```

The generator retains its work in `speech-fixture-work` beside the oracle.
Run from the frontend test source directory or supply an absolute `--output`.

Native inputs use `[1,1,F,D]` hidden states and `[1,1,F,12]` local code IDs.
`audio.depth.mask` is one for revealed slots, zero for masked slots, whose
lookup ID must be valid (use zero). Codec codes must be below 2016.
`audio.codec.positions` contains the continued positions `[1,1,1,6F]`.
Its additive mask is `[1,1,6F,29+6F]`; entry `(q,k)` is zero precisely when
`q <= k <= q+29` and `base-29+k >= 0`, and negative infinity otherwise.
This hides nonexistent cached keys after reset.

Outputs are `audio.depth_logits [1,F,12,2017]`,
`audio.feedback [1,1,F,Dtext]`, and `audio.spectrogram [1,1,6F,1282]`.
Spectral channels contain 641 log-magnitudes followed by 641 phases.
`audio.istft_window` is a 1280-sample periodic Hann window for host inverse STFT
at 24 kHz with hop 320. Recurrent input/output pairs are recorded for
`GGUFMakeStateful`; positions and masks remain caller inputs.
