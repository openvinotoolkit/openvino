# GGUF frontend and GenAI acceptance — September 2026

**Validation paused on September 24; full support is not yet qualified.** The requested
Qwen3.5 0.8B/2B/4B/9B, Qwen3.6 35B-A3B, Qwen3.8 27B, Gemma4
E2B/E4B/12B/26B-A4B/31B and Muse Glimmer 30B checkpoints are available locally
in both Q4_0 and Q4_K_M: 24 language files and 12 associated projectors.
All convert through the native frontend. The extra Qwen3.8 UD-Q4_K_M file
contains unsupported IQ4_XS tensors and is separate from the requested plain
Q4_K_M checkpoint.

The resumed run uses OpenVINO branch `mvafin/gguf/mmproj-support` at
`3639ee5276` and companion GenAI branch **`gguf-frontend-mmproj`** at
`70e88dd2`, with validator provenance improvements recorded separately.
The local worktree directory retains its earlier name
for build reuse; it does not indicate a separate acceptance branch.

## Implemented and tested

- Native GGUF language/projector pair loading, in-memory adaptation and tokenizer
  construction, with separate reachable vision/audio branches.
- Family adapters for the requested Qwen, Gemma4 and Muse checkpoints, including
  projected feature placement, dynamic positions, Gemma auxiliary token IDs,
  and Gemma4 audio preprocessing.
- SDPA padded batches and beam-state reordering for Qwen dense/MoE and Gemma
  MQA/MoE. A CPU attention crash on the first batched inference is fixed.
- PA recurrent prefix-cache checkpoint ownership and copy-on-write fixes.
  Qwen3.5 2B Q4_0 now matches 13/13 reference language choices with prefix
  caching, compared with 6/13 in the historical run.
- GGUF chat-template compatibility and encoder precision fixes. Native Q4_K language and projector weights
  use an integer zero point by default, which keeps the compressed FullyConnected path;
  `OV_GGUF_Q4_K_ZP_F16=1` preserves fractional zero points and avoids the additional lossy
  requantization, as used for the accuracy checks.
- GenAI modern audio-history ownership, follow-ups, edits and cancellation
  rollback, covered by a synthetic feature-placement regression.
- PA multimodal chat tokenization now recognizes templates applied by the caller,
  preventing a duplicate BOS token in legacy and modern histories. A real Gemma4
  E2B Q4_K_M reset reproducer now returns identical tokens for independent image
  requests and fresh chats, including after audio requests.

The current frontend suite passed **410 tests**, plus the external architecture-library test. The selected GenAI GGUF, scheduler,
model-runner, hybrid-cache and SDPA suites passed **121 tests**. This selected run
reports missing cache-type/backend-routing coverage because `CACHE_TYPES_CSV` is unset.
Final checkpoint qualification retains its own source manifests
and test records.

The resumed original-publisher language matrix completed all 48 configurations:
36 passed and 12 failed. Each variant below failed on both PA and SDPA; all
other variants passed the language criteria.

| Variant | Matching choices | First choice agrees |
|---|---:|---|
| Qwen3.5 0.8B Q4_K_M | 12/13 | No |
| Qwen3.6 35B Q4_0 | 11/13 | Yes |
| Gemma4 E2B Q4_0 | 11/13 | Yes |
| Gemma4 12B Q4_0 | 10/13 | No |
| Gemma4 26B Q4_0 | 7/13 | No |
| Gemma4 26B Q4_K_M | 8/13 | No |

Gemma4 26B Q4_K_M PA additionally failed the exact batch-versus-individual
generation comparison for one prompt. A bounded prefix-cache diagnostic is
pending; this failure is retained separately from publisher-reference accuracy.

## Real-checkpoint evidence and remaining gaps

The table below records historical checks, including comparisons against quantized
and represented-F32 references. It does not establish acceptance against the
original publisher weights. The resumed matrix uses pinned original BF16/F16
language and projector weights, with the actual tensor types and file hashes
recorded in each reference manifest.

| Configuration | Evidence | Remaining qualification |
|---|---|---|
| Qwen3.5 0.8B/2B/4B, both precisions and backends | Language, image/video, legacy/modern chat, beam and reset checks pass on v7 | Broader media/context coverage |
| Qwen3.5 9B, both precisions and backends | Language and bounded multimodal/API checks pass on v7/v9 | Broader media/context coverage |
| Gemma4 E2B Q4_0 PA/SDPA | Language, image/audio/mixed, chat, beam and reset checks pass on v7 | PA chat rerun after the tokenization fix |
| Gemma4 E2B Q4_K_M | Language passes; v9 PA has 100% represented-F32 agreement for all 11 media/chat cases, with beam, cancellation and reset passing; SDPA represented-F32 run also passes | Original quantized-reference audio first-token failure retained |
| Gemma4 E4B, both precisions and backends | Language and bounded multimodal/API checks pass on v7 | PA chat rerun after the tokenization fix |
| Gemma4 12B Q4_0 PA/SDPA | Represented-F32 language checks pass; earlier text/audio 100%, image/mixed 90% | Updated full multimodal/API matrix; quantized-reference language mismatch retained |
| Gemma4 26B Q4_0 | Represented-F32 native comparison matches 12/13 choices; layer comparison identifies a near-tied expert selection at layer 8 | First-token criterion remains failed; complete updated GenAI matrix |
| Other requested sizes/precisions | Conversion and historical numerical measurements recorded | Updated PA/SDPA language and multimodal matrix in progress |

The larger matrix uses frozen runtime copies, so rebuilding local libraries does
not invalidate running tests. Snapshot v9 includes the PA chat tokenization fix.
It retains v7 results for unchanged language, SDPA and Qwen execution paths;
affected PA multimodal cases are rerun. Each inherited result retains its
original runtime provenance. Snapshot v3 predates modern audio-history fixes.
Its extended API checks also lacked cleanup between scenarios and compared exact
greedy trajectories across API paths. Those API results require reruns. Updated
checks isolate scenarios and compare choices against llama.cpp on identical
histories, using each path's own generated transcript.

Known numerical failures remain visible. Examples from earlier runs include
Muse Q4_0 image-chat choice agreement of 75%, Gemma31 Q4_0 image agreement of
85%, and several language-reference failures. These must be reproduced on the
current runtime and investigated before qualification. Encoder accuracy alone
is not evidence of correct end-to-end media preprocessing or cached generation.

Gemma4 E2B/E4B/12B projectors contain audio and vision branches. The downloaded
Gemma4 26B/31B projectors contain vision only. Missing source modalities cannot
be enabled by conversion. GPU/NPU, long contexts, arbitrary concurrent requests,
all media shapes/chunk boundaries and exhaustive API combinations remain
unqualified. The resumed run also measures CPU load time, peak process RSS,
first-token latency and generation throughput for each model's Q4_K_M checkpoint
with PA and `OV_GGUF_Q4_K_ZP_F16` unset.
Performance optimization is restricted to frontend changes.

The Gemma4 unified-vision patch normalization now recenters pixels before the
F32 mean reduction. A low-contrast CPU-oracle fixture improves from normalized
MSE `0.00112` to `5.25e-7`; both raw and adapted dynamic-grid tests pass.
A targeted Gemma4 12B Q4_0 SDPA image comparison improves from 90% to 100%
matching choices against represented-F32 language/projector references. Video
and full API validation of this fix were interrupted at the user's request;
the earlier 85% video failure remains unresolved pending completion. The
Gemma4 12B Q4_K_M PA/SDPA language checks also passed before the pause.

The resumed Gemma4 12B Q4_0 represented-F32 diagnostic initially matched all text,
image, audio and mixed choices, but video matched 17/20. Investigation found that
`AdaptToGenAI` incorrectly made global layers bidirectional for image/video
tokens. Gemma4 permits this only in sliding-window layers. The frontend now
keeps global masks causal while retaining image-group bidirectionality in sliding
layers. A regression covering both Gemma3 and Gemma4, separate image groups and
cached prefixes fails on the unfixed Gemma4 graph and passes after the correction.
The fixed PA diagnostic matches 9/9 text choices and 20/20 choices for image,
video, audio and mixed input, with first-token and reset checks passing. This
resolves the represented-F32 video discrepancy. Original-publisher accuracy
remains a separate qualification requirement.

The fixed Gemma4 12B original-publisher matrix still fails all four configurations,
with identical PA/SDPA results. Q4_0 matches text 9/9, image 18/20, video 17/20,
audio 19/20 and mixed input 17/20. Q4_K_M matches text 9/9, image 19/20,
video 18/20, audio 18/20 and mixed input 15/20. Both quantizations miss the first
mixed-input token; all request/chat reset checks pass. These failures remain
recorded independently of the corrected mask behavior.

Original-publisher comparisons also expose differences present in quantized
checkpoints. Qwen3.5 0.8B with native original BF16 language/projector weights
matches every text, image and video choice in the bounded PA diagnostic. Its
Q4_K_M language checkpoint produces identical choices in quantized llama.cpp and
GenAI, including their first-choice disagreement with the original reference.
This explains that checkpoint's language failure, not every media discrepancy.

## Method and reproduction

The CPU language and multimodal oracle is llama.cpp
`03fa73cb27f5c251b9528489b18d303b1366aca4`, built without its OpenVINO backend.
Language checks require matching the first token and at least 90% of choices
on identical histories; the bounded language schedule has 13 choices. Media
checks normally use 20 choices. Native architecture/encoder F32 checks use
normalized MSE below `1e-5`. These small checks do not establish broad response
quality.

Quantized-checkpoint execution in llama.cpp and represented-weight F32 copies
are distinct references. The latter preserve the checkpoint's represented
weights; they do not recover the publisher's original weights. Gemma4 12B Q4_0 matches all 13 native language choices against the represented-weight
F32 reference (maximum logit NMSE `1.76e-4`); its quantized-reference failure is
retained separately. Gemma4 E4B Q4_K_M also matches 13/13 choices, but the
older native Q4_K requantization produced maximum logit NMSE `0.194847`.
Preserving fractional zero points reduces that to `9e-6`. These accuracy results
require `OV_GGUF_Q4_K_ZP_F16=1`; the current default uses integer zero points.
Gemma4 26B Q4_0 still fails first-token agreement against represented F32
(maximum logit NMSE `0.012255`). Layer outputs agree closely through layer 7.
At layer 8, router-logit NMSE is `1.8e-8`, but experts 56 and 35 exchange places
at the top-8 cutoff for the last prompt token. The reference gap is `0.000263`.
This localizes the first amplified difference to routing sensitivity; it does
not establish the source of all later error or satisfy the first-token criterion.

The raw cgraph weight-conversion path retains its separate quantization policy;
these native-loading results do not qualify that path.

Detailed checkpoint reports are generated validation artifacts and are kept
outside the source tree. Record checkpoint revisions, hashes, runtime provenance,
metrics and failures alongside each run. The checked-in `.npz` fixtures are inputs
and CPU-oracle expectations consumed by automated regression tests.

```sh
cmake --build build-mmproj --target openvino_gguf_frontend ov_gguf_frontend_tests pyopenvino -j 8
bin/intel64/Release/ov_gguf_frontend_tests
python3 src/frontends/gguf/tests/validate_genai.py language.gguf \
  --backend PA --reference llama.bin --report language-PA.json
```

Run GenAI's `tests/python_tests/validate_gguf_mmproj.py` with `--family`,
`--attention-backend`, `--oracle`, `--image`, `--chat` and `--api-checks`.
Add `--audio` for an audio-capable projector and `--video-frames` for supported
video assembly. `--runtime-manifest` records the frozen source manifest.
See [mmproj.md](mmproj.md#reproduction-and-evidence) for encoder validation.
