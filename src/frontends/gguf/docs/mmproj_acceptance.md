# GGUF frontend and GenAI acceptance — September 2026

**Validation is ongoing; full support is not yet qualified.** The requested
Qwen3.5 0.8B/2B/4B/9B, Qwen3.6 35B-A3B, Qwen3.8 27B, Gemma4
E2B/E4B/12B/26B-A4B/31B and Muse Glimmer 30B checkpoints are available locally
in both Q4_0 and Q4_K_M: 24 language files and 12 associated projectors.
All convert through the native frontend. The extra Qwen3.8 UD-Q4_K_M file
contains unsupported IQ4_XS tensors and is separate from the requested plain
Q4_K_M checkpoint.

OpenVINO branch `mvafin/gguf/mmproj-support` is rebased onto `upstream/master`
`ff92c0dbcc` and pushed through `274249be0a`. The companion GenAI branch is
**`gguf-frontend-mmproj`**, pushed through `69d87e60`. The local worktree directory retains its earlier name
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
- GGUF chat-template compatibility and encoder precision fixes. Q4_K projectors
  preserve fractional zero points, avoiding an additional lossy requantization.
- GenAI modern audio-history ownership, follow-ups, edits and cancellation
  rollback, covered by a synthetic feature-placement regression.

The frontend suite passed **406 tests**. The selected GenAI GGUF, scheduler,
model-runner, hybrid-cache and SDPA suites passed **120 tests** after the modern
audio-history change. Subsequent fixes and final checkpoint qualification must
retain their own source manifests and test records.

## Real-checkpoint evidence and remaining gaps

| Configuration | Evidence | Remaining qualification |
|---|---|---|
| Qwen3.5 2B Q4_0 PA | 13/13 language choices; language smoke scenarios pass | Complete updated multimodal/API matrix |
| Qwen3.5 0.8B Q4_0 SDPA | Language checks, padded batch and beam checks pass | Complete updated multimodal/API matrix |
| Gemma4 E2B Q4_0 PA/SDPA | Image/audio/mixed, legacy and modern chat, beam and cancellation/reset pass | Video and history-switching checks on the latest runtime |
| Gemma4 E4B Q4_0 SDPA | Language checks, padded batch and beam checks pass | Complete updated multimodal/API matrix |
| Gemma4 12B Q4_0 PA/SDPA | Text/audio 100%; image/mixed 90%; legacy image chat 95–100%; audio chat 100% | Modern API rerun; language-reference mismatch; PA video previously 85% |
| Qwen3.5 2B Q4_K_M | Text/image/chat 100% with faithful Q4_K and represented-weight F32 language reference | Default decoder precision still differs; not a default-mode pass |
| Other requested sizes/precisions | Conversion and historical numerical measurements recorded | Updated PA/SDPA language and multimodal matrix in progress |

The larger matrix uses frozen runtime copies, so rebuilding local libraries does
not invalidate running tests. Snapshot v3 predates modern audio-history fixes.
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
unqualified. Performance is outside this task's acceptance target.

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
retained separately. Decoder Q4_K
still defaults to lossy u4 requantization; `OV_GGUF_Q4_K_ZP_F16=1` enables a
separately reported faithful zero-point diagnostic. Projectors use faithful
zero points automatically.

[Machine-readable evidence](../tests/test_data/mmproj_accuracy/acceptance_2026_09.json)
contains current targeted runs and the complete historical baseline with
checkpoint revisions, hashes, metrics and errors. Historical descriptions of
missing mmproj loading or batch support apply only to the revisions recorded
there. Per-run frozen runtime source hashes identify dirty development builds
more accurately than embedded version strings.

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
