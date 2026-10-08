# GGUF model validation details

For GGUF batches across OpenVINO, GenAI, or llama.cpp, supplement the general
handoff with exact model and projector revisions and paths, quantization,
reference implementation and precision, prompts/media, seeds, and generation
settings. Include model, projector, media, and reference files in case inputs.

Separate conversion, text generation, accuracy, vision/audio, and PA/SDPA cases.
Passing one scenario does not establish support for the others. A quantized
reference can check backend parity, but cannot establish fidelity against an
FP32/FP16/BF16 baseline. Preserve the reference selected for the task; plausible
generated text alone is not evidence of numerical accuracy.

Report model/scenario outcomes and any missing coverage. Return numerical
differences to the main agent for diagnosis rather than changing precision,
thresholds, or expected-failure classifications to obtain a pass.

## Align the comparison before inference

Pin the actual CPU reference source/build and verify it has OpenVINO disabled. Specify
inference and KV cache precision independently, attention backend, dynamic activation
quantization, threads, sampling, chat template and special-token policy. Compare prompt
token IDs before blaming the encoder or decoder. Cache defaults may differ even with
the same inference precision; test explicit matching caches and the product default
separately. User properties must still override a model-specific default.

Inspect tensor-type inventory, not just checkpoint names: a file called Q4_0 can also
contain Q4_1, Q8_0 or K-quant tensors. When Q4_K_M loss is accepted pending a plugin fix,
use Q4_0 fixtures for the requested gate and retain K-quant results as diagnostics.
Do not generalize that exception into an assumption that every Q4_0 output matches.

Use the pinned quantizer to create or exactly expand fixtures when needed, preserving
input/output hashes, conversion revision/options, and tensor inventory. Requantization
creates a different fixture. An F16 expansion of the same quantized weights is useful
for isolating arithmetic; it is not the original unquantized checkpoint. Keep the original
failing comparison. Run a newly added quantizer through its actual CMake target as well
as exercising its output.

## Multimodal and API coverage

Choose a small discriminating matrix: synthetic square and real JPEG inputs, single/multi
media, selected video/audio scenarios, SDPA/PA, cached and modern chat, reset, cancellation
and beam search where supported. Include encoder geometry/limits from GGUF metadata.
Square smoke inputs alone can miss resize/token-limit and sliding-window bugs. Templates
with combining marks, raw byte-token spellings and repeated newlines can expose tokenizer
differences hidden by plain ASCII prompts.

For video, compare exact frame sampling, timestamps, boundary tokens and assembled token
positions. If the pinned mtmd revision needs an adapted oracle, describe the adaptation
and preserve its CPU frame encoder and decoder. This checks that assembly contract; it
does not claim an upstream video API was exercised. For Optimum compatibility, test
directory and serialized-map loading and existing processor defaults independently of
llama numerical parity. Shared layout does not imply identical media geometry.

Use report contracts to require each selected modality, API check and actual score. State
the replay semantics: teacher-forced greedy-choice agreement on one engine's generated
history measures local decisions, not identical free-running sequences. Tiny/random
fixtures can produce nonsense text while numerically agreeing. Keep parity, model quality
and fidelity to unquantized weights as distinct claims.

For a remaining numerical failure, use the GGUF accuracy skill's multimodal bisection
reference: template/token IDs, encoder tensors, media insertion/positions, mask/window
semantics, cache precision, then decoder logits with identical inputs. A top-1 mismatch
with identical quantized weights does not rule out different CPU activation arithmetic.
