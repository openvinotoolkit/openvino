# GGUF validation batches

Use this for repeated model/configuration checks across the GGUF frontend,
llama.cpp OpenVINO backend and GenAI. Use [testing.md](../../../../src/frontends/gguf/docs/testing.md)
for build targets, fixture generation and individual tests. Registration or
conversion alone does not establish accuracy or end-to-end integration.

## Define the matrix and reference

Record repository revisions, model/projector paths and revisions, tensor-type
inventory, fixtures/media, prompts, generation settings, seeds, toolchain and
runtime settings. Record hashes of the exact model, projector, media, reference
and build wrapper files used. Verify the libraries actually loaded after
inference; read [build and runtime identity](build-runtime.md) for overlays,
incremental builds, multiple worktrees or dispatched CPU kernels.

Specify selected scenarios, expected test counts and allowed skips, numerical
metrics/tolerances, per-case timeouts and artifact paths. Distinguish conversion,
decoder numerics, text quality, vision/audio, PA/SDPA and Optimum compatibility.
Passing one scenario does not establish support for the others. Keep exclusions
and missing coverage explicit. A quantized reference checks backend parity, not
fidelity against an unquantized FP32/FP16/BF16 checkpoint.

Pin the actual llama.cpp/ggml CPU source and build and verify OpenVINO is disabled
in that reference. Specify inference and KV cache precision independently,
attention backend, dynamic activation quantization, threads, sampling, chat
template and special-token policy. Compare prompt token IDs before blaming the
encoder or decoder. Check matching explicit caches and the product default
separately; user properties must override a model-specific default.

Inspect tensor types rather than checkpoint names: Q4_0 files can also contain
Q4_1, Q8_0 or K-quants. When the user accepts Q4_K_M loss pending a plugin fix,
use Q4_0 fixtures for the requested gate and retain K-quant diagnostics. Do not
assume every Q4_0 output matches or relax other parity gates.

Use the pinned quantizer to create or exactly expand fixtures when needed,
preserving hashes, conversion revision/options and tensor inventory.
Requantization creates a different fixture. An F16 expansion of quantized weights
isolates arithmetic; it is not the original unquantized checkpoint and does not
waive the failing comparison. Exercise a new quantizer through its actual CMake
target as well as checking its output.

Keep the chosen reference and criteria. Never silently change precision,
thresholds, expected failures or coverage to obtain a pass. For performance,
preserve hardware, load, warmup and repetition settings; noisy measurements do
not establish a regression or improvement.

## Multimodal and API coverage

Choose a small discriminating matrix: synthetic square and real JPEG inputs,
single/multi media, relevant video/audio, SDPA/PA, cached and modern chat, reset,
cancellation and beam search where supported. Include encoder geometry and
limits from GGUF metadata. Square smoke inputs miss resize/token-limit and
sliding-window bugs. Combining marks, raw byte-token spellings and repeated
newlines expose tokenizer differences hidden by ASCII-only prompts.

Compare exact video frame sampling, timestamps, boundary tokens and assembled
positions. If a pinned mtmd revision needs an adapted oracle, describe that
adaptation and preserve its CPU frame encoder and decoder. That validates the
assembly contract; it does not exercise an upstream video API. Test Optimum
directory and serialized-map loading and existing processor defaults separately
from llama parity. Shared layout does not imply identical media geometry.

State replay semantics: teacher-forced greedy choices on one engine's generated
history measure local decisions, not identical free-running sequences. Tiny or
random fixtures can produce nonsense text while numerically agreeing. Keep
parity, model quality and fidelity to unquantized weights as separate claims.

GenAI-adapted Gemma4 image groups intentionally follow Optimum-intel's window behavior; see the
[accepted parity gap](../../../../src/frontends/gguf/docs/mmproj.md#gemma4-image-window-parity-gap).
Keep affected llama.cpp comparisons as diagnostics with their original gates; report
SDPA/PA agreement and API compatibility separately. Do not attribute the semantic
mismatch to quantization or turn it into a numerical-parity pass.
Exercise repeated media requests with prefix caching enabled and disabled; an isolated
prefill pass cannot establish cache-reuse correctness. Keep default-cache failures
separate from the [image prefix-cache gap](../../../../src/frontends/gguf/docs/mmproj.md#bidirectional-image-prefix-cache-gap)
diagnostics.

For failures, follow [multimodal integration bisection](../../../../src/frontends/gguf/docs/debugging_accuracy.md#multimodal-integration-bisection):
template/token IDs, encoder tensors, insertion/positions, masks/windows, cache
precision, then logits on identical histories. Identical quantized weights do
not rule out CPU activation arithmetic differences.

## Execute with existing workflows

Iterate on a focused local reproducer, then use the repository's GHA workflows for
broader validation. The [frontend GTest](../../../../.github/workflows/job_cxx_unit_tests.yml),
[model-hub](../../../../.github/workflows/job_gguf_models_tests.yml) and
[llama.cpp backend](../../../../.github/workflows/job_gguf_llamacpp_validation.yml) jobs
run in OpenVINO's Ubuntu workflows. [Ubuntu 22](../../../../.github/workflows/ubuntu_22.yml)
also runs the [GenAI acceptance job](../../../../.github/workflows/job_gguf_genai_validation.yml)
against its own OpenVINO artifacts and a pinned GenAI SHA.
Check the workflow's actual jobs and filters before claiming coverage. A green job with a
skipped GGUF component or no collected tests provides no GGUF evidence.

Use `gh workflow run <workflow> --ref <branch>` with supported revision inputs,
then `gh run watch <run-id> --exit-status`. Record the workflow revision, tested
commit (including PR merge commits), companion repository SHAs and reference pin.
Verify artifact provenance when a workflow consumes another repository's build;
`latest_available_commit` does not identify the requested frontend revision.
Inspect GTest XML, pytest counts/skips and acceptance reports downloaded with
`gh run download <run-id> --dir <artifact-directory>`. Required coverage and
numerical gates belong in the tests, rather than a second report runner.

For local checks, invoke the GTest binary or pytest/acceptance script directly,
with explicit filters and report paths. Keep commands, logs and reports outside
the checkout, for example `~/.cache/ov-validation/<task>/`. Record resolved binary
and library hashes, configuration and untracked source dependencies when using
an overlay. A successful local run does not prove GHA packaging or job selection.

Retain failed attempts. Use `gh run rerun <run-id> --failed` for a deliberate retry
of transient failures; fixes require a new run at the new revision. Changed inputs
invalidate reuse. Expand or repeat only for changed inputs, a new failure or
missing required coverage. Stop on a user pause and retain the run IDs and logs.

## Delegate and report

When delegation is available and authorized, use one worker for long batches
with the user's configured model preference. Supply a fresh compact
brief with the [ov-gguf entrypoint](../SKILL.md), this guide, the workflow/run ID
or exact local command, stable source/build paths, acceptance contract and artifact paths. Run directly
if delegation is unavailable. The worker executes and reports evidence; the main
agent owns diagnosis, fixes and acceptance. Missing prerequisites or unexpected
results return to the main agent without expanding setup or changing criteria.

Freeze sources and binaries for the batch, using an isolated worktree/build when
needed. Wait for execution and the provenance check to finish before committing,
merging or rebuilding those inputs. Report useful checkpoints from the workflow
artifacts without per-test polling.

Return tested revisions and runtime identity, completed/remaining cases, exact
selected counts/skips, failures with excerpts, reproduction commands and artifact
paths. Distinguish selected skips from filtered or unregistered suites and disclose
standalone fixture adapters. Group equivalent errors without asserting a common
cause. Preserve original failures and numerical differences for diagnosis; known
arithmetic differences remain failed parity checks unless the user changes the
acceptance contract.
