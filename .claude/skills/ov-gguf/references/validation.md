# GGUF validation batches

Use this for repeated model/configuration checks across the GGUF frontend,
llama.cpp OpenVINO backend and GenAI. Use [testing.md](../../../../src/frontends/gguf/docs/testing.md)
for build targets, fixture generation and individual tests. Registration or
conversion alone does not establish accuracy or end-to-end integration.

## Family contract

Before implementing a family, inspect the pinned reference loader and graph. Record consumed
metadata and defaults, optional tensors, layer ordering, normalization/RoPE/routing order,
tensor layouts and recurrent state shapes. Metadata present in a file may be ignored by the
reference loader. Identify decoder, projector, embedding and audio contracts separately while
preserving the user's requested scope. Distinguish implemented behavior from validated checkpoints;
an unimplemented contract remains outstanding work, not an implicit scope reduction.

Test layout-sensitive recurrent paths with batching early, alongside single-sequence numerics.
Include multiple heads/tokens and unequal Q/KV dimensions where applicable.

Complete a coherent family stage before expanding implementation, preserving remaining
contracts in the manifest. Exercise the actual consumer constructor and a small prefill/decode
early, including draft/target pairing or media insertion when applicable. Test default settings
and supported overrides; conversion and factory fixtures do not exercise consumer restrictions.

## Multimodal contracts

Before connecting generation, record the pinned reference contract for each requested
modality. Read only the applicable rows; derive values from source and checkpoint metadata.

| Boundary | Facts to establish |
|---|---|
| Media preprocessing | Sample rate/channel layout; window, hop, padding and feature extraction; resize/tiling; normalization units and order |
| Token and embedding assembly | Actual special-token IDs/spellings; per-media lengths; ordered spans and positions, including adjacent and reversed inputs |
| Native and adapted graph | Input/output axes and dtypes; dynamic lengths; cache/state sizes, updates and reset; exact normalization and rotary-pair layout |
| Speech generation, when requested | Codebook layout; modality cadence, EOS and feedback; sampling defaults; spectral/output head, preserved frames and waveform reconstruction |
| API behavior | Supported configuration, media history, cancellation/reset and ownership/timing of streamed outputs |

Check one small real media input through the application early. Use independent component
references and boundary cases (unequal/dynamic dimensions, adjacent media, padding/window
edges) before a broad sweep. Record unsupported contracts as remaining work.

## Define the matrix and reference

Record repository revisions, model/projector paths and revisions, tensor-type
inventory, fixtures/media, prompts, generation settings, seeds, toolchain and
runtime settings. Record hashes of the exact model, projector, media, reference
and build wrapper files used. Verify the libraries actually loaded after
inference; read [build and runtime identity](build-runtime.md) for overlays,
incremental builds, multiple worktrees or dispatched CPU kernels.

Use one reusable runner and task manifest, extending the existing harness where available.
Update the manifest at milestones and input/environment changes; retain commands, completed
checks, failure signatures and the next diagnostic so handoffs can resume without full logs.
Diagnose the first differing boundary on identical inputs and record the predicted observation
before each experiment. Reassess the boundary, reference and loaded runtime after repeated
unchanged or inconclusive attempts before another sweep. Keep failed numerical results and
assess generated-text quality with [WWB](wwb.md) separately when that evaluation is requested.
Specify selected scenarios, per-case environments, expected test counts and allowed skips,
existing numerical metrics/tolerances, timeouts and artifact paths. Keep runtime identity,
completed cases and failed attempts in its reports; preserve the tests' acceptance gates
rather than duplicating numerical assertions in a report parser. Distinguish conversion,
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

Inspect tensor types rather than checkpoint names: a nominal quantization can contain mixed
weight types. Validate the requested original checkpoint; substituting another quantization
creates a different fixture. When quantization-quality evaluation is requested, compare both
backends with the same original-weight floating reference using [WWB](wwb.md). Retain the
original numerical gates and failures.

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

## Scenario environments

Set the environment before each process starts; do not inherit checkpoint settings into the
unfiltered unit suite. Record precision, KV precision and activation quantization independently.

| Scenario | Zero-point setting | Acceptance |
|---|---|---|
| Unfiltered frontend unit suite | Unset `OV_GGUF_Q4_K_ZP_F16` | Default test expectations, required fixtures and op coverage |
| Generated-text quality / WWB | Use and record the selected checkpoint environment | [Same-reference quality comparison](wwb.md); numerical and integration gates remain separate |
| Strict quantized checkpoint comparison | `OV_GGUF_Q4_K_ZP_F16=1` | Original full-logit or feature parity gate |
| Represented-F32 diagnostic | Explicit environment and tensor-type inventory | Arithmetic isolation; does not replace quantized parity |

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

Document intentional consumer/reference differences in the applicable frontend guide and
retain affected comparisons with their original gates. Report SDPA/PA agreement and API
compatibility separately; distinguish semantic mismatches from quantization differences.
Exercise repeated media requests with prefix caching enabled and disabled; an isolated
prefill does not establish cache-reuse correctness. Keep default-cache failures separate
from results using explicitly disabled caches.

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

Before repeated local batches, check selected IDs/counts and required fixtures with the
[coverage preflight](preflight.md#test-selection-and-required-fixtures). Retain post-run coverage
checks too: an absent or suppressed required test remains missing coverage even with exit zero.
Audit overlay link inputs when using a custom link command.

For local checks, invoke the GTest binary or pytest/acceptance script directly,
with explicit filters and report paths. Keep commands, logs and reports outside
the checkout, for example `~/.cache/ov-validation/<task>/`. Record resolved binary
and library hashes, configuration and untracked source dependencies when using
an overlay. A successful local run does not prove GHA packaging or job selection.

Retain failed attempts. Use `gh run rerun <run-id> --failed` for a deliberate retry
of transient failures; fixes require a new run at the new revision. Changed inputs
invalidate reuse. Expand or repeat only for changed inputs, a new failure or
missing required coverage. Before repeating an unchanged failing reproducer, record the
new hypothesis or diagnostic; verify the actual graph/runtime inputs before another sweep.
Stop on a user pause and retain the run IDs and logs.

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

Keep full XML, logs and inventories on disk. Return tested revisions and runtime identity,
completed/remaining cases, exact selected counts/skips, grouped failure signatures, reproduction
commands and artifact paths. Aim for at most 1,500 tokens of aggregate output per routine tool call
(including batched results) or worker report; read larger excerpts only for a concrete
investigation. Use [summarize_tests.py](../scripts/summarize_tests.py) for GTest/pytest XML:

```bash
python summarize_tests.py tests.xml --output test-summary.json --process-exit <test-exit-code>
```

The helper retains full failure/skip details in JSON and prints bounded counts/signatures.
Its exit status reports parsing success; test acceptance still requires the test process
status, required coverage and original numerical gates.

Send checkpoints for completion, failure, missing prerequisites
or a material state change. Distinguish selected skips from filtered or unregistered suites and disclose
standalone fixture adapters. Group equivalent errors without asserting a common
cause. Preserve original failures and numerical differences for diagnosis; known
arithmetic differences remain failed parity checks unless the user changes the
acceptance contract.
