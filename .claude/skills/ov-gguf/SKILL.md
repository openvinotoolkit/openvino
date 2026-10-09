---
name: ov-gguf
description: >
  Develop and troubleshoot the OpenVINO GGUF frontend (src/frontends/gguf) and the llama.cpp
  ggml-openvino backend that uses it. Use when checking whether a .gguf or mmproj model is
  supported; when a .gguf file is rejected as an unsupported architecture; when enabling a
  llama.cpp model family (llama, qwen, gemma, phi, MoE, hybrid/recurrent) through arch_registry,
  DecoderBuilder, a ModelBuilder or an ArchitectureExtension; when adding a vision/audio projector or
  ProjectorExtension; when conversion fails with "Translation for operation type GGML_OP_* is not
  implemented" or an op translator/op_case in src/op/ needs work; when making a converted model
  stateful or GenAI/PagedAttention-ready; or when a GGUF model converts but gives garbage, drifting
  tokens, a llama.cpp mismatch or a decode-only shape failure; or when running or resuming GGUF
  frontend, backend and GenAI validation batches. Not for general llama.cpp usage or
  unrelated build/CI failures.
---

Paths are relative to `src/frontends/gguf/`. Read only the route the task needs. First identify the
input path: a native `.gguf` file or a supplied `GgufDecoder` (llama.cpp cgraph). They share
converters; builders and validation differ. Registration or conversion alone never establishes accuracy.

| Task | Read |
|---|---|
| Is a model supported? | [supported_models.md](../../../src/frontends/gguf/docs/supported_models.md) or [mmproj.md](../../../src/frontends/gguf/docs/mmproj.md#supported-projectors); quote the checkpoint result and limitations, not just the catalog entry |
| Load a model, make it stateful/GenAI/PA-ready, tokenizer, IR export | [runtime.md](../../../src/frontends/gguf/docs/runtime.md) |
| Add or port an architecture | [architectures.md](../../../src/frontends/gguf/docs/architectures.md); [extensions.md](../../../src/frontends/gguf/docs/extensions.md) when shipping it as a plugin |
| Add or replace an mmproj branch | [extensions.md](../../../src/frontends/gguf/docs/extensions.md#extend-mmproj-with-a-projector-component), then the graph contracts in [mmproj.md](../../../src/frontends/gguf/docs/mmproj.md) |
| Missing ggml operation or `op_case` | [how_to_add_op.md](../../../src/frontends/gguf/docs/how_to_add_op.md) |
| Wrong numbers | [debugging_accuracy.md](../../../src/frontends/gguf/docs/debugging_accuracy.md) |
| GenAI/mmproj token, media or PA mismatch | [multimodal integration bisection](../../../src/frontends/gguf/docs/debugging_accuracy.md#multimodal-integration-bisection) |
| Unsupported weight type | [quantization.md](../../../src/frontends/gguf/docs/quantization.md); the fix belongs in `src/quant/`, not a builder |
| Build, run tests, regenerate fixtures | [testing.md](../../../src/frontends/gguf/docs/testing.md) |
| Run or resume GGUF validation across OpenVINO, GenAI and llama.cpp | [validation.md](references/validation.md); use its runner for stable inputs, coverage contracts and retained failures |

## Easy to miss

- **Which layer is missing.** An unclaimed `general.architecture` needs a catalog row or
  `ArchitectureExtension`; a new mmproj type needs a `ProjectorExtension`; a structurally different
  use of an existing op needs an `op_case`. Only a genuinely new GGML op needs a translator.
- **Registration timing.** Architecture/projector extensions before `load()`; converters and passes
  before `convert()`, on the same frontend instance. `GGUFMakeStateful` and `AdaptToGenAI` are C++ only.
- **Test build.** New translator or builder sources go into the explicit `FRONTEND_SRCS` list in
  `tests/CMakeLists.txt`, or the test binary fails to link. Run `ov_gguf_frontend_tests` unfiltered
  before finishing: the op-coverage gate runs in teardown, so check the exit status.
  `GGUFArchConversion` skips silently without generated headers (`tests/gen_arch_fixtures.py --fetch`).
  Check actual selected counts and skips; registration, conversion, decoder numerics and
  GenAI media/API integration establish different levels of support.
- **References.** Layout-sensitive expectations come from real ggml CPU via a committed oracle under
  `tests/`, with several heads/tokens and unequal Q/KV dimensions; NumPy or single-head tests can
  share the bug. Do not tighten `expect_near` tolerances from an x86-only run.
- **Comparisons.** Use full logits or features on identical inputs and token histories, with
  `INFERENCE_PRECISION_HINT=f32`, `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` and the reference's KV precision.
  Set `OV_GGUF_Q4_K_ZP_F16=1` before the process starts. Report quantized-CPU and represented-F32
  comparisons separately and keep failed results. `GGML_OPENVINO_*` variables configure only the
  llama.cpp backend.
  Inspect tensor-type inventory rather than checkpoint names. Identical quantized weights do
  not rule out CPU arithmetic differences; exact expansions isolate them but do not waive the
  original parity gate. Shared exported-model layout does not imply identical media geometry.
- **Shared changes.** Translator, VIEW/`op_case` or shared block changes rerun
  `GGUFArchitectureAccuracy`, `GGUFArchConversion` and the mmproj suites, not just the target model.
- **Long runs.** Redirect output to a file, keep the launched PID and stop only that process. Large Q2
  MoE checkpoints expand at CPU compilation; check memory before starting one.
- **Ineffective rebuilds.** Verify graph attributes, plugin/executor parameters and the linked,
  loaded CPU variants before repeating a sweep. Use [build and runtime identity](references/build-runtime.md)
  for overlays or multiple worktrees; version strings and `ldd` alone do not prove execution identity.
- **Validation batches.** Extend shared-path regression checks to tokenizer, mask and PA changes
  with representative SDPA/PA/media/chat cases. The [validation guide](references/validation.md)
  separates numerical parity, model quality and Optimum compatibility; a known arithmetic
  difference remains a failed parity check unless the user changes the acceptance contract.
