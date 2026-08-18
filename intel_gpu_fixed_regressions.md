# Intel GPU Fixed Regressions

## 1. Public OpenCL Device IDs Changed

- **Symptom:** C0 enumerated `GPU.ocl_0`/`GPU.ocl_1` instead of legacy
  `GPU.0`/`GPU.1`; pure Vulkan similarly exposed tagged default IDs.
- **Master result:** OCL devices are `GPU.0` and `GPU.1`.
- **Candidate-before result:** public IDs were runtime-tagged.
- **Root cause:** the runtime registry always used internal tagged identities for
  public default-runtime devices.
- **File/symbol:** `runtime_backend_registry::make_public_device_id`,
  `Plugin::resolve_device_id`, `Plugin::Plugin`.
- **Fix:** preserve untagged IDs for the configured default runtime and retain
  tagged aliases for non-default runtimes.
- **Regression test:** `public_identity_preserves_legacy_default_runtime_ids` and
  query-device runs in OCL-only, Vulkan-only and mixed builds.
- **Candidate-after result:** C1/C2 expose legacy default IDs; mixed Vulkan uses
  explicit `GPU.vulkan_0`/`GPU.vulkan_1`.
- **Performance after fix:** OCL and Vulkan production gates pass on both GPUs;
  identity resolution is not on the inference hot path.

## 2. Runtime Object ABI Heap Corruption

- **Symptom:** the C0 OCL harness aborts on both GPUs with `malloc(): invalid
  size`; Valgrind reports a constructor write beyond the allocated `ocl_engine`.
- **Master result:** identical harness passes.
- **Candidate-before result:** reproducible crash before meaningful inference.
- **Root cause:** split runtime OBJECT libraries were compiled with inconsistent
  `ENABLE_ONEDNN_FOR_GPU`, changing the C++ object layout across translation units.
- **File/symbol:** `src/plugins/intel_gpu/src/runtime/CMakeLists.txt`,
  `ov_gpu_configure_runtime_target`.
- **Fix:** link the oneDNN GPU target PUBLIC to every runtime object target so the
  ABI-affecting definition is consistent.
- **Regression test:** OCL harness, runtime registry loop and unit executable
  construction on both devices.
- **Candidate-after result:** no corruption; static, dynamic, import/cache and
  concurrent inference pass on both GPUs.
- **Performance after fix:** M/C3 OCL latency and throughput pass on both GPUs.
  The C0 mixed-production crash also independently demonstrates why an ABI-safe
  C0 mixed performance number cannot be manufactured.

## 3. Premature OCL Completion Event

- **Symptom:** activation synchronization tests returned incorrect/zero output on
  C1 before the fix while master passed.
- **Master result:** correct output.
- **Candidate-before result:** aggregate completion requested an output marker;
  an in-order OCL marker could be an already-completed user event.
- **Root cause:** `gpu_execution_plan::execute` upgraded final aggregation to an
  output marker rather than returning the actual requested dispatch completion.
- **File/symbol:** `gpu_execution_plan::execute` in
  `common_utils/gpu_execution_plan.hpp`.
- **Fix:** aggregate actual dispatch/dependency events without introducing an
  output marker.
- **Regression test:**
  `gpu_execution_plan.returns_requested_kernel_completion_without_output_marker`.
- **Candidate-after result:** repeated activation and architecture loops pass on
  both GPUs; harness checksums match master.
- **Performance after fix:** M/C3 OCL warm latency differs by +0.42% on iGPU and
  +0.19% on dGPU, both within measured baseline noise; throughput also passes.

## 4. Empty Fully Connected Execution Plan

- **Symptom:** two dynamic fully-connected tests produced zero output on the UHD
  730 OCL device; master passed.
- **Master result:** correct nonzero output.
- **Candidate-before result:** the custom constructor assigned kernel data but
  left the backend-neutral execution plan empty.
- **Root cause:** `fully_connected_impl(const kernel_data&)` bypassed the normal
  plan rebuild performed by other initialization paths.
- **File/symbol:** `fully_connected_impl` constructor in
  `graph/impls/ocl/fully_connected.cpp`.
- **Fix:** call `rebuild_execution_plan()` after storing kernel data.
- **Regression test:** existing dynamic FC cases plus the repeated architecture
  filter on each GPU.
- **Candidate-after result:** exact expected dynamic checksums on both GPUs.
- **Performance after fix:** dynamic OCL timing improves slightly on both GPUs;
  no repeatable compile, latency, or throughput regression is present.

## 5. Configuration-Specific Test Builds Failed

- **Symptom:** the OCL+oneDNN unit mock was abstract, and the Vulkan-only unit/
  functional build compiled sources that require OCL headers or symbols.
- **Master result:** OCL test configuration builds; Vulkan split is candidate-only.
- **Candidate-before result:** full test targets fail to compile/link in affected
  configurations, although production plugin targets build.
- **Root cause:** the mock lacked conditional `get_onednn_stream`; OCL-only unit
  sources and `CoreTest.smoke_singletonOclContext` were not runtime-gated.
- **File/symbol:** `gpu_execution_plan_test.cpp`, Intel GPU unit `CMakeLists.txt`,
  functional `behavior/infer_request.cpp`.
- **Fix:** implement the conditional mock method, classify OCL-only sources, and
  guard OCL includes/test code with `OV_GPU_WITH_OCL_RT`.
- **Regression test:** complete C1 OCL-only and Vulkan-only target builds.
- **Candidate-after result:** plugin, unit, functional, benchmark and sample
  targets build in both configurations; C1 mixed also completes 4243/4243.
- **Performance after fix:** test-only/configuration gating; no runtime hot-path change.

## 6. Vulkan Transient Slot Survived a Reset-Mode Transition

- **Symptom:** repeated dynamic-shape inference intermittently reported
  `Transient command resources were not released` at
  `vulkan_stream.cpp:1239`; current C2 reproduced in 3/5 initial dGPU processes
  and only 2/12 final C0 performance processes completed the dynamic sequence.
- **Master result:** not applicable; M has no equivalent Vulkan foundation path.
- **Candidate-before result:** both C0 and C2 reproduce, proving the defect is in
  the integration path rather than one of the earlier OCL recovery fixes.
- **Root cause:** a generation recycle could clear `slot.submission` while
  intentionally retaining `transient_command_buffer_submitted`. If adaptive
  tuning selected individual reset next, `recycle()` returned early solely
  because the submission was null and never reset that command buffer.
- **File/symbol:** `vulkan_stream::resource_state::recycle` in
  `src/plugins/intel_gpu/src/runtime/vulkan/vulkan_stream.cpp`.
- **Fix:** wait and release only when a submission exists, but handle the
  transient command-buffer reset independently. No wait, serialization,
  allocation, cache disable, or feature disable was added.
- **Regression test:**
  `vulkan_execution_plan.transient_slots_survive_pool_reset_tuning_transitions`
  alternates two retained dispatch/buffer sets 100 times through one stream.
- **Candidate-after result:** the module test passes 5/5 on each GPU plus clean
  Validation Layer runs; the high-level harness passes 10/10 stress processes
  on each GPU and 5/5 final C3 performance processes.
- **Performance after fix:** C0→C3 warm latency is -0.08% iGPU / +0.02% dGPU;
  throughput is -0.60% / +0.32%, all inside baseline variability. Dynamic
  medians improve slightly on both devices.

No candidate-specific correctness or performance regression remains in the
completed C3 gates. Baseline-only functional failures and the external dGPU
firmware warning remain documented in the validation report.
