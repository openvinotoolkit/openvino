# Coverage Expansion Analysis and Plan (GPU/NPU/NVIDIA)

## Current State (from repo)

- Current coverage workflow (`.github/workflows/coverage.yml`) is CPU-oriented and explicitly excludes unsupported hardware tests in that job.
- Dedicated Intel GPU tests already exist and are used in CI:
  - binaries: `ov_gpu_unit_tests`, `ov_gpu_func_tests`
  - reusable workflow: `.github/workflows/job_gpu_tests.yml`
- Existing CI already has a pattern for GPU runners (`iGPU`, `dGPU`) in `.github/workflows/ubuntu_22.yml` via `job_gpu_tests.yml`.
- Existing CI already has a pattern for NVIDIA plugin build in `.github/workflows/ubuntu_22.yml`:
  - clone `openvino_contrib`
  - configure/build `modules/nvidia_plugin`
  - targets `ov_nvidia_func_tests`, `ov_nvidia_unit_tests`
- Frontend tests often apply `--gtest_filter=-*IE_GPU*` in current workflows (and ONNX `add_test(...)` default), so enabling IE_GPU paths should be treated as an explicit scope increase and likely needs separate jobs/filters.

## Direct Answers

### 1) Can we add IE_GPU paths?
Yes. With iGPU/dGPU self-hosted runners, you can add a separate GPU coverage/test job where IE_GPU cases are enabled (or less restricted).

Recommended approach:
- Keep current CPU coverage job stable.
- Add dedicated GPU jobs and relax/remove `-IE_GPU` filters there.
- Start with smoke-only filters, then expand.

### 2) Do we have ov_gpu tests?
Yes.
- `ov_gpu_unit_tests`
- `ov_gpu_func_tests`

These are first-class binaries with an existing reusable workflow (`job_gpu_tests.yml`) that already handles driver/runtime setup and gtest-parallel execution.

### 3) What is needed for `ov_nvidia_func_tests`?
Yes, for meaningful functional execution you need NVIDIA GPU access.

Minimum practical requirements:
- `openvino_contrib` checkout (NVIDIA plugin lives there).
- OpenVINO developer package artifacts (for `OpenVINODeveloperPackage_DIR`).
- CUDA-capable environment (the CI pattern uses NVIDIA CUDA container).
- NVIDIA driver + container runtime/device exposure on runner (`/dev/nvidia*`).
- Build and then run `ov_nvidia_func_tests` on that GPU-enabled runner.

Without NVIDIA GPU, you may compile some targets but cannot reliably run NVIDIA functional tests.

## Expansion Plan (Decision-Complete)

## Phase 1: Split by hardware class

1. Keep `coverage.yml` CPU job as baseline coverage producer.
2. Add new workflow or extend current workflow with separate jobs:
   - `Coverage_iGPU`
   - `Coverage_dGPU`
   - `Coverage_NPU`
   - `Coverage_NVIDIA` (optional, gated)

Use self-hosted labels and container/device options similar to `ubuntu_22.yml` + `job_gpu_tests.yml`.

## Phase 2: GPU test scope increase

For iGPU/dGPU jobs:
1. Run `ov_gpu_unit_tests` with curated exclusion list (reuse `job_gpu_tests.yml` logic initially).
2. Run `ov_gpu_func_tests` with `*smoke*` filter first.
3. Add selected previously-excluded IE_GPU frontend paths in separate steps:
   - ONNX/TensorFlow frontend tests with adjusted filters.
4. Keep per-test continue behavior and always publish test artifacts.

## Phase 3: NPU job

1. Enable build with `-DENABLE_INTEL_NPU=ON` in the NPU-specific job.
2. Run:
   - `ov_npu_unit_tests`
   - `ov_npu_func_tests` (start with smoke/skip config)
3. Provide NPU runtime prerequisites and explicit environment:
   - `IE_NPU_TESTS_DEVICE_NAME`, optional skip config (`OV_NPU_TESTS_SKIP_CONFIG_FILE`).
4. Gate job on runner label availability and keep it non-blocking initially.

## Phase 4: NVIDIA plugin job

1. Checkout `openvino_contrib` in job.
2. Configure with `OpenVINODeveloperPackage_DIR` and build `ov_nvidia_func_tests`/`ov_nvidia_unit_tests`.
3. Execute tests on NVIDIA-enabled runner (GPU required).
4. Start as optional/nightly or non-blocking PR job until stable.

## Phase 5: Coverage reporting strategy

1. Keep CPU lcov report as primary required check.
2. GPU/NPU/NVIDIA jobs publish:
   - test XML/json artifacts
   - optional separate lcov reports if instrumentation is enabled and stable.
3. After stabilization, optionally merge multi-job coverage (CPU+GPU/NPU) in a final aggregation job.

## Risk Controls

- Runtime explosion: keep smoke filters first.
- Hardware flakiness: mark new hardware jobs non-blocking initially.
- Driver drift: pin runtime driver/container versions (as done in `job_gpu_tests.yml`).
- Debuggability: always upload per-job test artifacts and concise summaries.

## Concrete First Increment (recommended)

1. Reuse `job_gpu_tests.yml` in a new coverage-adjacent workflow for iGPU only.
2. Add one extra ONNX frontend run in that job with reduced IE_GPU exclusions.
3. Add NPU unit-tests-only job on NPU runner.
4. Add NVIDIA build+run smoke job behind manual trigger (`workflow_dispatch`) first.

