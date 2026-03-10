---
name: gpu-extend-moe3gemm-fused-compressed
description: 'Guide for extending MOE3GemmFusedCompressed with new routing pattern support in the OpenVINO GPU plugin. Use when asked to support a new MoE model in GPU plugin, add a new MoE routing variant (e.g. sigmoid-bias), extend MOE3GemmFusedCompressed, add a new MOE pattern, or debug MOE fusion failures. Covers analysis, decomposition, transformation patterns, OpenCL kernels, and test strategy.'
---

# Extending MOE3GemmFusedCompressed with New Pattern Support (GPU Plugin)

This skill describes the complete workflow for adding support for a new MoE (Mixture-of-Experts) routing pattern to the `MOE3GemmFusedCompressed` fused operation in the OpenVINO GPU plugin.

The reference implementation — adding Sigmoid+Bias routing support — was merged at commit [`b3175eb`](https://github.com/openvinotoolkit/openvino/commit/b3175eb70da72304f035045f5a5d23ea628b45c2). It serves as a concrete example of the end-to-end change set required for this type of work. When in doubt about what a specific file change should look like (new enum values, pattern branch, kernel stage, test parametrization), fetch the diff of that commit and use it as a reference.

## When to Use This Skill

- A new model uses an MoE routing pattern not yet fused by `FuseMOE3GemmCompressed`
- You need to add a new routing variant (e.g. sigmoid+bias)
- The GPU plugin falls back to unfused per-expert execution for a model that should use the optimized fused path
- You see `MOECompressed` nodes in the execution graph instead of `moe_3gemm_fused_compressed`

## Prerequisites

### Required Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `MODEL_PATH` | Yes | Absolute path to the MoE LLM model `.xml` (OpenVINO IR format). The model must exhibit the routing pattern you are trying to support. Used in all `benchmark_app` commands (Steps 1–2). |

Collect this from the user before starting. Steps 1–2 cannot be executed without it.

> **Device requirement:** `FuseVectorizedMOE3GEMM` (and the entire 3-GEMM fused pipeline) is gated on `device_info.supports_immad` — it is disabled at runtime on non-systolic-array architectures. All steps in this skill require a GPU with IMMAD support.

### Build Environment Setup

Helper scripts live in [`scripts/`](scripts/) and auto-detect the OpenVINO source root from their location. Override `SRC_DIR` or `BUILD_DIR` if needed.

#### 1. CMake Configuration

Run once (or after `CMakeLists.txt` changes):

```bash
./scripts/configure.sh
```

Key flags enabled: `ENABLE_INTEL_GPU`, `ENABLE_TESTS`, `ENABLE_DEBUG_CAPS`, `ENABLE_GPU_DEBUG_CAPS`.

#### 2. Build

Pass the CMake target name directly:

```bash
./scripts/build.sh openvino_intel_gpu_plugin   # to run a MoE model
./scripts/build.sh ov_gpu_unit_tests            # GPU unit tests
./scripts/build.sh ov_gpu_func_tests            # GPU functional tests
```

For multiple targets, build them in sequence or as a single `cmake --build` call.

#### 3. Run Tests

```bash
# Kernel-level accuracy tests (requires IMMAD GPU)
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*moe_3gemm*'

# Fusion/transformation tests only (no GPU hardware needed)
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*FuseMOE3GemmCompressed*'

# Full subgraph functional tests (requires IMMAD GPU)
./scripts/run_tests.sh ov_gpu_func_tests --gtest_filter='*MoE3GemmCompressed*'
```

#### 4. Run a Model with benchmark_app

```bash
<build_dir>/bin/benchmark_app \
    -m <path_to_model.xml> \
    -d GPU \
    -niter 1 \
    -inference_precision f16 \
    -hint latency \
    -data_shape "input_ids[1,1],input_ids_1[1,1],793[1,1,2048],647[1,1,2048]" \ # note: data shape or input names may be different
```

## Architecture Overview

`MOE3GemmFusedCompressed` is a GPU-plugin internal operation produced by the `FuseMOE3GemmCompressed` transformation. At runtime it is executed entirely by an OpenCL kernel implementation (`moe_3gemm_swiglu_fuse.cl`) which handles routing, expert gather/scatter, gate/up projections, SwiGLU activation, and down projection in a single fused dispatch.

### Transformation Pipeline

The four passes that form the compressed MoE 3-GEMM pipeline run in this order inside `TransformationsPipeline::apply()` (see `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp`):

```
Input IR (standard MoE ops)
        │
        ▼
1. FuseVectorizedMOE3GEMM          [ov::pass::FuseVectorizedMOE3GEMM]
   Fuses the three expert GEMMs (gate/up/down) and the SwiGLU
   activation into a single internal ov::op::internal::MOE node.
   ⚠ Disabled on non-IMMAD platforms (callback returns true).
        │
        ▼
2. ConvertMOEToMOECompressed       [ov::intel_gpu::ConvertMOEToMOECompressed]
   Converts MOE → MOECompressed when expert weights are
   weight-compressed (u4/u8 with scale/zp). Attaches decompression
   metadata to the config.
        │
        ▼
3. FuseMOE3GemmCompressed          [ov::intel_gpu::FuseMOE3GemmCompressed]
   Pattern-matches the routing subgraph (Softmax or Sigmoid+Bias)
   in front of MOECompressed and replaces the whole subgraph with
   a single MOE3GemmFusedCompressed node.
   ← THIS IS THE PASS YOU EXTEND FOR A NEW ROUTING PATTERN →
        │
        ▼
5. KeepMOE3GemmConstPrecision      [ov::intel_gpu::KeepMOE3GemmConstPrecision]
   Marks the newly constant-folded weight constants so that a
   subsequent ConvertPrecision pass does not change their dtype
   (u4/u8 must stay as-is for the kernel). Note: it is needed only for u4 weights
```

## Workflow

- [ ] Step 1: Run model with matcher logging — see [workflow-diagnosis.md](./references/workflow-diagnosis.md#step-1-run-with-matcher-logging)
- [ ] Step 2: Analyze logs (routes to Step 3 or Step 5) — see [workflow-diagnosis.md](./references/workflow-diagnosis.md#step-2-analyze-the-logs)
- [ ] Step 3: Reproduce transformation failure as a unit test — see [workflow-transformation.md](./references/workflow-transformation.md#step-3-create-test-cases-from-the-new-model)
- [ ] Step 4: Extend the transformation until unit tests pass — see [workflow-transformation.md](./references/workflow-transformation.md#step-4-extend-the-transformation-and-make-tests-pass)
- [ ] Step 5: Extend the end-to-end functional test — see [workflow-testing.md](./references/workflow-testing.md#step-5-extend-the-end-to-end-functional-test)
- [ ] Step 6: Fix the OpenCL kernel (only if Step 5 has accuracy failures) — see [workflow-testing.md](./references/workflow-testing.md#step-6-fix-the-opencl-kernel)
- [ ] Step 7: Done — verify all acceptance criteria below

## Step 7: Done

All of the following should hold:

- [ ] Matcher log (Step 1) shows all four passes firing on the real model
- [ ] `moe_3gemm_fused_compressed` appears in the runtime model
- [ ] `FuseMOE3GemmCompressed` unit test passes for all routing types (Step 4)
- [ ] `smoke_MoE3GemmCompressedFusion` functional test passes for all routing types and weight precisions (Step 5)

## File Modification Checklist

| Area | Files | Purpose |
|------|-------|---------|
| **Test builders** | `common_test_utils/.../moe_builders.hpp` / `.cpp` | New routing subgraph builder |
| **Internal Op** | `intel_gpu/op/moe_compressed.hpp` | `RoutingType` enum value |
| **Internal Op** | `intel_gpu/op/moe_3gemm_fused_compressed.hpp` / `.cpp` | New inputs, validation |
| **Op serialization** | `transformations/op/moe_compressed.cpp` | `EnumNames`, `visit_attributes` |
| **Fusion transformation** | `transformations/fuse_moe_3gemm_compressed.cpp` / `.hpp` | Pattern branch via `Or` |
| **Const precision** | `transformations/keep_moe_3gemm_const_precision.cpp` | Handle new input count |
| **Graph registration** | `plugin/ops/moe.cpp` | Dynamic input validation |
| **Kernel indices** | `impls/ocl_v2/moe/moe_3gemm_base.hpp` | New input enum values |
| **Kernel impl** | `impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` | Stage registration, dispatch |
| **OpenCL kernel** | `impls/ocl_v2/moe_3gemm_swiglu_fuse.cl` | New routing kernel stage |
| **Kernel unit tests** | `tests/unit/test_cases/moe_3gemm_gpu_test.cpp` | Kernel accuracy (routing reference + parametrize) |
| **Kernel test data** | `tests/unit/test_cases/moe_3gemm_test_data.h` | Reference outputs for new routing type |
| **Transf. tests** | `tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp` | Fusion correctness |
| **Functional tests** | `tests/functional/subgraph_tests/dynamic/moe.cpp` | End-to-end |
| **Skip config** | `tests/functional/.../skip_tests_config.cpp` | Platform gating |

## Step 8: Debug Capabilities

See `src/plugins/intel_gpu/docs/gpu_debug_utils.md` for the full reference. Most relevant capabilities for MoE kernel work:

- **`OV_GPU_DUMP_GRAPHS_PATH`** — verify `moe_3gemm_fused_compressed` is present in the compiled GPU graph
- **`OV_GPU_DUMP_TENSORS_PATH`** — inspect intermediate tensor values when a kernel produces wrong results (Step 6)
- **`OV_GPU_DUMP_SOURCES_PATH`** — inspect the expanded OpenCL kernel source to confirm a new JIT branch compiled correctly (Step 6)

## Troubleshooting

| Issue | See |
|-------|-----|
| `FuseVectorizedMOE3GEMM` does not fire despite expected MoE subgraph | [Known Issues → MatMul attribute mismatch](./references/workflow-diagnosis.md#known-issues) |

## References

### Skill Workflow Files

- [workflow-diagnosis.md](./references/workflow-diagnosis.md) — Steps 1–2: matcher logging setup and log analysis
- [workflow-transformation.md](./references/workflow-transformation.md) — Steps 3–4: transformation unit tests and extension
- [workflow-testing.md](./references/workflow-testing.md) — Steps 5–6: end-to-end functional tests and OpenCL kernel fixes

### GPU Plugin Documentation

- [`gpu_debug_utils.md`](../../../../src/plugins/intel_gpu/docs/gpu_debug_utils.md) — Debug environment variables (`OV_GPU_DUMP_*`, `OV_VERBOSE`)
- [`gpu_kernels.md`](../../../../src/plugins/intel_gpu/docs/gpu_kernels.md) — KernelSelector architecture for implementing a new routing `KernelGenerator`
- [`gpu_plugin_unit_test.md`](../../../../src/plugins/intel_gpu/docs/gpu_plugin_unit_test.md) — Unit test directory structure (`fusions/`, `test_cases/`, `module_tests/`)
- [`graph_optimization_passes.md`](../../../../src/plugins/intel_gpu/docs/graph_optimization_passes.md) — cldnn graph optimization pass sequence (background for how GPU-level fusion interacts with OV-level passes)

### Transformation Infrastructure

- [`matcher_logging.md`](../../../../src/common/transformations/docs/debug_capabilities/matcher_logging.md) — `OV_MATCHER_LOGGING` / `OV_MATCHERS_TO_LOG` usage

### Reference Implementation

- Commit [`b3175eb`](https://github.com/openvinotoolkit/openvino/commit/b3175eb70da72304f035045f5a5d23ea628b45c2) — Complete diff for Sigmoid+Bias routing support; use as a template for any new routing type

