# Workflow: Testing

## Step 5: Extend the End-to-End Functional Test

Once the transformation pipeline fires correctly on the real model, extend `MoE3GemmCompressedFusionTest` (see `src/plugins/intel_gpu/tests/functional/subgraph_tests/dynamic/moe.cpp`) to cover the new routing type.

Add the new `MoERoutingType` value to the `routing_types` vector in the instantiation:

```cpp
const std::vector<MoERoutingType> routing_types = {
    MoERoutingType::SOFTMAX,
    MoERoutingType::SIGMOID_BIAS,  // ← new
};
```

The test already:
- Compiles the full model on GPU with f16 inference precision
- Validates numerical accuracy against CPU reference
- Asserts `moe_3gemm_fused_compressed` is present in the runtime model

Run it:

```bash
./scripts/build.sh ov_gpu_func_tests
./scripts/run_tests.sh ov_gpu_func_tests --gtest_filter='*smoke_MoE3GemmCompressedFusion*'
```

- If the test **passes** → proceed to **Step 7** (done)
- If the test **fails with accuracy errors** → proceed to **Step 6**

## Step 6: Fix the OpenCL Kernel

Accuracy failures in `MoE3GemmCompressedFusionTest` indicate the kernel stage for the new routing type is missing or incorrect. Always fix at the kernel-level unit test layer first — do not iterate directly on the slower e2e test.

### 6.1 Reproduce in Kernel-Level Unit Tests

> **Reference:** [`gpu_plugin_unit_test.md`](../../../../src/plugins/intel_gpu/docs/gpu_plugin_unit_test.md) — explains the `fusions/`, `test_cases/`, and `module_tests/` subdirectory structure under `tests/unit/`.

**File:** `src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp`

Add a test case for the new routing type:
1. Add a C++ reference implementation of the new routing logic (e.g. `run_reference_sigmoid()`) that computes expected outputs in float
2. Parametrize the existing tests by routing type and add the new variant
3. Add reference output data for hardcoded weight cases (see `moe_3gemm_test_data.h`)
4. Adjust numerical tolerance if needed — new routing types may diverge slightly in f16

```bash
./scripts/build.sh ov_gpu_unit_tests
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*moe_3gemm*'
```

The new test case is **expected to fail**, confirming the kernel gap is reproduced at unit level.

### 6.2 Fix the Kernel

> **Reference:** [`gpu_kernels.md`](../../../../src/plugins/intel_gpu/docs/gpu_kernels.md) — explains the KernelSelector architecture: `kernel_selector` base class, params descriptor, and per-kernel selection heuristics that the new routing `KernelGenerator` must follow.

**Files:**
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` — stage registration and dispatch by routing type
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl` — OpenCL routing kernel stage

Key points:
- Create a new `KernelGenerator` subclass for the routing stage
- Guard the `.cl` kernel block with a new JIT constant (e.g. `#elif SIGMOID_BIAS_TOPK_ENABLE`)
- Handle both f16 and f32 paths
- Fix shape helpers if needed (e.g. `get_seq_len()` for inputs with rank ≠ 4)

Rebuild and run the unit tests after each change:

```bash
./scripts/build.sh ov_gpu_unit_tests
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*moe_3gemm*'
```

Only proceed once **all kernel-level unit tests pass**.

### 6.3 Re-run the End-to-End Test

```bash
./scripts/build.sh ov_gpu_func_tests
./scripts/run_tests.sh ov_gpu_func_tests --gtest_filter='*smoke_MoE3GemmCompressedFusion*'
```

Return to Step 6.1 if accuracy issues persist; return to Step 5 once the test passes.
