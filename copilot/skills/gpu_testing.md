# Skill: GPU Testing

> Source: `temp/SKILL.md` (Section 6 - Verification & Tests)
> Agent: `gpu_agent`

## Prerequisites

- Completed **gpu_kernel_development** (reference kernel) and
  **gpu_performance_profiling** (optimized kernel).

## Test Categories

### 1. Functional Single Layer Tests (Shared)

**Directory:** `src/plugins/intel_gpu/tests/functional/shared_tests_instances/single_layer_tests/`

| File | Purpose |
|------|---------|
| `<op_name>.cpp` | Instantiation of shared OpenVINO layer tests |

These tests validate the op through the standard OpenVINO test infrastructure,
ensuring consistency with other plugin implementations (CPU, Template).

### 2. Unit Tests (Internal)

**Directory:** `src/plugins/intel_gpu/tests/unit/test_cases/`

| File | Purpose |
|------|---------|
| `<op_name>_gpu_test.cpp` | gtest-based tests for edge cases, memory layouts |

Cover:
- All supported element types (`f32`, `f16`, `bf16`, `i32`, `i8`, etc.)
- All supported layouts (planar, blocked / fsv16, fsv32)
- Edge cases: empty tensors, scalar inputs, large shapes
- Both reference and optimized kernel paths
- Multi-GPU scenarios (if applicable): `--device_suffix=0` / `--device_suffix=1`

### 3. Kernel Dump Verification

```bash
export OV_GPU_DUMP_KERNELS=1
export OV_GPU_CACHE_MODEL=0
# Run any test and inspect the dumped .cl file
# Verify macro substitution (SIMD_SIZE, BLOCK_SIZE, etc.)
```

### 4. Benchmark Validation

```bash
# Release build required
./bin/intel64/Release/benchmark_app \
  -m ov_model/openvino_model.xml \
  -d GPU \
  -niter 100 \
  -api async
```

Capture: latency (median, p99), throughput (FPS).

## Execution

```bash
# Unit tests
./bin/intel64/Debug/ov_gpu_unit_tests --gtest_filter=*<OpName>*

# Functional tests
./bin/intel64/Debug/ov_gpu_func_tests --gtest_filter=*<OpName>*

# Single layer tests
./bin/intel64/Debug/ov_gpu_func_tests --gtest_filter=*single_layer*<OpName>*
```

## Output

- All tests pass → report `success` + benchmark data to OV Orchestrator.
- Test failures → fix kernel, re-run. Report `failed` + error details if
  the fix requires changes outside GPU plugin scope.
