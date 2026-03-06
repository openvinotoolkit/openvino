# Skill: GPU Kernel Development

> Source: `temp/SKILL.md` (Sections 2, 4, 6)
> Agent: `gpu_agent`

## Prerequisites

- Completed **gpu_hardware_analysis** skill - architecture, SIMD size, SLM size known.
- Missing op / kernel has been identified from error context.

## File Structure (Section 6)

All files follow `snake_case` for filenames, `CamelCase` for class names.

### 1. Kernel Selector Layer (Host Logic)

**Directory:** `src/plugins/intel_gpu/src/kernel_selector/kernels/<op_name>/`

| File | Purpose |
|------|---------|
| `<op_name>_kernel_selector.h/cpp` | Select between Ref, Opt, or layout-specific kernel |
| `<op_name>_kernel_base.h/cpp` | Parameter definitions (JitConstants) |
| `<op_name>_kernel_ref.h/cpp` | Reference implementation binder |
| `<op_name>_kernel_opt.h/cpp` | *(Optional)* Optimized implementation binder |

### 2. OpenCL Kernel Source (Device Code)

**Directory:** `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/`

| File | Purpose |
|------|---------|
| `<op_name>_ref.cl` | Reference kernel - straightforward, no HW-specific optimizations |
| `<op_name>_opt.cl` | Optimized kernel with sub-groups, SLM tiling, etc. |
| `<op_name>_<layout>.cl` | *(Optional)* Blocked-layout-specific kernel |

### 3. Primitive Definition

**Directory:** `src/plugins/intel_gpu/include/intel_gpu/primitives/`

| File | Purpose |
|------|---------|
| `<op_name>.hpp` | Structure definition inheriting from `primitive_base` |

### 4. Implementation Logic (Graph Registration)

**Directory:** `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`

| File | Purpose |
|------|---------|
| `<op_name>.cpp` | `create_<op_name>` + layout validation logic |

### 5. Operation Translation (Plugin Layer)

**Directory:** `src/plugins/intel_gpu/src/plugin/ops/`

| File | Purpose |
|------|---------|
| `<op_name>.cpp` | `Create<OpName>Op` function using `ProgramBuilder` |

### Example: `FillEmptyRows`

```
kernel_selector/kernels/fill_empty_rows/fill_empty_rows_kernel_selector.cpp
kernel_selector/cl_kernels/fill_empty_rows_ref.cl
primitives/fill_empty_rows.hpp
graph/impls/ocl_v2/fill_empty_rows.cpp
plugin/ops/fill_empty_rows.cpp
tests/unit/test_cases/fill_empty_rows_gpu_test.cpp
```

## Kernel Development Process

### Reference Kernel

1. Find the C++ reference kernel in:
   `src/core/reference/include/openvino/reference/<op_name>.hpp`
2. Use it as **logic reference only** - write OpenCL from scratch.
3. The reference `.cl` kernel must be straightforward:
   - No sub-group usage, no SLM tiling.
   - Clean baseline for correctness verification.

### Optimized Kernel

1. Create an optimized version incorporating HW-specific features:
   - Sub-group operations (`intel_sub_group_block_read`)
   - Local memory (SLM) tiling
   - Appropriate SIMD width based on hardware analysis
2. Conditionally select in the kernel selector based on HW capabilities.

### Opset Compatibility (Section 4)

- Compare opset versions - check if op definition changed.
- Preserve backward compatibility with legacy behavior.
- Check data type support: `bf16`/`fp16` based on `cl_khr_fp16` extension.

### Functional Verification

```bash
# Kernel dump (verify macro substitution)
export OV_GPU_DUMP_KERNELS=1
export OV_GPU_CACHE_MODEL=0

# Run tests
./bin/intel64/Debug/ov_gpu_unit_tests --gtest_filter=*<OpName>* --device_suffix=<GPU_INDEX>
./bin/intel64/Debug/ov_gpu_func_tests --gtest_filter=*<OpName>* --device_suffix=<GPU_INDEX>
```

Verify that macros derived from `clinfo` (e.g. `#define SIMD_SIZE 16`) are
correctly substituted in the dumped `.cl` file.

## Output

- All source files created/updated per the file structure above.
- Reference kernel passes functional tests.
- Proceed to **gpu_performance_profiling** skill.
