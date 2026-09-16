---
name: gpu-integrate-onednn-primitive
description: Integrate oneDNN primitive into OpenVINO GPU plugin when an existing oneDNN kernel is available for the operation
---

# Purpose

Guide the implementation of a oneDNN-based primitive for the OpenVINO GPU plugin. When a requested operation already has a high-performance oneDNN primitive available, it should be leveraged instead of (or in addition to) writing a custom OpenCL kernel. oneDNN provides optimized implementations for many common deep learning operations on Intel GPUs.

# When to Use

Use this skill **after** creating and profiling the reference OpenCL kernel (`gpu-kernel-enabling` + `gpu-kernel-device-timing`), once you decide to integrate a oneDNN primitive for the operation. After integration, use `gpu-kernel-device-timing` to collect timings for both OpenCL and oneDNN paths.

```mermaid
flowchart TD
  A[Op Implementation Request] --> B[Implement & Verify Reference OpenCL Kernel<br/>(gpu-kernel-enabling)]
  B --> C[Profile Reference OpenCL Kernel<br/>(gpu-kernel-device-timing)]
  C --> D{oneDNN primitive available<br/>& beneficial?}
  D -->|Yes| E[Integrate oneDNN Primitive<br/>(this skill)]
  D -->|No| F[Keep OpenCL-only Path]
  E --> G[Add Kernel Selection & Verify<br/>(OpenCL + oneDNN)]
  F --> G
  G --> H[Collect Timing for Both Paths<br/>(gpu-kernel-device-timing)]
```

# Procedure

1. **Step 1: Review Op Plan Strategy** — Check if oneDNN integration was recommended in the `plan-op-implementation` strategy for the target Op
2. **Step 2: Create oneDNN Implementation** — Write the oneDNN-based implementation
3. **Step 3: Register in Graph** — Register the oneDNN primitive in the GPU plugin graph
4. **Step 4: Add Kernel Selection Logic** — Enable selection between OpenCL and oneDNN implementations
5. **Step 5: Verify** — Run tests to ensure correctness

---

# Prerequisites Check

Verify that oneDNN is available in the OpenVINO build:

**Windows (PowerShell):**
```powershell
# Check whether bundled oneDNN artifacts are present in the build tree
Test-Path ".\build\third-party-programs\onednn" -ErrorAction SilentlyContinue

# Or check for built dnnl artifacts
Get-ChildItem -Path .\build -Recurse -Include "dnnl*","*onednn*" -ErrorAction SilentlyContinue | Select-Object -First 5
```

**Ubuntu:**
```bash
# Check whether bundled oneDNN artifacts are present in the build tree
test -d ./build/third-party-programs/onednn && echo "OK" || echo "Check build tree"

# Or check if dnnl/oneDNN libraries exist
find ./build -name "libdnnl*" -o -name "dnnl*" 2>/dev/null | head -5
```

- **If successful:** Proceed to "Quick Start - Main Steps"
- **If failed:** Rebuild using the canonical `build-openvino` workflow and verify bundled oneDNN artifacts are generated

---

# Quick Start

## Installation (Prerequisites Check failed)

oneDNN is typically bundled as a third-party dependency in OpenVINO GPU builds. Use the `build-openvino` skill in Debug mode with tests enabled rather than relying on an ad-hoc oneDNN-specific CMake flag.

---

## Main Steps (Prerequisites Check passed)

### Step 1: Review Op Plan Strategy

Examine the `plan-op-implementation` results. Before implementing, verify that this operation was identified as a candidate for oneDNN integration.

**Common oneDNN-supported operations:**
- Convolution (`dnnl::convolution_forward`)
- MatMul / Inner Product (`dnnl::matmul`, `dnnl::inner_product_forward`)
- Pooling (`dnnl::pooling_forward`)
- Batch Normalization (`dnnl::batch_normalization_forward`)
- Eltwise (ReLU, GELU, etc.) (`dnnl::eltwise_forward`)
- Softmax (`dnnl::softmax_forward`)
- Reduction (`dnnl::reduction`)
- Reorder / Reformat (`dnnl::reorder`)
- Binary operations (`dnnl::binary`)

**Check the oneDNN API reference:**
- https://oneapi-src.github.io/oneDNN/
- Look for the primitive in `dnnl::primitive_kind`

**Existing oneDNN implementations in GPU plugin:**
```
src/plugins/intel_gpu/src/graph/impls/onednn/
```

Review existing implementations for patterns and conventions.

### Step 2: Create oneDNN Implementation

**Directory:** `src/plugins/intel_gpu/src/graph/impls/onednn/`
**File:** `<op_name>_onednn.cpp`

**Implementation pattern:**

```cpp
// Key components for a oneDNN implementation:

// 1. Create oneDNN primitive descriptor
//    - Map OpenVINO tensor descriptors to oneDNN memory descriptors
//    - Configure the primitive attributes (post-ops, scales, etc.)

// 2. Create primitive
//    - Instantiate the oneDNN primitive from the descriptor
//    - Handle memory format selection

// 3. Execute
//    - Map GPU plugin memory to oneDNN memory objects
//    - Execute the primitive
//    - Handle output memory reorder if needed
```

**Key considerations:**
- Map `cldnn` layout formats to oneDNN memory formats
- Handle data type conversions (fp32, fp16, bf16, int8)
- Support post-operation fusion when applicable
- Ensure proper memory lifecycle management

### Step 3: Register in GPU Plugin Graph

Register the oneDNN implementation so the GPU plugin can discover and use it:

```cpp
// Registration pattern (simplified):
// 1. Implement the typed_primitive_onednn_impl class
// 2. Register the implementation factory
// 3. Add layout and data type validation
```

### Step 4: Add Kernel Selection Logic

The GPU plugin needs logic to choose between OpenCL and oneDNN implementations.

**Selection criteria:**
- Hardware support (oneDNN may only support certain GPU architectures)
- Data type availability (oneDNN may support int8 quantization that OpenCL ref doesn't)
- Layout compatibility (oneDNN may prefer specific blocked layouts)
- Performance characteristics (oneDNN is often faster for standard ops)

**File to modify:** Kernel selector or implementation factory registration.

### Step 5: Functional Verification

If oneDNN-specific test cases are needed (e.g., backend selection, int8 quantization paths), invoke `write-gpu-tests` to add them first.

Then invoke `run-gpu-tests` skill to verify the oneDNN integration:
- Run unit tests and functional tests filtered to the target op
- Verify both OpenCL and oneDNN paths produce correct results

**Verification checklist:**
- [ ] All existing tests still pass (no regression)
- [ ] oneDNN path is selected when expected
- [ ] Results match the reference OpenCL kernel within acceptable tolerance
- [ ] Performance is equal or better than OpenCL reference (verify via `gpu-kernel-device-timing`)

---

# Troubleshooting

- **oneDNN primitive not found**: Verify the op is supported in the oneDNN version bundled with OpenVINO
- **Memory format mismatch**: Ensure proper mapping between `cldnn` layouts and `dnnl::memory::format_tag`
- **Accuracy regression**: Check data type handling, especially for fp16/bf16 operations
- **oneDNN path not selected**: Verify the implementation factory registration and selection priority
- **Build errors**: Verify the canonical `build-openvino` workflow completed successfully and that bundled dnnl/oneDNN artifacts were generated in the build tree
- **Runtime errors**: Check oneDNN verbose output with `ONEDNN_VERBOSE=1` environment variable

---

# References

- Related skills: `gpu-kernel-enabling`, `gpu-kernel-device-timing`, `run-gpu-tests`, `write-gpu-tests`, `gpu-op-file-structure`
- Canonical build workflow: `build-openvino`
- oneDNN API reference: https://oneapi-src.github.io/oneDNN/
- Existing oneDNN implementations: `src/plugins/intel_gpu/src/graph/impls/onednn/`
- oneDNN GPU support: https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
