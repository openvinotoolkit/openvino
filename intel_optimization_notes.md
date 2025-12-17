# Intel Gen9 / UHD 620 Optimization Notes

## 1. Architecture Overview (Gen9)
*   **Device**: Intel UHD Graphics 620 (Gen9.5 architecture derivative).
*   **Execution Units (EUs)**: Typically 24 EUs in a single slice configuration (GT2).
*   **Threads**: 7 hardware threads per EU.
    *   Total Threads: 24 EUs * 7 threads = **168 hardware threads**.
*   **SIMD Widths**: Supports SIMD-8, SIMD-16, and SIMD-32.
    *   SIMD-16 is often the sweet spot for compute kernels.
    *   Max concurrent kernel instances: 168 threads * 32 (SIMD-32) = 5376 instances.

## 2. Memory Hierarchy
*   **Unified Memory Architecture**: Shares system DRAM with CPU. Enables **Zero Copy** buffer transfers (no PCIe bus bottleneck).
*   **L3 Cache**:
    *   **768 KB per slice** total.
    *   Typically **512 KB** allocated for application data cache.
    *   Bandwidth: 64 bytes/cycle read & write per slice.
*   **Shared Local Memory (SLM)**:
    *   **64 KB per subslice** (8 EUs).
    *   Highly banked for efficient scatter/gather.
    *   Critical for inter-thread communication within a workgroup.
*   **Register File (GRF)**:
    *   128 registers per thread.
    *   Each register is 32 bytes (SIMD-8 x 32-bit).
    *   Total GRF: 4 KB per thread / 28 KB per EU.

## 3. Compute Shader Optimizations
*   **Thread Occupancy**:
    *   Target enough threads to hide memory latency.
    *   Avoid excessive register usage (spilling) which limits the number of active threads.
*   **Bank Conflicts**: Access SLM with stride-1 patterns where possible to avoid bank conflicts.
*   **Subgroups**: Use subgroup operations (SIMD shuffles) to exchange data between lanes without going to SLM.

## 4. FP16 / Half-Precision Performance
*   **Native Support**: Gen9 introduced improved native FP16 support.
*   **Throughput**:
    *   FP32: 16 ops/cycle per EU.
    *   FP16: **32 ops/cycle per EU** (2x throughput).
*   **Recommendation**: Aggressively convert weights and activations to FP16.
    *   Use `f16` types in OpenCL/Compute Shaders.
    *   OpenVINO `FP16` IR format is preferred.

## 5. Ternary Logic & "Cyberspore" Specific Optimizations
*Based on `INTEL_UHD620_TERNARY_OPTIMIZATION_REPORT.pdf`*

*   **Ternary Advantage**: Ternary logic on Gen9 can yield **2.8-3.0x speedup**.
*   **Key Mechanisms**:
    1.  **FP16 Acceleration**: Leveraging the 2x throughput of FP16 for ternary operations.
    2.  **Zero-Skipping Sparsity**: 1.5x multiplicative gain by skipping zero values in ternary weights {-1, 0, +1}.
    3.  **Monotone Power Optimization**: 20% lower power consumption allows for higher sustained clock speeds.
*   **Ternary Primitives**:
    *   `TERN_MUL(a, b)`: Implemented via look-up tables (0 cost vs expensive FP32 MUL).
    *   `TERN_ADD/SUB`: Simple ADD/SUB instructions.
*   **Strategy**:
    *   Quantize weights to Balanced Ternary `{-1, 0, +1}`.
    *   Use custom OpenVINO kernels (like the `CompositeTSSN` layer observed in logs) to implement these primitives.

## 6. Critical Recommendations for Evolutionary Algorithms
1.  **Saturate the EUs**: Ensure your population size / batch size is large enough to spawn at least ~5000 threads (SIMD-32) to fully hide latency.
2.  **Minimize Data Movement**: Keep the "genome" (weights) in L3 or SLM if possible during mutation steps.
3.  **Use FP16/Ternary**: The evolutionary evaluation phase (inference) should strictly use FP16 or Ternary kernels to maximize throughput on the UHD 620.

## 7. Hidden Gems from Graphics API Performance Guide (Gen9)
*Based on graphics-api-performance-guide-2-5.pdf*

### 7.1 Compute Dispatch & Occupancy
*   **Hardware Threads**: Gen9 has **56 hardware threads per subslice**.
*   **Occupancy Formula**: 
    HW Threads per Group = (GroupDimX * GroupDimY * GroupDimZ) / SIMD Width
    *   **Goal**: Ensure enough groups fit on a subslice to utilize all 56 threads.
    *   **Example**: A 16x16x1 group (256 threads) at SIMD16 uses 16 HW threads. 56 / 16 = 3.5. So 3 groups fit, utilizing 3 * 16 = 48 threads (85% occupancy).
*   **SLM Limit**: **64KB per subslice**.
    *   If a workgroup uses > 21KB of SLM, you drop to 2 groups per subslice (max).
    *   If > 32KB, you drop to 1 group (poor occupancy).

### 7.2 Memory & Caching
*   **Partial Writes**: **Avoid partial cache-line writes**. Always try to write full 64-byte chunks (e.g., float16 x 32 or float32 x 16) to maximize bandwidth.
*   **Structure Padding**: Pad structures to non-multiples of 16 bytes to avoid bank collisions in Shared Local Memory (SLM).
*   **Root Constants**: Use Root Constants (if accessible via low-level config) for high-frequency updates like mutation rate instead of constant buffers.

### 7.3 Power & SIMD
*   **Sporadic SIMD**: Avoid mixing scalar code with rare SIMD instructions. The voltage increase for SIMD will throttle frequency, hurting the scalar parts. **Go full SIMD or stay scalar.**
*   **Power vs Frame Rate**: Unconstrained frame rates (or inference loops) increase power but maximize utilization. For Metabolic War, we accept the power cost for speed.

### 7.4 Zero Copy is Real
*   **Unified Memory**: Gen9 shares system DRAM. Ensure OpenVINO input/output tensors are allocated in a way that allows **Zero Copy** (avoiding CPU<->GPU copies).
    *   Use ov::Tensor with host pointers aligned to 4KB page boundaries if possible (though OpenVINO handles this mostly).

