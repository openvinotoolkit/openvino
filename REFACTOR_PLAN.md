# Refactoring Plan: "The Silicon Bridge" (Gen9 Optimization)

## Objective
Address the "Ignition Failure" (utilization drop) and maximize performance on Intel UHD 620 (Gen9) by implementing deep architectural changes derived from the *Graphics API Performance Guide* and *Ternary Optimization Report*.

## 1. The "Ignition Failure" Diagnosis
**Symptom**: GPU utilization spikes then drops to single digits.
**Root Cause**: CPU Starvation / Memory Bandwidth Saturation.
- The Python loop `schedule_next_job` is too slow to feed the hungry GPU.
- `request.set_tensor` copies data every time, saturating the memory bus (even with Unified Memory, the driver overhead is high).
- The "Steady State" queue is being drained faster than it is filled.

## 2. Refactoring Strategy

### A. The "Delta Update" Protocol (CPU Optimization)
Instead of re-uploading the entire genome (weights) for every mutation:
1.  **Persistent Tensors**: Keep `ov::Tensor` objects alive in a cache (`global_genome_tensors`).
2.  **Version Control**: Track the "version" of the genome on each request.
3.  **Delta Only**: Only upload the *changed* layer's weights.
4.  **Revert Logic**: If a mutation fails, revert only that layer on the next use of the request.
*Status: Partially implemented in `evolve_gemma_v4_steady_state.py` (pending verification).*

### B. "Zero Copy" & Memory Alignment (Memory Optimization)
Gen9 shares system DRAM. We must avoid driver-level copies.
1.  **Aligned Allocation**: Ensure all numpy arrays are aligned to 4KB page boundaries (or at least 64-byte cache lines).
2.  **Remote Blob**: Use OpenVINO's `RemoteTensor` API (if available via Python) or ensure `ov::Tensor` wraps the numpy array directly without copying.
3.  **Partial Writes**: In the OpenCL kernel, ensure we write full 64-byte chunks (float16 x 32).

### C. Kernel "Occupancy" Tuning (Compute Optimization)
Maximize the 168 hardware threads on Gen9.
1.  **Work Group Size**: Explicitly set `reqd_work_group_size(256, 1, 1)` to fit 3 groups per subslice (85% occupancy).
2.  **SIMD16**: Ensure the compiler picks SIMD16 (via `sub_group` extensions or compiler hints).
3.  **SLM Usage**: Keep SLM usage under 21KB per workgroup to allow 3 groups.

### D. Ternary Logic Implementation (ALU Optimization)
Switch from `float32` emulation of ternary logic to optimized integer ops.
1.  **Data Type**: Use `int8` for weights/signals where possible.
2.  **Instruction Selection**: Use `mad` (multiply-add) and `select` (ternary if) which map to hardware instructions.
3.  **Native Math**: Use `native_sin` instead of `sin` for the "T_WAVE" function (precision trade-off for speed).

## 3. Action Plan

1.  **Finalize Python Refactor**: Complete the "Delta Update" logic in `evolve_gemma_v4_steady_state.py`.
2.  **Kernel Update**: Verify `composite_tssn_kernel.cl` has the `reqd_work_group_size` and `native_sin` changes.
3.  **Alignment Utility**: Create a helper `aligned_zeros` / `aligned_array` in Python to ensure 4KB alignment for all input tensors.
4.  **Dry Run**: Run a "Dry Run" (no mutation, just inference loop) to verify sustained GPU utilization.

## 4. Future: "The Subagent"

If Python remains the bottleneck, we must move the "Evolutionary Loop" into C++ (the `openvino_tssn_extension.dll` itself) or a standalone C++ host application. Python is simply too slow for >60,000 FPS dispatch.

## 5. Implemented Concepts (Post-Audit)

Added Dec 4, 2025 - Based on "INTEL UHD 620 × TERNARY LOGIC" Report

### A. True FP16 Pipeline

- **Kernel**: Switched all pointers to `half*` to halve memory bandwidth usage.
- **XML**: Defined `precision="FP16"` for all custom layer inputs/outputs.
- **Compiler**: Enabled `-cl-fp16-enable` build option.

### B. Ternary Arithmetic (No-Mul)

- **Logic**: Replaced floating point multiplication with branchless addition/subtraction.
- **Implementation**:

  ```c
  // Branchless Ternary Logic
  half val = (w > 0) ? input_val : -input_val;
  sum += (w != 0) ? val : 0;
  ```

- **Benefit**: Reduces ALU pressure and energy consumption by avoiding complex FMA operations.

### C. Zero Skipping (Sparsity)

- **Logic**: Explicitly checks `if (w == 0) continue;` (or equivalent branchless select).
- **Benefit**: Exploits the ~33% natural sparsity of balanced ternary weights to skip 1/3 of all operations.

### D. Monotone Bias & Consensus

- **Monotone Functions**: Added `MIN`, `MAX`, and `CONSENSUS` (Majority Vote) activation functions.
- **Power Efficiency**: These functions are computationally cheaper than transcendental functions (SIGMOID, TANH) and align with the "Power Gating" strategy.

### E. System Audit & Benchmarking

- **Startup Check**: Added `core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")` to verify FP16 hardware support at runtime.
- **Benchmark Mode**: Added a dedicated mode to measure raw FPS without evolutionary overhead.


