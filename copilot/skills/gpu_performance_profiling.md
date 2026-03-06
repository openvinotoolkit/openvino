# Skill: GPU Performance Profiling

> Source: `temp/SKILL.md` (Sections 3 + 5)
> Agent: `gpu_agent`

## Prerequisites

- Completed **gpu_kernel_development** skill - reference kernel passes functional tests.
- Hardware context from **gpu_hardware_analysis** (architecture, EU count, SIMD size).

## Profiling with clintercept

### Basic Profiling

```bash
clintercept -d -t -- ./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter=*<OpName>*
```

### Key Metric

- **`DeviceTotalTime`** - primary performance indicator.
- If high → check: Are we using enough EUs? Is Work-Group size too small?

## Hardware-Aware Optimization (BKM)

### A. Dynamic Sub-group Size Selection

| Hardware | `Max sub-group size` | Strategy |
|----------|---------------------|----------|
| Arc / Xe-HPC | 32 | Use `sub_group_size(32)` for compute-heavy ops; `sub_group_size(16)` if register pressure is high |
| iGPU / Xe-LP | 16 | Strictly use `sub_group_size(16)` |

```c
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))  // SIMD_SIZE from HW analysis
```

### B. Memory Layout & Block Reads (fsv16)

- Maximize bandwidth with `intel_sub_group_block_read` (reads 4 bytes × SIMD_SIZE).
- **Alignment:** discrete GPU (Arc) → align to 128 bytes for cache line prefetching.
  Minimum hard requirement: 16 bytes.

### C. Local Work Size (LWS) Tuning

- LWS must be a **multiple of SIMD_SIZE**.
- `Ideal LWS = min(Max_work_group_size, SLM_constraint)`.
- **Gen12/Arc:** prefer LWS of 256 or 512 to fully occupy XVE/EU threads.

### D. Register Pressure Management

- On smaller GPUs (fewer EUs), reducing register usage is critical.
- Use `half` (fp16) precision if `cl_khr_fp16` is supported.
- Monitor register spilling - high spill rate degrades performance significantly.

## Optimization Loop

1. Profile with `clintercept` → measure `DeviceTotalTime`.
2. Apply optimization (sub-group, SLM, LWS tuning, fp16).
3. Re-profile → compare `DeviceTotalTime`.
4. Repeat until performance is acceptable or no further optimization is possible.

## Output

- Optimized kernel (`<op_name>_opt.cl`) with HW-specific tuning.
- Benchmark results: latency (ms), throughput (FPS/inferences per second).
- Proceed to **gpu_testing** skill.
