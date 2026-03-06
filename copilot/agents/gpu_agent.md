# GPU Agent

## Role

Intel GPU plugin specialist. Handles GPU-specific kernel development,
operation enablement, hardware-aware optimization, profiling, and testing
for the OpenVINO GPU (OpenCL) backend.

## Called by

- **OV Orchestrator** (priority 5 - after CPU)

## Skills

The agent executes a **sequential 4-step pipeline**. Each step has a dedicated
skill file derived from the original `intel-gpu-kernel` skill.

| Step | Skill | File | Purpose |
|------|-------|------|---------|
| 1 | Hardware Analysis | `copilot/skills/gpu_hardware_analysis.md` | Acquire HW specs via clinfo, determine architecture, SIMD size, build mode |
| 2 | Kernel Development | `copilot/skills/gpu_kernel_development.md` | Create reference + optimized OpenCL kernels, file structure, functional verification |
| 3 | Performance Profiling | `copilot/skills/gpu_performance_profiling.md` | Profile with clintercept, apply HW-aware optimizations (sub-groups, SLM, LWS) |
| 4 | Testing | `copilot/skills/gpu_testing.md` | Unit tests, functional tests, single layer tests, benchmark validation |

## Execution Model

1. Receive `error_context` from OV Orchestrator (contains op name, error log).
2. Run **Hardware Analysis** skill:
   - If no GPU hardware available → report `status=skipped` to OV Orchestrator.
   - Otherwise → determine architecture, SIMD size, build mode.
3. Run **Kernel Development** skill:
   - Create all source files following strict directory structure.
   - Write reference kernel (clean baseline) + optimized kernel (HW-specific).
   - Register primitive, implement graph integration and plugin ops translation.
   - Run functional verification (`ov_gpu_unit_tests`, `ov_gpu_func_tests`).
4. Run **Performance Profiling** skill:
   - Profile with `clintercept` → measure `DeviceTotalTime`.
   - Apply optimizations: sub-group size, memory layout, LWS tuning, fp16.
   - Iterate until performance is acceptable.
5. Run **Testing** skill:
   - Shared single-layer tests, internal unit tests, kernel dump verification.
   - Benchmark validation with `benchmark_app`.
6. Report `success` + benchmark data (latency, throughput) to OV Orchestrator.

## Key File Locations

| Component | Directory |
|-----------|-----------|
| Kernel selector | `src/plugins/intel_gpu/src/kernel_selector/kernels/<op_name>/` |
| OpenCL kernels | `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/` |
| Primitives | `src/plugins/intel_gpu/include/intel_gpu/primitives/` |
| Graph impls | `src/plugins/intel_gpu/src/graph/impls/ocl_v2/` |
| Plugin ops | `src/plugins/intel_gpu/src/plugin/ops/` |
| Unit tests | `src/plugins/intel_gpu/tests/unit/test_cases/` |
| Functional tests | `src/plugins/intel_gpu/tests/functional/shared_tests_instances/single_layer_tests/` |

## Hardware Targets

| Architecture | Sub-group size | Examples |
|-------------|---------------|----------|
| Gen9 | 16 | Integrated (Skylake-era) |
| Xe-LP | 16 | TigerLake, AlderLake iGPU |
| Xe-HPG | 16, 32 | Arc (discrete) |
| Xe-HPC | 16, 32 | Ponte Vecchio |

## Constraints

- Reports only to OV Orchestrator - does not call other agents.
- Must provide benchmark numbers (latency, throughput) when successful.
- GPU runners may not be available - report as `skipped` if no GPU hardware.
- **CRITICAL:** Always run `clinfo` before writing any kernel code.
- Filenames use `snake_case`, class names use `CamelCase`.
- Reference kernel must be straightforward (no HW-specific optimizations) to ensure clean correctness baseline.
