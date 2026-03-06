# Skill: GPU Hardware Analysis

> Source: `temp/SKILL.md` (Sections 0 + 1)
> Agent: `gpu_agent`

## When to Use

- A model needs GPU plugin enablement or optimization.
- GPU inference fails or performance is subpar.
- Need to determine the right kernel tuning parameters before writing OpenCL code.

## Procedure

### Step 0: Acquire Hardware Specs

Run `clinfo` to gather the hardware baseline:

```bash
clinfo | grep -E "Device Name|Max compute units|Max work group size|Max sub-groups|Max sub-group size|Local memory type|Local memory size|Global memory cache size"
```

### Step 1: Analyse Parameters

| Parameter | How to use |
|-----------|-----------|
| **Device Name** | iGPU vs dGPU. If both exist, verify/validate/optimize on both |
| **Target Architecture** | Gen9 (Integrated), Xe-LP (TigerLake/AlderLake), Xe-HPG (Arc), Xe-HPC (Ponte Vecchio) |
| **EU Count (Compute Units)** | Determines if kernel needs massive parallelism or latency hiding |
| **Preferred Sub-group Size** | Check if `8`, `16`, or `32` is supported (critical for `intel_sub_group_block_read`) |
| **SLM Size** | Defines maximum tile size for blocked algorithms |
| **FP16 support** | Check for `cl_khr_fp16` extension |

### Step 2: Determine Build Mode

Based on the task, choose the build command (build dir: `./build`):

| Mode | Command | When |
|------|---------|------|
| **Debug** | `cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON .. && make -j$(nproc)` | Fixing kernel bugs, functional issues |
| **Release** | `cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON .. && make -j$(nproc)` | Performance profiling |
| **Python/Wheel** | `cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON .. && make -j$(nproc)` | Python integration testing |

## Output

Return a hardware context:

```
device_name:      <name>
architecture:     <Gen9|Xe-LP|Xe-HPG|Xe-HPC>
type:             <iGPU|dGPU|both>
eu_count:         <number>
max_subgroup:     <8|16|32>
slm_size:         <bytes>
fp16_supported:   <yes|no>
build_mode:       <Debug|Release>
simd_size:        <recommended SIMD_SIZE>
```

If no GPU hardware available → report `status=skipped` to OV Orchestrator.

Otherwise → proceed to **gpu_kernel_development** skill.
