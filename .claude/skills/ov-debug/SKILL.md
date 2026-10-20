---
name: ov-debug
description: Debug OpenVINO CPU/GPU plugin inference, accuracy, performance, memory, and device compilation issues, or inspect graph transformations, using debug capabilities such as tensor dumps, execution graphs, profiling, and transformation tracing. Do NOT use for frontend model conversion failures (unsupported operators, translation errors, or source-model loading), build/configuration failures, or generic test/CI failures.
---

# Debug Skill

## Scope
Use this skill when CPU/GPU plugin or graph transformation debug capabilities apply to the observed symptom. Frontend conversion failures are outside its scope; use an applicable frontend skill or investigate the frontend directly. Device compilation here means compiling an OpenVINO model for CPU/GPU execution, not converting a source model into OpenVINO IR.

## Prerequisites
Build flags that enable debug capabilities (check CMakeCache.txt in the build dir):
- `-DENABLE_DEBUG_CAPS=ON` — CPU/GPU plugin debug env vars, transformation matcher logging

## Components

| Component                 | Reference file to read                | Routing hints                                                                 |
|---------------------------|---------------------------------------|-------------------------------------------------------------------------------|
| openvino_intel_cpu_plugin | [@components/debug-intel-cpu-plugin.md](components/debug-intel-cpu-plugin.md) | CPU: inference issues, wrong results, slow inference, tensor dumps, execution graphs |
| openvino_intel_gpu_plugin | [@components/debug-intel-gpu-plugin.md](components/debug-intel-gpu-plugin.md) | GPU: inference issues, wrong results, slow inference, tensor dumps, execution graphs |
| transformations           | [@components/debug-transformations.md](components/debug-transformations.md)  | transformation not applied, pass not firing, transformation tracing, slow compilation, graph inspection |

## Steps
1. Match the user's symptom to the routing hints above to identify the component(s)
2. Load the component's reference file
3. Follow the instructions and recommendations for debugging
