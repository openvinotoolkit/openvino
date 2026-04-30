---
name: debug
description: Troubleshooting all sorts of failures, crashes, exceptions and errors using debug capabilities. Analyze accuracy, performance, model compilation, or memory issues. Dump tensors and intermediate blobs. Serialize and visualize IRs, execution graphs. Enable verbose, logging. Profile execution. Compare layer outputs. Inspect, trace or dump transformations. Identify executed operations, nodes, primitives, kernels.
---

# Debug Skill

## Prerequisites
Build flags that enable debug capabilities (check CMakeCache.txt in the build dir):
- `-DENABLE_DEBUG_CAPS=ON` — CPU/GPU plugin debug env vars
- `-DENABLE_OPENVINO_DEBUG=ON` — transformation matcher logging

## Components

| Component                 | Reference file to read                | Routing hints                                                                 |
|---------------------------|---------------------------------------|-------------------------------------------------------------------------------|
| openvino_intel_cpu_plugin | @components/debug-intel-cpu-plugin.md | CPU: inference issues, wrong results, slow inference, tensor dumps, execution graphs |
| openvino_intel_gpu_plugin | @components/debug-intel-gpu-plugin.md | GPU: inference issues, wrong results, slow inference, tensor dumps, execution graphs |
| transformations           | @components/debug-transformations.md  | transformation not applied, pass not firing, slow compilation, graph inspection |

## Steps
1. Match the user's symptom to the routing hints above to identify the component(s)
2. Load the component's reference file
3. Follow the instructions and recommendations for debugging
