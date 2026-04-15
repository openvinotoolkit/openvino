---
name: debug
description: Troubleshooting all sorts of failures, crashes, exceptions and errors using debug capabilities. Analyze accuracy, performance, or memory issues. Dump tensors and intermediate blobs. Serialize IRs, execution graphs. Enable verbose, logging. Profile execution. Compare layer outputs. Inspect, trace or dump transformations. Identify executed operations, nodes, primitives, kernels.
---

# Debug Skill

## Prerequisites
Many debug env vars require the build flag `-DENABLE_DEBUG_CAPS=ON`.
Before suggesting env vars, verify the build was configured with this flag
(check CMakeCache.txt in the build dir).

## Components

| Component                 | Reference file to read                |
|---------------------------|---------------------------------------|
| openvino_intel_cpu_plugin | @components/debug-intel-cpu-plugin.md |
| openvino_intel_gpu_plugin | @components/debug-intel-gpu-plugin.md |

## Steps
1. Identify the component (or multiple components), then load its reference file
2. Use the instructions and recommendations for debugging
