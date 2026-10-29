# GPU plugin Level Zero support

OpenVINO GPU plugin can be compiled with **experimental** Level Zero support. When enabled, plugin will use Level Zero as GPU compute API instead of OpenCL.

Level Zero is the better-performing runtime on Xe2 and newer Intel GPU architectures; on pre-Xe2 GPUs, OpenCL remains the better choice.

## Requirements

The target system needs an Intel GPU driver that provides the Level Zero packages (`intel-level-zero-gpu`, `level-zero` on Linux), as described in the [GPU configuration guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html).

## How to build with Level Zero support

1. Configure cmake with the following additional option:
    - `-DGPU_RT_TYPE=ZE`
2. Build OpenVINO
    - `cmake --build . --config Release`

## How to check what runtime is used
### With [DEBUG_CAPS cmake option](gpu_debug_utils.md)
0. Prerequisite: Build OpenVINO with [DEBUG_CAPS cmake option](gpu_debug_utils.md)
1. Set `OV_VERBOSE=1` environment variable.
2. Run workload
3. Find what variant of stream is reported in the logs (`ze_stream` or `ocl_stream`)
    - Example output for Level Zero runtime: `ze_stream: [GPU] Created Level Zero stream`
### Without DEBUG_CAPS
* Check which libraries are loaded at runtime.
  * On Linux, use `strace` to confirm that Level Zero libraries are loaded.
  * On Windows, use `Process Explorer` to confirm that Level Zero libraries are loaded.
  * If you see Level Zero libraries such as `ze_loader.dll` and `ze_intel_gpu64.dll`, Level Zero is in use.
* Use [clintercept](https://github.com/intel/opencl-intercept-layer/tree/main) to check whether any OpenCL calls are made.
  * If no OpenCL calls appear, Level Zero is in use.

## Limitations

* This is an experimental, single-runtime build: the resulting plugin binary talks to the GPU over Level Zero only and cannot fall back to OpenCL.
* To ship both OpenCL and Level Zero runtimes side by side, with automatic per-device selection, see [GPU dual-runtime dispatch (COMBINED build)](gpu_dual_runtime_dispatch.md) instead.
* For OpenCL-based remote context/tensor interop while running on the Level Zero runtime, see [Level Zero - OpenCL interoperability](blog/ze_ocl_interop_2026_3.md).