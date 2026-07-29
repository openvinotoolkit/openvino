# GPU plugin Level Zero support

OpenVINO GPU plugin can be compiled with **experimental** Level Zero support. When enabled, plugin will use Level Zero as GPU compute API instead of OpenCL.

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