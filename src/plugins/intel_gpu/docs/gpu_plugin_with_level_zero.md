# GPU plugin Level Zero support

OpenVINO GPU plugin can be compiled with **experimental** Level Zero support. When enabled, plugin will use Level Zero as GPU compute API instead of OpenCL.

## Remote tensor sample

The [Hello Remote Tensor C++ sample](../../../../samples/cpp/hello_remote_tensor/README.md)
demonstrates how to integrate OpenVINO into an existing Level Zero application: GPU input and
output buffers are shared with Level Zero and data transfers are done with Level Zero. It is
also listed in the
[OpenVINO samples](../../../../docs/articles_en/get-started/learn-openvino/openvino-samples/hello-remote-tensor.rst).
The plugin does not expose a dedicated Level Zero remote context, so the sample relies on
Level Zero - OpenCL interoperability (LEO): it converts the plugin's default OpenCL context to
the underlying Level Zero context with a single `clGetContextInfo(..., CL_L0_CONTEXT_HANDLE, ...)`
query. This works with any plugin runtime (`OCL`, `ZE`, `COMBINED`). The `ZE` context type is
an edge case for the Level Zero plugin running on a driver without LEO; the sample then uses its
native handle directly.

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