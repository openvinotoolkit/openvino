# GPU Plugin Debugging Guide

This document describes debugging practices that can help diagnose issues in the GPU plugin.

## OpenCL Out-Of-Resource
During execution, OpenCL may return an out-of-resource (OOR) error. This can happen for two primary reasons:

* **Actual resource exhaustion**: The system runs out of resources—for example, because the workload requires more memory than is available.
* **GPU execution failures reported as OOR**: Some GPU-side failures do not have dedicated OpenCL error codes. In these cases, the driver may report the failure as an OOR error. A known (and common) example—starting from Xe2— is a page fault inside a GPU kernel. When this occurs, the GPU may report OOR even though the root cause is not memory pressure.

### Steps to Debug OOR Issues

1. **Verify Memory Consumption**
   First, assess whether memory usage is within a reasonable range.
   * Use the [OV_VERBOSE](gpu_debug_utils.md#gpu-plugin-debug-utils) logs or
   * Examine memory allocations using *CLI_CallLogging* from the [opencl-intercept-layer](https://github.com/intel/opencl-intercept-layer/)

   If memory consumption is unexpectedly high, investigate:
   * It may be expected (e.g., large model size, large intermediate buffers).
   * Or it may indicate a potential issue in the OpenVINO GPU plugin.

2. **Identify the Kernel That Triggers the OOR**
   If memory usage appears normal, the next step is locating the specific kernel that causes the failure. Use the opencl-intercept-layer with:
   * CLI_CallLogging – to record OpenCL calls
   * CLI_FinishAfterEnqueue – to force each OpenCL enqueue call to execute in a blocking manner

   With these options enabled, the OOR error will be associated with a specific enqueue call, allowing you to determine which layer caused it.

3. **Validate by Replacing the Kernel with a Dummy Implementation**
   Once you identify the problematic layer, you can try replacing the kernel with a dummy implementation. For OpenVINO kernels, you can temporarily comment out the corresponding OpenCL code.

   If the OOR disappears after replacing the kernel, this indicates the error originates from within the kernel execution itself. You can then bisect the kernel code to isolate the exact section responsible.

   > Note: When modifying or partially removing code, keep in mind that the OpenCL compiler may optimize away sections which are considered "unused." Be cautious when interpreting results after such modifications.