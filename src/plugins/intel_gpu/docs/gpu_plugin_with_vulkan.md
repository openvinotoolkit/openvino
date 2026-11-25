# Vulkan runtime for the OpenVINO GPU plugin

Select the Vulkan runtime with `-DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=VULKAN`.
The build requires Vulkan 1.3 headers and a loader. The target needs a Vulkan
driver that exposes a Vulkan 1.3 integrated or discrete GPU with a
compute-capable queue and `synchronization2`.

The GPU plugin can enumerate devices and create a default remote context.
Use that context to allocate a remote tensor and copy data between host and
device tensors. This runtime supports linear buffers and synchronous
transfers for fixed-size element types.

Model compilation, model import, and external GPU handle interop are
unavailable in this runtime.

The Vulkan foundation builds the common plugin entry point and remote tensor
code. Vulkan sources provide the diagnostics for model operations and
external contexts.

The execution follow-up will build the common graph and kernel selector with
the Vulkan implementation and SPIR-V generation. The compilation plan uses
compatible OpenCL C kernel sources as inputs to CLSPV. The Vulkan build will
select its own graph implementation sources and link the Vulkan loader.

Set `-DOV_GPU_VULKAN_SMOKE_TESTS=ON` to build and run the runtime and memory smoke tests.

```sh
cmake --build build --target ov_gpu_vulkan_runtime_smoke ov_gpu_vulkan_memory_smoke
ctest --test-dir build -R '^ov_gpu_vulkan_.*_smoke$' --output-on-failure
```
