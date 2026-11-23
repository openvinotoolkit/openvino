# Vulkan runtime for the OpenVINO GPU plugin

Select the Vulkan runtime with `-DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=VULKAN`.
The build requires Vulkan 1.3 headers and a loader. The target needs a Vulkan
driver that exposes a Vulkan 1.3 integrated or discrete GPU with a
compute-capable queue and `synchronization2`.

The GPU plugin can enumerate devices and create a default remote context.
Use that context to allocate a remote tensor and copy data between host and
device tensors. This runtime supports linear buffers and synchronous
transfers.

Model compilation, model import, and external GPU handle interop are
unavailable in this runtime.

Set `-DOV_GPU_VULKAN_SMOKE_TESTS=ON` to build and run the runtime smoke test.

```sh
cmake --build build --target ov_gpu_vulkan_runtime_smoke
ctest --test-dir build/src/plugins/intel_gpu/tests/vulkan -R '^ov_gpu_vulkan_runtime_smoke$' --output-on-failure
```
