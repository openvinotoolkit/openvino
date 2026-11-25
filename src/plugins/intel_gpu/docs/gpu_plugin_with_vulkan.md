# Vulkan runtime for the OpenVINO GPU plugin

Select the Vulkan runtime with `-DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=VULKAN`.
The build requires Vulkan 1.3 headers and a loader. The target needs a Vulkan
driver that exposes a Vulkan 1.3 integrated or discrete GPU with a
compute-capable queue and `synchronization2`.

The GPU plugin can enumerate devices and create a default remote context.
Use that context to allocate a remote tensor and copy data between host and
device tensors. This runtime supports linear buffers and synchronous
transfers for fixed-size element types.

The `GPU_MEMORY_STATISTICS` property reports live Vulkan buffer bytes under
`vk_buffer`.

Model compilation, model import, and external GPU handle interop are
unavailable in this runtime.

The Vulkan foundation builds the common plugin entry point and remote tensor
code. Vulkan sources provide the diagnostics for model operations and
external contexts.

The execution follow-up will build the common graph and kernel selector with
the Vulkan implementation and SPIR-V generation. The compilation plan uses
compatible OpenCL C kernel sources as inputs to CLSPV. The Vulkan build will
select its own graph implementation sources and link the Vulkan loader.

## macOS runtime dependencies

Vulkan Loader and MoltenVK are runtime dependencies on macOS. Provide both
through the application's package dependencies or include them in the app
bundle. Configure the application to resolve `libvulkan.1.dylib`. For a
self-contained bundle, ship MoltenVK with its driver manifest and use
paths relative to the bundle.
[Vulkan SDK deployment on macOS](https://vulkan.lunarg.com/doc/view/1.4.341.1/mac/getting_started.html)

For development from a shell, set `VULKAN_SDK` to the private SDK's `macOS`
directory. Set these variables before launching OpenVINO from that shell.

```sh
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
```

## Smoke tests

Configure the Vulkan build with `-DENABLE_TESTS=ON` to enable the smoke tests.
On macOS and Linux, CTest also checks that the installed Vulkan loader reports
no OpenVINO GPU devices when a compatible Vulkan driver is unavailable.

```sh
cmake --build build --target ov_gpu_vulkan_runtime_smoke ov_gpu_vulkan_memory_smoke
ctest --test-dir build -R '^ov_gpu_vulkan_.*_smoke$' --output-on-failure
```
