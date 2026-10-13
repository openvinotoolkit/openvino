---
sidebar_position: 1
sidebar_label: "Remote Tensor API of NPU Plugin"
description: "The Remote Tensor API of NPU plugin in OpenVINO™ supports interoperability with existing native APIs, such as NT handle, or DMA-BUF file descriptor, and provides mechanisms for mapping files into memory for efficient data access."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Remote Tensor API of NPU Plugin

The NPU plugin supports memory sharing between OpenVINO and native APIs such as OpenCL, Vulkan, or DirectX 12.
It implements the `ov::RemoteContext` and `ov::RemoteTensor` interfaces, providing mechanisms for efficient memory sharing.
On Windows, the plugin exports an NT handle; on Linux, it uses a DMA-BUF file descriptor. You can share this memory by
passing the pointer as the `shared_buffer` member to the `remote_tensor(..., shared_buffer)` create function.
Another option is to import memory by mapping a file into memory or by using a CPU virtual address allocation. These methods
help avoid memory copy overhead when plugging OpenVINO inference into an existing NPU pipeline.

Supported scenario by the Remote Tensor API:

- The NPU plugin context and memory objects can be constructed from low-level device, display, or memory handles and used to create the OpenVINO™ `ov::CompiledModel` or `ov::Tensor` objects.

Class and function declarations for the API are defined in the following file: `src/inference/include/openvino/runtime/intel_npu/level_zero/level_zero.hpp`

The most common way to enable the interaction of your application with the Remote Tensor API is to use user-side utility classes
and functions that consume or produce native handles directly.

## Context Sharing Between Application and NPU Plugin

NPU plugin classes that implement the `ov::RemoteContext` interface are responsible for context sharing.
Obtaining a context object is the first step in sharing pipeline objects.
The context object of the NPU plugin directly wraps Level Zero context, setting a scope for sharing the
`ov::RemoteTensor` objects. The `ov::RemoteContext` object is retrieved from the NPU plugin.

Once you have obtained the context, you can use it to create the `ov::RemoteTensor` objects.

### Getting RemoteContext from the Plugin

To request the current default context of the plugin, use one of the following methods:

<Tabs>
  <TabItem value="get-context-core" label="Get context from Core">
  ```cpp
          auto npu_context = core.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
          // Extract raw level zero context handle from RemoteContext
          void* context_handle = npu_context.get();
  ```
  </TabItem>
  <TabItem value="get-context-compiled-model" label="Get context from compiled model">
  ```cpp
          auto npu_context = compiled_model.get_context().as<ov::intel_npu::level_zero::ZeroContext>();
          // Extract raw level zero context handle from RemoteContext
          void* context_handle = npu_context.get();
  ```
  </TabItem>
</Tabs>

## Memory Sharing Between Application and NPU Plugin

The classes that implement the `ov::RemoteTensor` interface are the wrappers for native API
memory handles, which can be obtained from them at any time.

To create a shared tensor from a native memory handle or a file, use dedicated `create_tensor`, `create_l0_host_tensor`, or `create_host_tensor`
methods of the `ov::RemoteContext` sub-classes.
`ov::intel_npu::level_zero::LevelZero` has multiple overloads methods which enable wrapping pre-allocated native handles with the `ov::RemoteTensor`
object or requesting plugin to allocate specific device memory.
For more details, see the code snippets below:

<Tabs>
  <TabItem value="native-handle-and-file-mapping" label="Native Handle and File Mapping">
  <Tabs>
    <TabItem value="import-cpu-va" label="Import CPU virtual address allocation">
    ```cpp
            void* standard_allocation = nullptr;
            ov::intel_npu::MemType memory_type = ov::intel_npu::MemType::CPU_VA;
            auto remote_tensor = npu_context.create_tensor(in_element_type, in_shape, standard_allocation, memory_type);
    ```
    </TabItem>
    <TabItem value="nt-handle" label="NT handle">
    ```cpp
            void* shared_buffer = nullptr;
            ov::intel_npu::MemType memory_type = ov::intel_npu::MemType::SHARED_BUF;
            auto remote_tensor = npu_context.create_tensor(in_element_type, in_shape, shared_buffer, memory_type);
    ```
    </TabItem>
    <TabItem value="dma-buf" label="DMA-BUF file descriptor">
    ```cpp
            int dma_buf_fd = 0;
            ov::intel_npu::MemType memory_type = ov::intel_npu::MemType::SHARED_BUF;
            auto remote_tensor = npu_context.create_tensor(in_element_type, in_shape, dma_buf_fd, memory_type);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="allocate-device-memory" label="Allocate device memory">
  <Tabs>
    <TabItem value="remote-level-zero-host-memory" label="Remote Tensor - Level Zero host memory">
    ```cpp
            auto remote_tensor = npu_context.create_l0_host_tensor(in_element_type, in_shape);
            // Extract raw level zero pointer from remote tensor
            void* level_zero_ptr = remote_tensor.get();
    ```
    </TabItem>
    <TabItem value="level-zero-host-memory" label="Tensor - Level Zero host memory">
    ```cpp
            auto tensor = npu_context.create_host_tensor(in_element_type, in_shape);
            // Extract raw level zero pointer from remote tensor
            void* level_zero_ptr = tensor.data();
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

### Limitations

The NPU plugin does not support methods for direct allocation of native handles.

:::warning
**CPU Virtual Address Allocation Requirements**
When using CPU virtual address allocations, you **must** comply with the following requirements to prevent memory corruption and crashes:

**1. Memory Alignment (Mandatory)**
Both the allocation pointer and its size must be aligned to the standard page size (4KB). Non-aligned allocations will be rejected.

**2. Allocation Lifetime (Critical)**
The allocation must remain valid **until ALL** of the following have occurred:

- All inference requests using this remote tensor have completed execution, **AND**
- All inference requests using this remote tensor have been destroyed, **AND**
- The remote tensor has been destroyed

Failure to maintain the allocation for the entire lifecycle will result in undefined behavior and potential crashes.
:::

## Low-Level Methods for RemoteContext and RemoteTensor Creation

The high-level wrappers mentioned above bring a direct dependency on native APIs to your program.
If you want to avoid the dependency, you still can directly use the `ov::Core::create_context()`,
`ov::RemoteContext::create_tensor()`, and `ov::RemoteContext::get_params()` methods.
On this level, native handles are re-interpreted as void pointers and all arguments are passed
using `ov::AnyMap` containers that are filled with the `std::string, ov::Any` pairs.
Two types of map entries are possible: a descriptor and a container.
The descriptor sets the expected structure and possible parameter values of the map.

For possible low-level properties and their description, refer to the header file:
[remote\_properties.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/intel_npu/remote_properties.hpp).

## Additional Resources

- [ov::Core](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_core.html)
- [ov::RemoteTensor](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_remote_tensor.html)
