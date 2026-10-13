---
sidebar_position: 1
sidebar_label: "Remote Tensor API of GPU Plugin"
description: "The Remote Tensor API of GPU plugin in OpenVINO™ supports interoperability with existing native APIs, such as OpenCL, Microsoft DirectX, or VAAPI."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Remote Tensor API of GPU Plugin

The GPU plugin implementation of the `ov::RemoteContext` and `ov::RemoteTensor` interfaces supports GPU
pipeline developers who need video memory sharing and interoperability with existing native APIs,
such as OpenCL, Microsoft DirectX, or VAAPI.

The `ov::RemoteContext` and `ov::RemoteTensor` interface implementation targets the need for memory sharing and
interoperability with existing native APIs, such as OpenCL, Microsoft DirectX, and VAAPI.
They allow you to avoid any memory copy overhead when plugging OpenVINO™ inference
into an existing GPU pipeline. They also enable OpenCL kernels to participate in the pipeline to become
native buffer consumers or producers of the OpenVINO™ inference.

There are two interoperability scenarios supported by the Remote Tensor API:

- The GPU plugin context and memory objects can be constructed from low-level device, display, or memory handles and used to create the OpenVINO™ `ov::CompiledModel` or `ov::Tensor` objects.
- The OpenCL context or buffer handles can be obtained from existing GPU plugin objects, and used in OpenCL processing on the application side.

Class and function declarations for the API are defined in the following files:

- Windows -- `openvino/runtime/intel_gpu/ocl/ocl.hpp` and `openvino/runtime/intel_gpu/ocl/dx.hpp`
- Linux -- `openvino/runtime/intel_gpu/ocl/ocl.hpp` and `openvino/runtime/intel_gpu/ocl/va.hpp`

The most common way to enable the interaction of your application with the Remote Tensor API is to use user-side utility classes
and functions that consume or produce native handles directly.

## Context Sharing Between Application and GPU Plugin

GPU plugin classes that implement the `ov::RemoteContext` interface are responsible for context sharing.
Obtaining a context object is the first step in sharing pipeline objects.
The context object of the GPU plugin directly wraps OpenCL context, setting a scope for sharing the
`ov::CompiledModel` and `ov::RemoteTensor` objects. The `ov::RemoteContext` object can be either created on top of
an existing handle from a native API or retrieved from the GPU plugin.

Once you have obtained the context, you can use it to compile a new `ov::CompiledModel` or create `ov::RemoteTensor`
objects. For network compilation, use a dedicated flavor of `ov::Core::compile_model()`, which accepts the context as an additional parameter.

### Creation of RemoteContext from Native Handle

To create the `ov::RemoteContext` object for user context, explicitly provide the context to the plugin using constructor for one
of `ov::RemoteContext` derived classes.

<Tabs>
  <TabItem value="windows-cpp" label="Windows/C++">
  <Tabs>
    <TabItem value="create-from-cl-context" label="Create from cl_context">
    ```cpp
        cl_context ctx = get_cl_context();
        ov::intel_gpu::ocl::ClContext gpu_context(core, ctx);
    ```
    </TabItem>
    <TabItem value="create-from-cl-queue" label="Create from cl_queue">
    ```cpp
        cl_command_queue queue = get_cl_queue();
        ov::intel_gpu::ocl::ClContext gpu_context(core, queue);
    ```
    </TabItem>
    <TabItem value="create-from-id3d11device" label="Create from ID3D11Device">
    ```cpp
        ID3D11Device* device = get_d3d_device();
        ov::intel_gpu::ocl::D3DContext gpu_context(core, device);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="windows-c" label="Windows/C">
  <Tabs>
    <TabItem value="create-from-cl-context" label="Create from cl_context">
    ```c
        cl_context cl_context = get_cl_context();
        ov_core_create_context(core,
                               "GPU",
                               4,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "OCL",
                               ov_property_key_intel_gpu_ocl_context,
                               cl_context);
    ```
    </TabItem>
    <TabItem value="create-from-cl-queue" label="Create from cl_queue">
    ```c
        cl_command_queue cl_queue = get_cl_queue();
        cl_context cl_context = get_cl_context();
        ov_core_create_context(core,
                               "GPU",
                               6,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "OCL",
                               ov_property_key_intel_gpu_ocl_context,
                               cl_context,
                               ov_property_key_intel_gpu_ocl_queue,
                               cl_queue);
    ```
    </TabItem>
    <TabItem value="create-from-id3d11device" label="Create from ID3D11Device">
    ```c
        ID3D11Device* device = get_d3d_device();
        ov_core_create_context(core,
                               "GPU",
                               4,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "VA_SHARED",
                               ov_property_key_intel_gpu_va_device,
                               device);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="linux-cpp" label="Linux/C++">
  <Tabs>
    <TabItem value="create-from-cl-context" label="Create from cl_context">
    ```cpp
        cl_context ctx = get_cl_context();
        ov::intel_gpu::ocl::ClContext gpu_context(core, ctx);
    ```
    </TabItem>
    <TabItem value="create-from-cl-queue" label="Create from cl_queue">
    ```cpp
        cl_command_queue queue = get_cl_queue();
        ov::intel_gpu::ocl::ClContext gpu_context(core, queue);
    ```
    </TabItem>
    <TabItem value="create-from-vadisplay" label="Create from VADisplay">
    ```cpp
        VADisplay display = get_va_display();
        ov::intel_gpu::ocl::VAContext gpu_context(core, display);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="linux-c" label="Linux/C">
  <Tabs>
    <TabItem value="create-from-cl-context" label="Create from cl_context">
    ```c
        cl_context cl_context = get_cl_context();
        ov_core_create_context(core,
                               "GPU",
                               4,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "OCL",
                               ov_property_key_intel_gpu_ocl_context,
                               cl_context);
    ```
    </TabItem>
    <TabItem value="create-from-cl-queue" label="Create from cl_queue">
    ```c
        cl_command_queue cl_queue = get_cl_queue();
        cl_context cl_context = get_cl_context();
        ov_core_create_context(core,
                               "GPU",
                               6,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "OCL",
                               ov_property_key_intel_gpu_ocl_context,
                               cl_context,
                               ov_property_key_intel_gpu_ocl_queue,
                               cl_queue);
    ```
    </TabItem>
    <TabItem value="create-from-vadisplay" label="Create from VADisplay">
    ```c
        VADisplay display = get_va_display();
        ov_core_create_context(core,
                               "GPU",
                               4,
                               &gpu_context,
                               ov_property_key_intel_gpu_context_type,
                               "VA_SHARED",
                               ov_property_key_intel_gpu_va_device,
                               display);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

### Getting RemoteContext from the Plugin

If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all `ov::CompiledModel` objects created during this time share the same context.
Once the plugin options have been changed, the internal context is replaced by the new one.

To request the current default context of the plugin, use one of the following methods:

<Tabs>
  <TabItem value="cpp" label="C++">
  <Tabs>
    <TabItem value="get-context-core" label="Get context from Core">
    ```cpp
        auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
        // Extract ocl context handle from RemoteContext
        cl_context context_handle = gpu_context.get();
    ```
    </TabItem>
    <TabItem value="get-context-compiled-model" label="Get context from compiled model">
    ```cpp
        auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
        // Extract ocl context handle from RemoteContext
        cl_context context_handle = gpu_context.get();
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="c" label="C">
  <Tabs>
    <TabItem value="get-context-core" label="Get context from Core">
    ```c
        ov_core_get_default_context(core, "GPU", &gpu_context);
        // Extract ocl context handle from RemoteContext
        size_t size = 0;
        char* params = nullptr;
        // params is format like: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"
        // You need parse it.
        ov_remote_context_get_params(gpu_context, &size, &params);
    ```
    </TabItem>
    <TabItem value="get-context-compiled-model" label="Get context from compiled model">
    ```c
        ov_compiled_model_get_context(compiled_model, &gpu_context);
        // Extract ocl context handle from RemoteContext
        size_t size = 0;
        char* params = nullptr;
        // params is format like: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"
        // You need parse it.
        ov_remote_context_get_params(gpu_context, &size, &params);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

## Memory Sharing Between Application and GPU Plugin

The classes that implement the `ov::RemoteTensor` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use dedicated `create_tensor` or `create_tensor_nv12` methods
of the `ov::RemoteContext` sub-classes.
`ov::intel_gpu::ocl::ClContext` has multiple overloads of `create_tensor` methods which allow to wrap pre-allocated native handles with the `ov::RemoteTensor`
object or request plugin to allocate specific device memory. There also provides C APIs to do the same things with C++ APIs.
For more details, see the code snippets below:

:::note
In the C API, `ov_core_create_context()` and `ov_remote_context_create_tensor()` take property
values as variadic arguments and only handle-type properties are read as raw pointers:
`ov_property_key_intel_gpu_ocl_context`, `ov_property_key_intel_gpu_ocl_queue`,
`ov_property_key_intel_gpu_va_device`, `ov_property_key_intel_gpu_mem_handle`, and,
on Windows only, `ov_property_key_intel_gpu_dev_object_handle` (an `ID3D11Resource*`).
Every other value is read with `va_arg(..., char*)` and must be passed as a null-terminated
string, including numeric ones such as `ov_property_key_intel_gpu_va_plane`,
`ov_property_key_intel_gpu_dev_object_handle` on Linux (a `VASurfaceID`),
`ov_property_key_intel_gpu_ocl_context_device_id`, and `ov_property_key_intel_gpu_tile_id`.
Passing such a value as an integer is undefined behavior.
:::

<Tabs>
  <TabItem value="wrap-native-handles" label="Wrap native handles/C++">
  <Tabs>
    <TabItem value="usm-pointer" label="USM pointer">
    ```cpp
        void* shared_buffer = allocate_usm_buffer(input_size);
        auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    ```
    </TabItem>
    <TabItem value="cpu-pointer" label="CPU pointer">
    Use this overload when your application owns a CPU virtual address, for example memory
    allocated with `ov::util::aligned_alloc` or memory mapped from a file. On the OpenCL
    backend, the pointer address and allocation size must be aligned to
    `ov::intel_gpu::cacheline_size`. The memory must remain valid for the whole lifetime of
    the created remote tensor.

    ```cpp
        // Allocation part - must be done with alignment(for OCL backend) - align the address to cache line size
        // and the allocation size must be a multiple of cache line size.
        // In case of size of tensor lower than cache line size, allocate at least one cache line size 
        // and use ov::intel_gpu::VirtualAddressMemory(cpu_pointer, allocated_size)
        size_t size = input_size * in_element_type.size();
        const std::string target_device = "GPU";
        const uint32_t cacheline_size = core.get_property(target_device, ov::intel_gpu::cacheline_size);
        size = ov::util::align_size_up(size, cacheline_size); // ensure OCL runtime compatibility
        void* cpu_pointer = ov::util::aligned_alloc(size, cacheline_size);
        // end of Allocation part
        {
            // real wrapping cpu pointer to remote tensor
            auto remote_tensor = gpu_context.create_tensor(in_element_type,
                                                           in_shape,
                                                           ov::intel_gpu::VirtualAddressMemory(cpu_pointer, size));
        }
        ov::util::aligned_free(cpu_pointer);
    ```
    </TabItem>
    <TabItem value="cl-mem" label="cl_mem">
    ```cpp
        cl_mem shared_buffer = allocate_cl_mem(input_size);
        auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    ```
    </TabItem>
    <TabItem value="buffer" label="cl::Buffer">
    ```cpp
        cl::Buffer shared_buffer = allocate_buffer(input_size);
        auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    ```
    </TabItem>
    <TabItem value="image2d" label="cl::Image2D">
    ```cpp
        cl::Image2D shared_buffer = allocate_image(input_size);
        auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    ```
    </TabItem>
    <TabItem value="external-shared-handle" label="external shared handle">
    Use this overload when your application already owns an OS-level shared memory handle
    (for example, DX12 NT handle on Windows or DMA-BUF file descriptor on Linux).

    ```cpp
        auto shared_handle = get_shared_handle();
        auto remote_tensor = gpu_context.create_tensor(in_element_type,
                                                       in_shape,
                                                       shared_handle);
    ```

    The `shape` and `element type` must describe the same memory layout as the external buffer.
    The handle must remain valid for the whole lifetime of the created remote tensor.
    </TabItem>
    <TabItem value="file" label="file">
    Use this overload to wrap tensor data stored in a file. The access mode is declared in the
    descriptor: `AccessMode::READ` maps the file read-only and may only be used as an inference
    input, while `AccessMode::READ_WRITE` requires a writable file, may be used as an inference
    output, and makes changes done through the tensor visible in the file. The plugin keeps the
    mapping alive for the whole lifetime of the created remote tensor, so the file must not
    otherwise be modified until the tensor is destroyed.

    ```cpp
        // The plugin memory-maps the file and keeps the mapping alive for the tensor lifetime,
        // so the file must not be modified until the returned tensor is destroyed.
        ov::intel_gpu::FileDescriptor file_descriptor{"input.bin",
                                                      /*offset_in_bytes=*/0,
                                                      ov::intel_gpu::AccessMode::READ};
        auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, file_descriptor);
    ```
    </TabItem>
    <TabItem value="biplanar-nv12-surface" label="biplanar NV12 surface">
    ```cpp
        cl::Image2D y_plane_surface = allocate_image(y_plane_size);
        cl::Image2D uv_plane_surface = allocate_image(uv_plane_size);
        auto remote_tensor = gpu_context.create_tensor_nv12(y_plane_surface, uv_plane_surface);
        auto y_tensor = remote_tensor.first;
        auto uv_tensor = remote_tensor.second;
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="allocate-device-memory" label="Allocate device memory/C++">
  <Tabs>
    <TabItem value="usm-host-memory" label="USM host memory">
    ```cpp
        ov::intel_gpu::ocl::USMTensor remote_tensor = gpu_context.create_usm_host_tensor(in_element_type, in_shape);
        // Extract raw usm pointer from remote tensor
        void* usm_ptr = remote_tensor.get();
    ```
    </TabItem>
    <TabItem value="usm-device-memory" label="USM device memory">
    ```cpp
        auto remote_tensor = gpu_context.create_usm_device_tensor(in_element_type, in_shape);
        // Extract raw usm pointer from remote tensor
        void* usm_ptr = remote_tensor.get();
    ```
    </TabItem>
    <TabItem value="buffer" label="cl::Buffer">
    ```cpp
        ov::RemoteTensor remote_tensor = gpu_context.create_tensor(in_element_type, in_shape);
        // Cast from base to derived class and extract ocl memory handle
        auto buffer_tensor = remote_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
        cl_mem handle = buffer_tensor.get();
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

<Tabs>
  <TabItem value="wrap-native-handles" label="Wrap native handles/C">
  <Tabs>
    <TabItem value="usm-pointer" label="USM pointer">
    ```c
        void* shared_buffer = allocate_usm_buffer(input_size);
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        4,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "USM_USER_BUFFER",
                                        ov_property_key_intel_gpu_mem_handle,
                                        shared_buffer);
    ```
    </TabItem>
    <TabItem value="cl-mem" label="cl_mem">
    ```c
        cl_mem shared_buffer = allocate_cl_mem(input_size);
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        4,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_BUFFER",
                                        ov_property_key_intel_gpu_mem_handle,
                                        shared_buffer);
    ```
    </TabItem>
    <TabItem value="buffer" label="cl::Buffer">
    ```c
        cl::Buffer shared_buffer = allocate_buffer(input_size);
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        4,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_BUFFER",
                                        ov_property_key_intel_gpu_mem_handle,
                                        shared_buffer.get());
    ```
    </TabItem>
    <TabItem value="image2d" label="cl::Image2D">
    ```c
        cl::Image2D shared_buffer = allocate_image(input_size);
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        4,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_IMAGE2D",
                                        ov_property_key_intel_gpu_mem_handle,
                                        shared_buffer.get());
    ```
    </TabItem>
    <TabItem value="biplanar-nv12-surface" label="biplanar NV12 surface">
    ```c
        cl::Image2D y_plane_surface = allocate_image(y_plane_size);
        cl::Image2D uv_plane_surface = allocate_image(uv_plane_size);

        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        shape_y,
                                        4,
                                        &remote_tensor_y,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_IMAGE2D",
                                        ov_property_key_intel_gpu_mem_handle,
                                        y_plane_surface.get());

        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        shape_uv,
                                        4,
                                        &remote_tensor_uv,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "OCL_IMAGE2D",
                                        ov_property_key_intel_gpu_mem_handle,
                                        uv_plane_surface.get());

        ov_tensor_free(remote_tensor_y);
        ov_tensor_free(remote_tensor_uv);
        ov_shape_free(&shape_y);
        ov_shape_free(&shape_uv);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="allocate-device-memory" label="Allocate device memory/C">
  <Tabs>
    <TabItem value="usm-host-memory" label="USM host memory">
    ```c
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        2,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "USM_HOST_BUFFER");
        // Extract raw usm pointer from remote tensor
        void* usm_ptr = NULL;
        ov_tensor_data(remote_tensor, &usm_ptr);
    ```
    </TabItem>
    <TabItem value="usm-device-memory" label="USM device memory">
    ```c
        ov_remote_context_create_tensor(gpu_context,
                                        input_type,
                                        input_shape,
                                        2,
                                        &remote_tensor,
                                        ov_property_key_intel_gpu_shared_mem_type,
                                        "USM_USER_BUFFER");
        // Extract raw usm pointer from remote tensor
        void* usm_ptr = NULL;
        ov_tensor_data(remote_tensor, &usm_ptr);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

The `ov::intel_gpu::ocl::D3DContext` and `ov::intel_gpu::ocl::VAContext` classes are derived from `ov::intel_gpu::ocl::ClContext`.
Therefore, they provide the functionality described above and extend it
to enable creation of `ov::RemoteTensor` objects from `ID3D11Buffer`, `ID3D11Texture2D`
pointers or the `VASurfaceID` handle, as shown in the examples below:

<Tabs>
  <TabItem value="id3d11-buffer" label="ID3D11Buffer">
  ```cpp
  // ...

  // initialize the core and load the network
  ov::Core core;
  auto model = core.read_model("model.xml");
  auto compiled_model = core.compile_model(model, "GPU");
  auto infer_request = compiled_model.create_infer_request();


  // obtain the RemoteContext from the compiled model object and cast it to D3DContext
  auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::D3DContext>();

  auto input = model->get_parameters().at(0);
  ID3D11Buffer* d3d_handle = get_d3d_buffer();
  auto tensor = gpu_context.create_tensor(input->get_element_type(), input->get_shape(), d3d_handle);
  infer_request.set_tensor(input, tensor);
  ```
  </TabItem>
  <TabItem value="id3d11-texture" label="ID3D11Texture2D">
  ```cpp
  using namespace ov::preprocess;
  auto p = PrePostProcessor(model);
  p.input().tensor().set_element_type(ov::element::u8)
                    .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                    .set_memory_type(ov::intel_gpu::memory_type::surface);
  p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
  p.input().model().set_layout("NCHW");
  model = p.build();

  CComPtr<ID3D11Device> device_ptr = get_d3d_device_ptr()
  // create the shared context object
  auto shared_d3d_context = ov::intel_gpu::ocl::D3DContext(core, device_ptr);
  // compile model within a shared context
  auto compiled_model = core.compile_model(model, shared_d3d_context);

  auto param_input_y = model->get_parameters().at(0);
  auto param_input_uv = model->get_parameters().at(1);

  D3D11_TEXTURE2D_DESC texture_description = get_texture_desc();
  CComPtr<ID3D11Texture2D> dx11_texture = get_texture();
  //     ...
  //wrap decoder output into RemoteBlobs and set it as inference input
  auto nv12_blob = shared_d3d_context.create_tensor_nv12(texture_description.Heights, texture_description.Width, dx11_texture);

  auto infer_request = compiled_model.create_infer_request();
  infer_request.set_tensor(param_input_y->get_friendly_name(), nv12_blob.first);
  infer_request.set_tensor(param_input_uv->get_friendly_name(), nv12_blob.second);
  infer_request.start_async();
  infer_request.wait();
  ```
  </TabItem>
  <TabItem value="vasurfaceid" label="VASurfaceID">
  ```cpp
  using namespace ov::preprocess;
  auto p = PrePostProcessor(model);
  p.input().tensor().set_element_type(ov::element::u8)
                    .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                    .set_memory_type(ov::intel_gpu::memory_type::surface);
  p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
  p.input().model().set_layout("NCHW");
  model = p.build();

  CComPtr<ID3D11Device> device_ptr = get_d3d_device_ptr()
  // create the shared context object
  auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, device_ptr);
  // compile model within a shared context
  auto compiled_model = core.compile_model(model, shared_va_context);

  auto param_input_y = model->get_parameters().at(0);
  auto param_input_uv = model->get_parameters().at(1);

  auto shape = param_input_y->get_shape();
  auto width = shape[1];
  auto height = shape[2];

  VASurfaceID va_surface = decode_va_surface();
  //     ...
  //wrap decoder output into RemoteBlobs and set it as inference input
  auto nv12_blob = shared_va_context.create_tensor_nv12(height, width, va_surface);

  auto infer_request = compiled_model.create_infer_request();
  infer_request.set_tensor(param_input_y->get_friendly_name(), nv12_blob.first);
  infer_request.set_tensor(param_input_uv->get_friendly_name(), nv12_blob.second);
  infer_request.start_async();
  infer_request.wait();
  ```
  </TabItem>
</Tabs>

:::info[Important]
Currently, only sharing of D3D11 surfaces is supported via the
[cl\_intel\_d3d11\_nv12\_media\_sharing](https://github.com/KhronosGroup/OpenCL-Registry/blob/main/extensions/intel/cl_intel_d3d11_nv12_media_sharing.txt)
extension, which provides interoperability between OpenCL and DirectX.
:::

## Direct NV12 Video Surface Input

To support the direct consumption of a hardware video decoder output, the GPU plugin accepts:

- Two-plane NV12 video surface input - calling the `create_tensor_nv12()` function creates
  a pair of `ov::RemoteTensor` objects, representing the Y and UV planes.
- Single-plane NV12 video surface input - calling the `create_tensor()` function creates one
  `ov::RemoteTensor` object, representing the Y and UV planes at once (Y elements before UV elements).
- NV12 to Grey video surface input conversion - calling the `create_tensor()` function creates one
  `ov::RemoteTensor` object, representing only the Y plane.

To ensure that the plugin generates a correct execution graph, static preprocessing
should be added before model compilation:

<Tabs>
  <TabItem value="two-plane" label="two-plane">
  <Tabs>
    <TabItem value="cpp" label="C++">
    ```cpp
        using namespace ov::preprocess;
        auto p = PrePostProcessor(model);
        p.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(ov::intel_gpu::memory_type::surface);
        p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        p.input().model().set_layout("NCHW");
        auto model_with_preproc = p.build();
    ```
    </TabItem>
    <TabItem value="c" label="C">
    ```c
        ov_preprocess_prepostprocessor_create(model, &preprocess);
        ov_preprocess_prepostprocessor_get_input_info(preprocess, &preprocess_input_info);
        ov_preprocess_input_info_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info);
        ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, ov_element_type_e::U8);
        ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info,
                                                                      ov_color_format_e::NV12_TWO_PLANES,
                                                                      2,
                                                                      "y",
                                                                      "uv");
        ov_preprocess_input_tensor_info_set_memory_type(preprocess_input_tensor_info, "GPU_SURFACE");
        ov_preprocess_input_tensor_info_set_spatial_static_shape(preprocess_input_tensor_info, height, width);
        ov_preprocess_input_info_get_preprocess_steps(preprocess_input_info, &preprocess_input_steps);
        ov_preprocess_preprocess_steps_convert_color(preprocess_input_steps, ov_color_format_e::BGR);
        ov_preprocess_preprocess_steps_resize(preprocess_input_steps, RESIZE_LINEAR);
        ov_preprocess_input_info_get_model_info(preprocess_input_info, &preprocess_input_model_info);
        ov_layout_create("NCHW", &layout);
        ov_preprocess_input_model_info_set_layout(preprocess_input_model_info, layout);
        ov_preprocess_prepostprocessor_build(preprocess, &model_with_preproc);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="single-plane" label="single-plane">
  ```cpp
      using namespace ov::preprocess;
      auto p = PrePostProcessor(model);
      p.input().tensor().set_element_type(ov::element::u8)
                        .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                        .set_memory_type(ov::intel_gpu::memory_type::surface);
      p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
      p.input().model().set_layout("NCHW");
      auto model_with_preproc = p.build();
  ```
  </TabItem>
  <TabItem value="nv12-grey" label="NV12 to Grey">
  ```cpp
      using namespace ov::preprocess;
      auto p = PrePostProcessor(model);
      p.input().tensor().set_element_type(ov::element::u8)
                        .set_layout("NHWC")
                        .set_memory_type(ov::intel_gpu::memory_type::surface);
      p.input().model().set_layout("NCHW");
      auto model_with_preproc = p.build();
  ```
  </TabItem>
</Tabs>

Since the `ov::intel_gpu::ocl::ClImage2DTensor` and its derived classes do not support batched surfaces,
if batching and surface sharing are required at the same time,
inputs need to be set via the `ov::InferRequest::set_tensors` method with vector of shared surfaces for each plane:

<Tabs>
  <TabItem value="single-batch" label="Single Batch">
  <Tabs>
    <TabItem value="two-plane" label="two-plane">
    <Tabs>
      <TabItem value="cpp" label="C++">
      ```cpp
          auto input0 = model_with_preproc->get_parameters().at(0);
          auto input1 = model_with_preproc->get_parameters().at(1);
          ov::intel_gpu::ocl::ClImage2DTensor y_tensor = get_y_tensor();
          ov::intel_gpu::ocl::ClImage2DTensor uv_tensor = get_uv_tensor();
          infer_request.set_tensor(input0->get_friendly_name(), y_tensor);
          infer_request.set_tensor(input1->get_friendly_name(), uv_tensor);
          infer_request.infer();
      ```
      </TabItem>
      <TabItem value="cpp-2" label="C">
      ```c
              ov_model_const_input_by_index(model, 0, &input_port0);
              ov_model_const_input_by_index(model, 1, &input_port1);
              ov_port_get_any_name(input_port0, &input_name0);
              ov_port_get_any_name(input_port1, &input_name1);

              ov_shape_t shape_y, shape_uv;
              ov_tensor_t* remote_tensor_y = NULL;
              ov_tensor_t* remote_tensor_uv = NULL;
              ov_const_port_get_shape(input_port0, &shape_y);
              ov_const_port_get_shape(input_port1, &shape_uv);

              cl::Image2D image_y = get_y_image();
              cl::Image2D image_uv = get_uv_image();
              ov_remote_context_create_tensor(gpu_context,
                                              ov_element_type_e::U8,
                                              shape_y,
                                              4,
                                              &remote_tensor_y,
                                              ov_property_key_intel_gpu_shared_mem_type,
                                              "OCL_IMAGE2D",
                                              ov_property_key_intel_gpu_mem_handle,
                                              image_y.get());

              ov_remote_context_create_tensor(gpu_context,
                                              ov_element_type_e::U8,
                                              shape_uv,
                                              4,
                                              &remote_tensor_y,
                                              ov_property_key_intel_gpu_shared_mem_type,
                                              "OCL_IMAGE2D",
                                              ov_property_key_intel_gpu_mem_handle,
                                              image_uv.get());

              ov_infer_request_set_tensor(infer_request, input_name0, remote_tensor_y);
              ov_infer_request_set_tensor(infer_request, input_name1, remote_tensor_uv);
              ov_infer_request_infer(infer_request);
      ```
      </TabItem>
    </Tabs>
    </TabItem>
    <TabItem value="single-plane" label="single-plane">
    ```cpp
        auto input_yuv = model_with_preproc->input(0);
        ov::intel_gpu::ocl::ClImage2DTensor yuv_tensor = get_yuv_tensor();
        infer_request.set_tensor(input_yuv.get_any_name(), yuv_tensor);
        infer_request.infer();
    ```
    </TabItem>
    <TabItem value="nv12-grey" label="NV12 to Grey">
    ```cpp
        cl::Image2D img_y_plane;
        auto input_y = model_with_preproc->input(0);
        auto remote_y_tensor = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane);
        infer_request.set_tensor(input_y.get_any_name(), remote_y_tensor);
        infer_request.infer();
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="multiple-batches" label="Multiple Batches">
  <Tabs>
    <TabItem value="two-plane" label="two-plane">
    ```cpp
        auto input0 = model_with_preproc->get_parameters().at(0);
        auto input1 = model_with_preproc->get_parameters().at(1);
        std::vector<ov::Tensor> y_tensors = {y_tensor_0, y_tensor_1};
        std::vector<ov::Tensor> uv_tensors = {uv_tensor_0, uv_tensor_1};
        infer_request.set_tensors(input0->get_friendly_name(), y_tensors);
        infer_request.set_tensors(input1->get_friendly_name(), uv_tensors);
        infer_request.infer();
    ```
    </TabItem>
    <TabItem value="single-plane" label="single-plane">
    ```cpp
        auto input_yuv = model_with_preproc->input(0);
        std::vector<ov::Tensor> yuv_tensors = {yuv_tensor_0, yuv_tensor_1};
        infer_request.set_tensors(input_yuv.get_any_name(), yuv_tensors);
        infer_request.infer();
    ```
    </TabItem>
    <TabItem value="nv12-grey" label="NV12 to Grey">
    ```cpp
        cl::Image2D img_y_plane_0, img_y_plane_l;
        auto input_y = model_with_preproc->input(0);
        auto remote_y_tensor_0 = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane_0);
        auto remote_y_tensor_1 = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane_l);
        std::vector<ov::Tensor> y_tensors = {remote_y_tensor_0, remote_y_tensor_1};
        infer_request.set_tensors(input_y.get_any_name(), y_tensors);
        infer_request.infer();
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

I420 color format can be processed in a similar way

## Context & Queue Sharing

The GPU plugin supports creation of shared context from the `cl_command_queue` handle. In that case,
the `opencl` context handle is extracted from the given queue via OpenCL™ API, and the queue itself is used inside
the plugin for further execution of inference primitives. Sharing the queue changes the behavior of the `ov::InferRequest::start_async()`
method to guarantee that submission of inference primitives into the given queue is finished before
returning control back to the calling thread.

This sharing mechanism allows performing pipeline synchronization on the app side and avoiding blocking the host thread
on waiting for the completion of inference. The pseudo-code may look as follows:

<details>
<summary>Queue and context sharing example</summary>

```cpp

    // ...

    // initialize the core and read the model
    ov::Core core;
    auto model = core.read_model("model.xml");

    // get opencl queue object
    cl::CommandQueue queue = get_ocl_queue();
    cl::Context cl_context = get_ocl_context();

    // share the queue with GPU plugin and compile model
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue.get());
    auto exec_net_shared = core.compile_model(model, remote_context);

    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto output_size = ov::shape_size(output->get_shape());
    cl_int err;

    // create the OpenCL buffers within the context
    cl::Buffer shared_in_buffer(cl_context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    cl::Buffer shared_out_buffer(cl_context, CL_MEM_READ_WRITE, output_size, NULL, &err);
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto shared_in_blob = remote_context.create_tensor(input->get_element_type(), input->get_shape(), shared_in_buffer);
    auto shared_out_blob = remote_context.create_tensor(output->get_element_type(), output->get_shape(), shared_out_buffer);
    auto infer_request = exec_net_shared.create_infer_request();
    infer_request.set_tensor(input, shared_in_blob);
    infer_request.set_tensor(output, shared_out_blob);

    // ...
    // execute user kernel
    cl::Program program;
    cl::Kernel kernel_preproc(program, "user_kernel_preproc");
    kernel_preproc.setArg(0, shared_in_buffer);
    queue.enqueueNDRangeKernel(kernel_preproc,
                               cl::NDRange(0),
                               cl::NDRange(input_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);
    // Blocking clFinish() call is not required, but this barrier is added to the queue to guarantee that user kernel is finished
    // before any inference primitive is started
    queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    // ...

    // pass results to the inference
    // since the remote context is created with queue sharing, start_async() guarantees that scheduling is finished
    infer_request.start_async();

    // execute some postprocessing kernel.
    // infer_request.wait() is not called, synchronization between inference and post-processing is done via
    // enqueueBarrierWithWaitList call.
    cl::Kernel kernel_postproc(program, "user_kernel_postproc");
    kernel_postproc.setArg(0, shared_out_buffer);
    queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    queue.enqueueNDRangeKernel(kernel_postproc,
                               cl::NDRange(0),
                               cl::NDRange(output_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);

    // Wait for pipeline completion
    queue.finish();
```

</details>

### Limitations

- Some primitives in the GPU plugin may block the host thread on waiting for the previous primitives before adding its kernels to the command queue. In such cases, the `ov::InferRequest::start_async()` call takes much more time to return control to the calling thread as internally it waits for a partial or full network completion. Examples of operations: Loop, TensorIterator, DetectionOutput, NonMaxSuppression
- Synchronization of pre/post processing jobs and inference pipeline inside a shared queue is user's responsibility.
- Throughput mode is not available when queue sharing is used, i.e., only a single stream can be used for each compiled model.

## Low-Level Methods for RemoteContext and RemoteTensor Creation

The high-level wrappers mentioned above bring a direct dependency on native APIs to the user program.
If you want to avoid the dependency, you still can directly use the `ov::Core::create_context()`,
`ov::RemoteContext::create_tensor()`, and `ov::RemoteContext::get_params()` methods.
On this level, native handles are re-interpreted as void pointers and all arguments are passed
using `ov::AnyMap` containers that are filled with `std::string, ov::Any` pairs.
Two types of map entries are possible: descriptor and container.
Descriptor sets the expected structure and possible parameter values of the map.

For possible low-level properties and their description, refer to the header file:
[remote\_properties.hpp](https://github.com/openvinotoolkit/openvino/blob/releases/2026/0/src/inference/include/openvino/runtime/intel_gpu/remote_properties.hpp).

## Examples

To see pseudo-code of usage examples, refer to the sections below.

:::note
For low-level parameter usage examples, see the source code of user-side wrappers from the include files mentioned above.
:::

<details>
<summary>OpenCL Kernel Execution on a Shared Buffer</summary>

This example uses the OpenCL context obtained from a compiled model object.

```cpp

    // ...

    // initialize the core and load the network
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU");
    auto infer_request = compiled_model.create_infer_request();


    // obtain the RemoteContext from the compiled model object and cast it to ClContext
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    // obtain the OpenCL context handle from the RemoteContext,
    // get device info and create a queue
    cl::Context cl_context = gpu_context;
    cl::Device device = cl::Device(cl_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl::CommandQueue queue = cl::CommandQueue(cl_context, device, props);

    // create the OpenCL buffer within the obtained context
    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());
    cl_int err;
    cl::Buffer shared_buffer(cl_context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    // wrap the buffer into RemoteBlob
    auto shared_blob = gpu_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);

    // ...
    // execute user kernel
    cl::Program program;
    cl::Kernel kernel(program, "user_kernel");
    kernel.setArg(0, shared_buffer);
    queue.enqueueNDRangeKernel(kernel,
                               cl::NDRange(0),
                               cl::NDRange(input_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);
    queue.finish();
    // ...
    // pass results to the inference
    infer_request.set_tensor(input, shared_blob);
    infer_request.infer();
```

</details>

<details>
<summary>Running GPU Plugin Inference within User-Supplied Shared Context</summary>

```cpp
    cl::Context ctx = get_ocl_context();

    ov::Core core;
    auto model = core.read_model("model.xml");

    // share the context with GPU plugin and compile ExecutableNetwork
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, ctx.get());
    auto exec_net_shared = core.compile_model(model, remote_context);
    auto inf_req_shared = exec_net_shared.create_infer_request();


    // ...
    // do OpenCL processing stuff
    // ...

    // run the inference
    inf_req_shared.infer();
```

</details>

<details>
<summary>Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux</summary>

<Tabs>
  <TabItem value="cpp" label="C++">
  ```cpp

      // ...

      using namespace ov::preprocess;
      auto p = PrePostProcessor(model);
      p.input().tensor().set_element_type(ov::element::u8)
                        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                        .set_memory_type(ov::intel_gpu::memory_type::surface);
      p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
      p.input().model().set_layout("NCHW");
      model = p.build();

      VADisplay disp = get_va_display();
      // create the shared context object
      auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, disp);
      // compile model within a shared context
      auto compiled_model = core.compile_model(model, shared_va_context);

      auto input0 = model->get_parameters().at(0);
      auto input1 = model->get_parameters().at(1);

      auto shape = input0->get_shape();
      auto width = shape[1];
      auto height = shape[2];

      // execute decoding and obtain decoded surface handle
      VASurfaceID va_surface = decode_va_surface();
      //     ...
      //wrap decoder output into RemoteBlobs and set it as inference input
      auto nv12_blob = shared_va_context.create_tensor_nv12(height, width, va_surface);

      auto infer_request = compiled_model.create_infer_request();
      infer_request.set_tensor(input0->get_friendly_name(), nv12_blob.first);
      infer_request.set_tensor(input1->get_friendly_name(), nv12_blob.second);
      infer_request.start_async();
      infer_request.wait();
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```c

      // ...

      ov_preprocess_prepostprocessor_create(model, &preprocess);
      ov_preprocess_prepostprocessor_get_input_info(preprocess, &preprocess_input_info);
      ov_preprocess_input_info_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info);
      ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, U8);
      ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info,
                                                                    NV12_TWO_PLANES,
                                                                    2,
                                                                    "y",
                                                                    "uv");
      ov_preprocess_input_tensor_info_set_memory_type(preprocess_input_tensor_info, "GPU_SURFACE");
      ov_preprocess_input_tensor_info_set_spatial_static_shape(preprocess_input_tensor_info, height, width);
      ov_preprocess_input_info_get_preprocess_steps(preprocess_input_info, &preprocess_input_steps);
      ov_preprocess_preprocess_steps_convert_color(preprocess_input_steps, BGR);
      ov_preprocess_preprocess_steps_resize(preprocess_input_steps, RESIZE_LINEAR);
      ov_preprocess_input_info_get_model_info(preprocess_input_info, &preprocess_input_model_info);
      ov_layout_create("NCHW", &layout);
      ov_preprocess_input_model_info_set_layout(preprocess_input_model_info, layout);
      ov_preprocess_prepostprocessor_build(preprocess, &new_model);

      VADisplay display = get_va_display();
      // create the shared context object
      ov_core_create_context(core,
                             "GPU",
                             4,
                             &shared_va_context,
                             ov_property_key_intel_gpu_context_type,
                             "VA_SHARED",
                             ov_property_key_intel_gpu_va_device,
                             display);

      // compile model within a shared context
      ov_core_compile_model_with_context(core, new_model, shared_va_context, 0, &compiled_model);

      ov_output_const_port_t* port_0 = NULL;
      char* input_name_0 = NULL;
      ov_model_const_input_by_index(new_model, 0, &port_0);
      ov_port_get_any_name(port_0, &input_name_0);

      ov_output_const_port_t* port_1 = NULL;
      char* input_name_1 = NULL;
      ov_model_const_input_by_index(new_model, 1, &port_1);
      ov_port_get_any_name(port_1, &input_name_1);

      ov_shape_t shape_y = {0, NULL};
      ov_shape_t shape_uv = {0, NULL};
      ov_const_port_get_shape(port_0, &shape_y);
      ov_const_port_get_shape(port_1, &shape_uv);

      // execute decoding and obtain decoded surface handle
      VASurfaceID va_surface = decode_va_surface();
      //     ...
      //wrap decoder output into RemoteBlobs and set it as inference input

      // the C API reads every non-handle property value with va_arg(..., char*),
      // so numeric values such as the VASurfaceID and the plane index must be decimal strings
      char surface_id[21];
      snprintf(surface_id, sizeof(surface_id), "%u", (unsigned int)va_surface);

      ov_tensor_t* remote_tensor_y = NULL;
      ov_tensor_t* remote_tensor_uv = NULL;
      ov_remote_context_create_tensor(shared_va_context,
                                      U8,
                                      shape_y,
                                      6,
                                      &remote_tensor_y,
                                      ov_property_key_intel_gpu_shared_mem_type,
                                      "VA_SURFACE",
                                      ov_property_key_intel_gpu_dev_object_handle,
                                      surface_id,
                                      ov_property_key_intel_gpu_va_plane,
                                      "0");
      ov_remote_context_create_tensor(shared_va_context,
                                      U8,
                                      shape_uv,
                                      6,
                                      &remote_tensor_uv,
                                      ov_property_key_intel_gpu_shared_mem_type,
                                      "VA_SURFACE",
                                      ov_property_key_intel_gpu_dev_object_handle,
                                      surface_id,
                                      ov_property_key_intel_gpu_va_plane,
                                      "1");

      ov_compiled_model_create_infer_request(compiled_model, &infer_request);
      ov_infer_request_set_tensor(infer_request, input_name_0, remote_tensor_y);
      ov_infer_request_set_tensor(infer_request, input_name_1, remote_tensor_uv);
      ov_infer_request_infer(infer_request);
  ```
  </TabItem>
</Tabs>

</details>

## See Also

- [ov::Core](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_core.html)
- [ov::RemoteTensor](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_remote_tensor.html)
