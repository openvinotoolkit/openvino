# Remote Tensor API of GPU Plugin {#openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API}

The GPU plugin implementation of the `ov::RemoteContext` and `ov::RemoteTensor` interfaces supports GPU
pipeline developers who need video memory sharing and interoperability with existing native APIs, 
such as OpenCL, Microsoft DirectX, or VAAPI.

The `ov::RemoteContext` and `ov::RemoteTensor` interface implementation targets the need for memory sharing and
interoperability with existing native APIs, such as OpenCL, Microsoft DirectX, and VAAPI.
They allow you to avoid any memory copy overhead when plugging OpenVINO™ inference
into an existing GPU pipeline. They also enable OpenCL kernels to participate in the pipeline to become
native buffer consumers or producers of the OpenVINO™ inference.

There are two interoperability scenarios supported by the Remote Tensor API:

* The GPU plugin context and memory objects can be constructed from low-level device, display, or memory
handles and used to create the OpenVINO™ `ov::CompiledModel` or `ov::Tensor` objects.
* The OpenCL context or buffer handles can be obtained from existing GPU plugin objects, and used in OpenCL processing on the application side.

Class and function declarations for the API are defined in the following files:
* Windows -- `openvino/runtime/intel_gpu/ocl/ocl.hpp` and `openvino/runtime/intel_gpu/ocl/dx.hpp`
* Linux -- `openvino/runtime/intel_gpu/ocl/ocl.hpp` and `openvino/runtime/intel_gpu/ocl/va.hpp`

The most common way to enable the interaction of your application with the Remote Tensor API is to use user-side utility classes
and functions that consume or produce native handles directly.

## Context Sharing Between Application and GPU Plugin

GPU plugin classes that implement the `ov::RemoteContext` interface are responsible for context sharing.
Obtaining a context object is the first step in sharing pipeline objects.
The context object of the GPU plugin directly wraps OpenCL context, setting a scope for sharing the
`ov::CompiledModel` and `ov::RemoteTensor` objects. The `ov::RemoteContext` object can be either created on top of
an existing handle from a native API or retrieved from the GPU plugin.

Once you have obtained the context, you can use it to compile a new `ov::CompiledModel` or create `ov::RemoteTensor`
objects.
For network compilation, use a dedicated flavor of `ov::Core::compile_model()`, which accepts the context as an
additional parameter.

### Creation of RemoteContext from Native Handle
To create the `ov::RemoteContext` object for user context, explicitly provide the context to the plugin using constructor for one
of `ov::RemoteContext` derived classes.

@sphinxdirective

.. tab:: Linux

   .. tab:: Create from cl_context
 
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_cl_context

   .. tab:: Create from cl_queue

      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_cl_queue

   .. tab:: Create from VADisplay

      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_va_display

.. tab:: Windows

   .. tab:: Create from cl_context

      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_cl_context

   .. tab:: Create from cl_queue

      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_cl_queue

   .. tab:: Create from ID3D11Device
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: context_from_d3d_device

@endsphinxdirective

### Getting RemoteContext from the Plugin
If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all `ov::CompiledModel` objects created during this time share the same context.
Once the plugin options have been changed, the internal context is replaced by the new one.

To request the current default context of the plugin, use one of the following methods:

@sphinxdirective

.. tab:: Get context from Core

   .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
      :language: cpp
      :fragment: default_context_from_core

.. tab:: Get context from compiled model

   .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
      :language: cpp
      :fragment: default_context_from_model

@endsphinxdirective

## Memory Sharing Between Application and GPU Plugin

The classes that implement the `ov::RemoteTensor` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use dedicated `create_tensor`or `create_tensor_nv12` methods
of the `ov::RemoteContext` sub-classes.
`ov::intel_gpu::ocl::ClContext` has multiple overloads of `create_tensor` methods which allow to wrap pre-allocated native handles with the `ov::RemoteTensor`
object or request plugin to allocate specific device memory. For more details, see the code snippets below:

@sphinxdirective

.. tab:: Wrap native handles

   .. tab:: USM pointer
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: wrap_usm_pointer

   .. tab:: cl_mem
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: wrap_cl_mem

   .. tab:: cl::Buffer
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: wrap_cl_buffer         

   .. tab:: cl::Image2D
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: wrap_cl_image   

   .. tab:: biplanar NV12 surface
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: wrap_nv12_surface   

.. tab:: Allocate device memory

   .. tab:: USM host memory
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: allocate_usm_host

   .. tab:: USM device memory
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: allocate_usm_device

   .. tab:: cl::Buffer
   
      .. doxygensnippet:: docs/snippets/gpu/remote_objects_creation.cpp
         :language: cpp
         :fragment: allocate_cl_buffer

@endsphinxdirective

The `ov::intel_gpu::ocl::D3DContext` and `ov::intel_gpu::ocl::VAContext` classes are derived from `ov::intel_gpu::ocl::ClContext`.
Therefore, they provide the functionality described above and extend it
to allow creation of `ov::RemoteTensor` objects from `ID3D11Buffer`, `ID3D11Texture2D` pointers or the `VASurfaceID` handle respectively.


## Direct NV12 Video Surface Input

To support the direct consumption of a hardware video decoder output, the GPU plugin accepts:

* Two-plane NV12 video surface input - calling the `create_tensor_nv12()` function creates 
  a pair of `ov::RemoteTensor` objects, representing the Y and UV planes. 
* Single-plane NV12 video surface input - calling the `create_tensor()` function creates one 
  `ov::RemoteTensor` object, representing the Y and UV planes at once (Y elements before UV elements).
* NV12 to Grey video surface input conversion - calling the `create_tensor()` function creates one 
  `ov::RemoteTensor` object, representing only the Y plane.

To ensure that the plugin generates a correct execution graph, static preprocessing
should be added before model compilation:

@sphinxdirective

.. tab:: two-plane

    .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_two_planes.cpp
       :language: cpp
       :fragment: [init_preproc]

.. tab:: single-plane

    .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_single_plane.cpp
       :language: cpp
       :fragment: [init_preproc]

.. tab:: NV12 to Grey

    .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_to_gray.cpp
       :language: cpp
       :fragment: [init_preproc]

@endsphinxdirective


Since the `ov::intel_gpu::ocl::ClImage2DTensor` and its derived classes do not support batched surfaces, 
if batching and surface sharing are required at the same time, 
inputs need to be set via the `ov::InferRequest::set_tensors` method with vector of shared surfaces for each plane:


@sphinxdirective

.. tab:: Single Batch

   .. tab:: two-plane

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_two_planes.cpp
         :language: cpp
         :fragment: single_batch

   .. tab:: single-plane
   
      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_single_plane.cpp
         :language: cpp
         :fragment: single_batch

   .. tab:: NV12 to Grey

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_to_gray.cpp
         :language: cpp
         :fragment: single_batch

.. tab:: Multiple Batches

   .. tab:: two-plane

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_two_planes.cpp
         :language: cpp
         :fragment: batched_case

   .. tab:: single-plane
                                            
      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_single_plane.cpp
         :language: cpp
         :fragment: batched_case

   .. tab:: NV12 to Grey

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_to_gray.cpp
         :language: cpp
         :fragment: batched_case

@endsphinxdirective

I420 color format can be processed in a similar way

## Context & Queue Sharing

The GPU plugin supports creation of shared context from the `cl_command_queue` handle. In that case,
the `opencl` context handle is extracted from the given queue via OpenCL™ API, and the queue itself is used inside
the plugin for further execution of inference primitives. Sharing the queue changes the behavior of the `ov::InferRequest::start_async()`
method to guarantee that submission of inference primitives into the given queue is finished before
returning control back to the calling thread.

This sharing mechanism allows performing pipeline synchronization on the app side and avoiding blocking the host thread
on waiting for the completion of inference. The pseudo-code may look as follows:

@sphinxdirective

.. dropdown:: Queue and context sharing example

   .. doxygensnippet:: docs/snippets/gpu/queue_sharing.cpp
      :language: cpp
      :fragment: queue_sharing

@endsphinxdirective

### Limitations

 - Some primitives in the GPU plugin may block the host thread on waiting for the previous primitives before adding its kernels
   to the command queue. In such cases, the `ov::InferRequest::start_async()` call takes much more time to return control to the calling thread
   as internally it waits for a partial or full network completion.
   Examples of operations: Loop, TensorIterator, DetectionOutput, NonMaxSuppression
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

For possible low-level properties and their description, refer to the `openvino/runtime/intel_gpu/remote_properties.hpp` header file .

## Examples

To see pseudo-code of usage examples, refer to the sections below.

@sphinxdirective

.. NOTE::
   
   For low-level parameter usage examples, see the source code of user-side wrappers from the include files mentioned above.

.. dropdown:: OpenCL Kernel Execution on a Shared Buffer

   This example uses the OpenCL context obtained from a compiled model object.

   .. doxygensnippet:: docs/snippets/gpu/context_sharing.cpp
      :language: cpp
      :fragment: context_sharing_get_from_ov

.. dropdown:: Running GPU Plugin Inference within User-Supplied Shared Context

   .. doxygensnippet:: docs/snippets/gpu/context_sharing.cpp
      :language: cpp
      :fragment: context_sharing_user_handle

.. dropdown:: Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux

   .. doxygensnippet:: docs/snippets/gpu/context_sharing_va.cpp
      :language: cpp
      :fragment: context_sharing_va

@endsphinxdirective


## See Also

* ov::Core
* ov::RemoteTensor
