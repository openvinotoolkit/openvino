# Remote Tensor API of GPU Plugin {#openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API}

The GPU plugin implementation of the `ov::RemoteContext` and `ov::RemoteTensor` interfaces supports GPU
pipeline developers who need video memory sharing and interoperability with existing native APIs, 
such as OpenCL, Microsoft DirectX, or VAAPI.
Using these interfaces allows you to avoid any memory copy overhead when plugging OpenVINO™ inference
into an existing GPU pipeline. It also enables OpenCL kernels to participate in the pipeline to become
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
Obtaining a context object is the first step of sharing pipeline objects.
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

@sphinxtabset

@sphinxtab{Linux}

@sphinxtabset

@sphinxtab{Create from cl_context}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_cl_context

@endsphinxtab

@sphinxtab{Create from cl_queue}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_cl_queue

@endsphinxtab

@sphinxtab{Create from VADisplay}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_va_display

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Windows}

@sphinxtabset

@sphinxtab{Create from cl_context}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_cl_context

@endsphinxtab

@sphinxtab{Create from cl_queue}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_cl_queue

@endsphinxtab

@sphinxtab{Create from ID3D11Device}

@snippet docs/snippets/gpu/remote_objects_creation.cpp context_from_d3d_device

@endsphinxtab

@endsphinxtabset

@endsphinxtabset


### Getting RemoteContext from the Plugin
If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all `ov::CompiledModel` objects created during this time share the same context.
Once the plugin options have been changed, the internal context is replaced by the new one.

To request the current default context of the plugin, use one of the following methods:

@sphinxtabset

@sphinxtab{Get context from Core}

@snippet docs/snippets/gpu/remote_objects_creation.cpp default_context_from_core

@endsphinxtab

@sphinxtab{Batching via throughput hint}

@snippet docs/snippets/gpu/remote_objects_creation.cpp default_context_from_model

@endsphinxtab

@endsphinxtabset

## Memory Sharing Between Application and GPU Plugin

The classes that implement the `ov::RemoteTensor` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use dedicated `create_tensor`or `create_tensor_nv12` methods
of the `ov::RemoteContext` sub-classes.
`ov::intel_gpu::ocl::ClContext` has multiple overloads of `create_tensor` methods which allow to wrap pre-allocated native handles with the `ov::RemoteTensor`
object or request plugin to allocate specific device memory. For more details, see the code snippets below:

@sphinxtabset

@sphinxtab{Wrap native handles}

@sphinxtabset

@sphinxtab{USM pointer}

@snippet docs/snippets/gpu/remote_objects_creation.cpp wrap_usm_pointer

@endsphinxtab

@sphinxtab{cl_mem}

@snippet docs/snippets/gpu/remote_objects_creation.cpp wrap_cl_mem

@endsphinxtab

@sphinxtab{cl::Buffer}

@snippet docs/snippets/gpu/remote_objects_creation.cpp wrap_cl_buffer

@endsphinxtab

@sphinxtab{cl::Image2D}

@snippet docs/snippets/gpu/remote_objects_creation.cpp wrap_cl_image

@endsphinxtab

@sphinxtab{biplanar NV12 surface}

@snippet docs/snippets/gpu/remote_objects_creation.cpp wrap_nv12_surface

@endsphinxtab

@endsphinxtabset
@endsphinxtab

@sphinxtab{Allocate device memory}

@sphinxtabset

@sphinxtab{USM host memory}

@snippet docs/snippets/gpu/remote_objects_creation.cpp allocate_usm_host

@endsphinxtab

@sphinxtab{USM device memory}

@snippet docs/snippets/gpu/remote_objects_creation.cpp allocate_usm_device

@endsphinxtab

@sphinxtab{cl::Buffer}

@snippet docs/snippets/gpu/remote_objects_creation.cpp allocate_cl_buffer

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

The `ov::intel_gpu::ocl::D3DContext` and `ov::intel_gpu::ocl::VAContext` classes are derived from `ov::intel_gpu::ocl::ClContext`.
Therefore, they provide the functionality described above and extend it
to allow creation of `ov::RemoteTensor` objects from `ID3D11Buffer`, `ID3D11Texture2D` pointers or the `VASurfaceID` handle respectively.

## Direct NV12 Video Surface Input

To support the direct consumption of a hardware video decoder output, the plugin accepts two-plane video
surfaces as arguments for the `create_tensor_nv12()` function, which creates a pair of `ov::RemoteTensor`
objects which represent the Y and UV planes.

To ensure that the plugin generates the correct execution graph for the NV12 dual-plane input, static preprocessing
should be added before model compilation:

@snippet snippets/gpu/preprocessing.cpp init_preproc

Since the `ov::intel_gpu::ocl::ClImage2DTensor` and its derived classes do not support batched surfaces, if batching and surface sharing are required
at the same time, inputs need to be set via the `ov::InferRequest::set_tensors` method with vector of shared surfaces for each plane:

@sphinxtabset

@sphinxtab{Single batch}

@snippet docs/snippets/gpu/preprocessing.cpp single_batch

@endsphinxtab

@sphinxtab{Multiple batches}

@snippet docs/snippets/gpu/preprocessing.cpp batched_case

@endsphinxtab

@endsphinxtabset


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
.. raw:: html

   <div class="collapsible-section" data-title="Queue and context sharing example">

@endsphinxdirective

@snippet snippets/gpu/queue_sharing.cpp queue_sharing

@sphinxdirective
.. raw:: html

   </div>

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

> **NOTE**: For low-level parameter usage examples, see the source code of user-side wrappers from the include files mentioned above.


@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="OpenCL Kernel Execution on a Shared Buffer">

@endsphinxdirective

This example uses the OpenCL context obtained from a compiled model object.

@snippet snippets/gpu/context_sharing.cpp context_sharing_get_from_ov

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective


@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Running GPU Plugin Inference within User-Supplied Shared Context">

@endsphinxdirective

@snippet snippets/gpu/context_sharing.cpp context_sharing_user_handle

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective


@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux">

@endsphinxdirective

@snippet snippets/gpu/context_sharing_va.cpp context_sharing_va

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## See Also

* ov::Core
* ov::RemoteTensor
