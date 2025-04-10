Remote Tensor API of GPU Plugin
===============================


.. meta::
   :description: The Remote Tensor API of GPU plugin in OpenVINO™ supports
                 interoperability with existing native APIs, such as OpenCL,
                 Microsoft DirectX, or VAAPI.


The GPU plugin implementation of the ``ov::RemoteContext`` and ``ov::RemoteTensor`` interfaces supports GPU
pipeline developers who need video memory sharing and interoperability with existing native APIs,
such as OpenCL, Microsoft DirectX, or VAAPI.

The ``ov::RemoteContext`` and ``ov::RemoteTensor`` interface implementation targets the need for memory sharing and
interoperability with existing native APIs, such as OpenCL, Microsoft DirectX, and VAAPI.
They allow you to avoid any memory copy overhead when plugging OpenVINO™ inference
into an existing GPU pipeline. They also enable OpenCL kernels to participate in the pipeline to become
native buffer consumers or producers of the OpenVINO™ inference.

There are two interoperability scenarios supported by the Remote Tensor API:

* The GPU plugin context and memory objects can be constructed from low-level device, display, or memory handles and used to create the OpenVINO™ ``ov::CompiledModel`` or ``ov::Tensor`` objects.
* The OpenCL context or buffer handles can be obtained from existing GPU plugin objects, and used in OpenCL processing on the application side.

Class and function declarations for the API are defined in the following files:

* Windows -- ``openvino/runtime/intel_gpu/ocl/ocl.hpp`` and ``openvino/runtime/intel_gpu/ocl/dx.hpp``
* Linux -- ``openvino/runtime/intel_gpu/ocl/ocl.hpp`` and ``openvino/runtime/intel_gpu/ocl/va.hpp``

The most common way to enable the interaction of your application with the Remote Tensor API is to use user-side utility classes
and functions that consume or produce native handles directly.

Context Sharing Between Application and GPU Plugin
###########################################################

GPU plugin classes that implement the ``ov::RemoteContext`` interface are responsible for context sharing.
Obtaining a context object is the first step in sharing pipeline objects.
The context object of the GPU plugin directly wraps OpenCL context, setting a scope for sharing the
``ov::CompiledModel`` and ``ov::RemoteTensor`` objects. The ``ov::RemoteContext`` object can be either created on top of
an existing handle from a native API or retrieved from the GPU plugin.

Once you have obtained the context, you can use it to compile a new ``ov::CompiledModel`` or create ``ov::RemoteTensor``
objects. For network compilation, use a dedicated flavor of ``ov::Core::compile_model()``, which accepts the context as an additional parameter.

Creation of RemoteContext from Native Handle
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To create the ``ov::RemoteContext`` object for user context, explicitly provide the context to the plugin using constructor for one
of ``ov::RemoteContext`` derived classes.

.. tab-set::

   .. tab-item:: Windows/C++
      :sync: windows-cpp

      .. tab-set::

         .. tab-item:: Create from cl_context
            :sync: create-from-cl-context

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_cl_context]

         .. tab-item:: Create from cl_queue
            :sync: create-from-cl-queue

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_cl_queue]

         .. tab-item:: Create from ID3D11Device
            :sync: create-from-id3d11device

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_d3d_device]

   .. tab-item:: Windows/C
      :sync: windows-c

      .. tab-set::

         .. tab-item:: Create from cl_context
            :sync: create-from-cl-context

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_cl_context]

         .. tab-item:: Create from cl_queue
            :sync: create-from-cl-queue

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_cl_queue]

         .. tab-item:: Create from ID3D11Device
            :sync: create-from-id3d11device

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_d3d_device]

   .. tab-item:: Linux/C++
      :sync: linux-cpp

      .. tab-set::

         .. tab-item:: Create from cl_context
            :sync: create-from-cl-context

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_cl_context]

         .. tab-item:: Create from cl_queue
            :sync: create-from-cl-queue

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_cl_queue]

         .. tab-item:: Create from VADisplay
            :sync: create-from-vadisplay

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [context_from_va_display]

   .. tab-item:: Linux/C
      :sync: linux-c

      .. tab-set::

         .. tab-item:: Create from cl_context
            :sync: create-from-cl-context

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_cl_context]

         .. tab-item:: Create from cl_queue
            :sync: create-from-cl-queue

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_cl_queue]

         .. tab-item:: Create from VADisplay
            :sync: create-from-vadisplay

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [context_from_va_display]

Getting RemoteContext from the Plugin
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all ``ov::CompiledModel`` objects created during this time share the same context.
Once the plugin options have been changed, the internal context is replaced by the new one.

To request the current default context of the plugin, use one of the following methods:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: Get context from Core
            :sync: get-context-core

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [default_context_from_core]

         .. tab-item:: Get context from compiled model
            :sync: get-context-compiled-model

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [default_context_from_model]

   .. tab-item:: C
      :sync: c

      .. tab-set::

         .. tab-item:: Get context from Core
            :sync: get-context-core

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [default_context_from_core]

         .. tab-item:: Get context from compiled model
            :sync: get-context-compiled-model

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [default_context_from_model]


Memory Sharing Between Application and GPU Plugin
###########################################################

The classes that implement the ``ov::RemoteTensor`` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use dedicated ``create_tensor`` or ``create_tensor_nv12`` methods
of the ``ov::RemoteContext`` sub-classes.
``ov::intel_gpu::ocl::ClContext`` has multiple overloads of ``create_tensor`` methods which allow to wrap pre-allocated native handles with the ``ov::RemoteTensor``
object or request plugin to allocate specific device memory. There also provides C APIs to do the same things with C++ APIs.
For more details, see the code snippets below:


.. tab-set::

   .. tab-item:: Wrap native handles/C++
      :sync: wrap-native-handles

      .. tab-set::

         .. tab-item:: USM pointer
            :sync: usm-pointer

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_usm_pointer]

         .. tab-item:: cl_mem
            :sync: cl-mem

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_cl_mem]

         .. tab-item:: cl::Buffer
            :sync: buffer

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_cl_buffer]

         .. tab-item:: cl::Image2D
            :sync: image2D

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_cl_image]

         .. tab-item:: biplanar NV12 surface
            :sync: biplanar-nv12-surface

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_nv12_surface]

   .. tab-item:: Allocate device memory/C++
      :sync: allocate-device-memory

      .. tab-set::

         .. tab-item:: USM host memory
            :sync: usm-host-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [allocate_usm_host]

         .. tab-item:: USM device memory
            :sync: usm-device-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [allocate_usm_device]

         .. tab-item:: cl::Buffer
            :sync: buffer

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation.cpp
               :language: cpp
               :fragment: [allocate_cl_buffer]

.. tab-set::

   .. tab-item:: Wrap native handles/C
      :sync: wrap-native-handles

      .. tab-set::

         .. tab-item:: USM pointer
            :sync: usm-pointer

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [wrap_usm_pointer]

         .. tab-item:: cl_mem
            :sync: cl-mem

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [wrap_cl_mem]

         .. tab-item:: cl::Buffer
            :sync: buffer

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
              :language: c
              :fragment: [wrap_cl_buffer]

         .. tab-item:: cl::Image2D
            :sync: image2D

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [wrap_cl_image]

         .. tab-item:: biplanar NV12 surface
            :sync: biplanar-nv12-surface

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [create_nv12_surface]

   .. tab-item:: Allocate device memory/C
      :sync: allocate-device-memory

      .. tab-set::

         .. tab-item:: USM host memory
            :sync: usm-host-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [allocate_usm_host]

         .. tab-item:: USM device memory
            :sync: usm-device-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/remote_objects_creation_c.cpp
               :language: c
               :fragment: [allocate_usm_device]

The ``ov::intel_gpu::ocl::D3DContext`` and ``ov::intel_gpu::ocl::VAContext`` classes are derived from ``ov::intel_gpu::ocl::ClContext``.
Therefore, they provide the functionality described above and extend it
to enable creation of ``ov::RemoteTensor`` objects from ``ID3D11Buffer``, ``ID3D11Texture2D``
pointers or the ``VASurfaceID`` handle, as shown in the examples below:


.. tab-set::

   .. tab-item:: ID3D11Buffer
      :sync: id3d11-buffer

      .. code-block:: cpp

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

   .. tab-item:: ID3D11Texture2D
      :sync: id3d11-texture

      .. code-block:: cpp

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

   .. tab-item:: VASurfaceID
      :sync: vasurfaceid

      .. code-block:: cpp

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


.. important::

   Currently, only sharing of D3D11 surfaces is supported via the
   `cl_intel_d3d11_nv12_media_sharing <https://github.com/KhronosGroup/OpenCL-Registry/blob/main/extensions/intel/cl_intel_d3d11_nv12_media_sharing.txt>`__
   extension, which provides interoperability between OpenCL and DirectX.


Direct NV12 Video Surface Input
###########################################################

To support the direct consumption of a hardware video decoder output, the GPU plugin accepts:

* Two-plane NV12 video surface input - calling the ``create_tensor_nv12()`` function creates
  a pair of ``ov::RemoteTensor`` objects, representing the Y and UV planes.
* Single-plane NV12 video surface input - calling the ``create_tensor()`` function creates one
  ``ov::RemoteTensor`` object, representing the Y and UV planes at once (Y elements before UV elements).
* NV12 to Grey video surface input conversion - calling the ``create_tensor()`` function creates one
  ``ov::RemoteTensor`` object, representing only the Y plane.

To ensure that the plugin generates a correct execution graph, static preprocessing
should be added before model compilation:

.. tab-set::

   .. tab-item:: two-plane
      :sync: two-plane

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes.cpp
               :language: cpp
               :fragment: [init_preproc]

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes_c.cpp
               :language: c
               :fragment: [init_preproc]

   .. tab-item:: single-plane
      :sync: single-plane

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_single_plane.cpp
         :language: cpp
         :fragment: [init_preproc]

   .. tab-item:: NV12 to Grey
      :sync: nv12-grey

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_to_gray.cpp
         :language: cpp
         :fragment: [init_preproc]


Since the ``ov::intel_gpu::ocl::ClImage2DTensor`` and its derived classes do not support batched surfaces,
if batching and surface sharing are required at the same time,
inputs need to be set via the ``ov::InferRequest::set_tensors`` method with vector of shared surfaces for each plane:

.. tab-set::

   .. tab-item:: Single Batch
      :sync: single-batch

      .. tab-set::

         .. tab-item:: two-plane
            :sync: two-plane

            .. tab-set::

               .. tab-item:: C++
                  :sync: cpp

                  .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes.cpp
                     :language: cpp
                     :fragment: [single_batch]

               .. tab-item:: C
                  :sync: cpp

                  .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes_c.cpp
                     :language: c
                     :fragment: [single_batch]

         .. tab-item:: single-plane
            :sync: single-plane

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_single_plane.cpp
               :language: cpp
               :fragment: [single_batch]

         .. tab-item:: NV12 to Grey
            :sync: nv12-grey

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_to_gray.cpp
               :language: cpp
               :fragment: [single_batch]

   .. tab-item:: Multiple Batches
      :sync: multiple-batches

      .. tab-set::

         .. tab-item:: two-plane
            :sync: two-plane

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes.cpp
               :language: cpp
               :fragment: [batched_case]

         .. tab-item:: single-plane
            :sync: single-plane

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_single_plane.cpp
               :language: cpp
               :fragment: [batched_case]

         .. tab-item:: NV12 to Grey
            :sync: nv12-grey

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_to_gray.cpp
               :language: cpp
               :fragment: [batched_case]


I420 color format can be processed in a similar way

Context & Queue Sharing
###########################################################

The GPU plugin supports creation of shared context from the ``cl_command_queue`` handle. In that case,
the ``opencl`` context handle is extracted from the given queue via OpenCL™ API, and the queue itself is used inside
the plugin for further execution of inference primitives. Sharing the queue changes the behavior of the ``ov::InferRequest::start_async()``
method to guarantee that submission of inference primitives into the given queue is finished before
returning control back to the calling thread.

This sharing mechanism allows performing pipeline synchronization on the app side and avoiding blocking the host thread
on waiting for the completion of inference. The pseudo-code may look as follows:

.. dropdown:: Queue and context sharing example

   .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/queue_sharing.cpp
      :language: cpp
      :fragment: [queue_sharing]


Limitations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Some primitives in the GPU plugin may block the host thread on waiting for the previous primitives before adding its kernels to the command queue. In such cases, the ``ov::InferRequest::start_async()`` call takes much more time to return control to the calling thread as internally it waits for a partial or full network completion. Examples of operations: Loop, TensorIterator, DetectionOutput, NonMaxSuppression
* Synchronization of pre/post processing jobs and inference pipeline inside a shared queue is user's responsibility.
* Throughput mode is not available when queue sharing is used, i.e., only a single stream can be used for each compiled model.

Low-Level Methods for RemoteContext and RemoteTensor Creation
#####################################################################

The high-level wrappers mentioned above bring a direct dependency on native APIs to the user program.
If you want to avoid the dependency, you still can directly use the ``ov::Core::create_context()``,
``ov::RemoteContext::create_tensor()``, and ``ov::RemoteContext::get_params()`` methods.
On this level, native handles are re-interpreted as void pointers and all arguments are passed
using ``ov::AnyMap`` containers that are filled with ``std::string, ov::Any`` pairs.
Two types of map entries are possible: descriptor and container.
Descriptor sets the expected structure and possible parameter values of the map.

For possible low-level properties and their description, refer to the header file:
`remote_properties.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2025/0/src/inference/include/openvino/runtime/intel_gpu/remote_properties.hpp>`__.

Examples
###########################################################

To see pseudo-code of usage examples, refer to the sections below.


.. NOTE::

   For low-level parameter usage examples, see the source code of user-side wrappers from the include files mentioned above.


.. dropdown:: OpenCL Kernel Execution on a Shared Buffer

   This example uses the OpenCL context obtained from a compiled model object.

   .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/context_sharing.cpp
      :language: cpp
      :fragment: [context_sharing_get_from_ov]

.. dropdown:: Running GPU Plugin Inference within User-Supplied Shared Context

   .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/context_sharing.cpp
      :language: cpp
      :fragment: [context_sharing_user_handle]

.. dropdown:: Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux

   .. tab-set::

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/context_sharing_va.cpp
            :language: cpp
            :fragment: [context_sharing_va]

      .. tab-item:: C
         :sync: c

         .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/context_sharing_va_c.cpp
            :language: c
            :fragment: [context_sharing_va]

See Also
#######################################

* `ov::Core <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_core.html>`__
* `ov::RemoteTensor <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_remote_tensor.html>`__

