Remote Tensor API of GPU Plugin
===============================


.. meta::
   :description: The Remote Tensor API of GPU plugin in OpenVINO™ supports
                 interoperability with native device memory via the Vulkan
                 runtime.


The GPU plugin implementation of the ``ov::RemoteContext`` and ``ov::RemoteTensor`` interfaces allows GPU
pipeline developers to share device memory between OpenVINO™ inference and native applications
without memory copy overhead.

The GPU plugin runs on the Vulkan runtime and supports sharing of USM-like device memory
through the low-level ``ov::AnyMap`` based API. Native handles are passed as ``void`` pointers and all
arguments are provided using ``ov::AnyMap`` containers filled with ``std::string, ov::Any`` pairs.
Two types of map entries are possible: descriptor and container.
Descriptor sets the expected structure and possible parameter values of the map.

For possible low-level properties and their description, refer to the header file:
`remote_properties.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2026/0/src/inference/include/openvino/runtime/intel_gpu/remote_properties.hpp>`__.

Supported memory types:

* ``USM_USER_BUFFER`` - shared USM pointer allocated by the user
* ``USM_HOST_BUFFER`` - shared USM pointer type with host allocation type allocated by the plugin
* ``USM_DEVICE_BUFFER`` - shared USM pointer type with device allocation type allocated by the plugin
* ``BUFFER_FROM_HANDLE`` - shared buffer from an external device memory handle
* ``CPU_VA`` - shared host pointer

Getting RemoteContext from the Plugin
###########################################################

If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all ``ov::CompiledModel`` objects created during this time share the same context.
Once the plugin options have been changed, the internal context is replaced by the new one.

To request the current default context of the plugin, use one of the following methods:

.. tab-set::

   .. tab-item:: C++

      .. tab-set::

         .. tab-item:: Get context from Core

            .. code-block:: cpp

               ov::Core core;
               auto context = core.get_default_context("GPU");

         .. tab-item:: Get context from compiled model

            .. code-block:: cpp

               auto context = compiled_model.get_context();

   .. tab-item:: C

      .. tab-set::

         .. tab-item:: Get context from Core

            .. code-block:: c

               ov_core_t* core = nullptr;
               ov_core_create(&core);
               ov_remote_context* context = nullptr;
               ov_core_get_default_context(core, "GPU", &context);

         .. tab-item:: Get context from compiled model

            .. code-block:: c

               ov_remote_context* context = nullptr;
               ov_compiled_model_get_context(compiled_model, &context);

Memory Sharing Between Application and GPU Plugin
###########################################################

The classes that implement the ``ov::RemoteTensor`` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use the ``create_tensor`` method
of the ``ov::RemoteContext`` implementation with the low-level properties:

.. code-block:: cpp

   ov::AnyMap tensor_params = {
       {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
       {ov::intel_gpu::mem_handle.name(), usm_pointer},
   };
   auto tensor = context.create_tensor(ov::element::f32, shape, tensor_params);

Device memory allocated by the plugin can be requested via the context parameters as well:

.. code-block:: cpp

   ov::AnyMap tensor_params = {
       {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER},
   };
   auto tensor = context.create_tensor(ov::element::f32, shape, tensor_params);

The device memory handle can be obtained back from the tensor with ``get_params()``
and reused by native Vulkan code.

Limitations
###########################################################

* Sharing of video decoder surfaces (VAAPI, DirectX) and OpenCL objects is not supported
  by the Vulkan runtime.
* Importing external buffer handles is supported only for the device memory handles
  that the runtime can interpret (see ``BUFFER_FROM_HANDLE``).

See Also
#######################################

* `ov::Core <https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_core.html>`__
* `ov::RemoteTensor <https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_remote_tensor.html>`__
