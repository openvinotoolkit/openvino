GPU Device
==========


.. toctree::
   :maxdepth: 1
   :hidden:

   gpu-device/remote-tensor-api-gpu-plugin

.. meta::
   :description: The GPU plugin in the Intel® Distribution of OpenVINO™ toolkit
                 is an OpenCL based plugin for inference of deep neural
                 networks on Intel® GPus.


The GPU plugin is an OpenCL based plugin for inference of deep neural networks on Intel GPUs, both integrated and discrete ones.
For an in-depth description of the GPU plugin, see:

- `GPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md>`__
- `OpenVINO Runtime GPU plugin source files <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/>`__
- `Start AI Development with Intel <https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html>`__

The GPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit. For more information on how to configure a system to use it, see the :doc:`GPU configuration <../../../get-started/install-openvino/configurations/configurations-intel-gpu>`.

Device Naming Convention
#######################################

* Devices are enumerated as ``GPU.X``, where ``X={0, 1, 2,...}`` (only Intel® GPU devices are considered).
* If the system has an integrated GPU, its ``id`` is always 0 (``GPU.0``).
* The order of other GPUs is not predefined and depends on the GPU driver.
* The ``GPU`` is an alias for ``GPU.0``.
* If the system does not have an integrated GPU, devices are enumerated, starting from 0.
* For GPUs with multi-tile architecture (multiple sub-devices in OpenCL terms), a specific tile may be addressed as ``GPU.X.Y``, where ``X,Y={0, 1, 2,...}``, ``X`` - id of the GPU device, ``Y`` - id of the tile within device ``X``

For demonstration purposes, see the :doc:`Hello Query Device C++ Sample <../../../get-started/learn-openvino/openvino-samples/hello-query-device>` that can print out the list of available devices with associated indices. Below is an example output (truncated to the device names only):

.. code-block:: sh

   ./hello_query_device
   Available devices:
       Device: CPU
   ...
       Device: GPU.0
   ...
       Device: GPU.1


Then, the device name can be passed to the ``ov::Core::compile_model()`` method, running on:

.. tab-set::

   .. tab-item:: default device

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
               :language: Python
               :fragment: compile_model_default_gpu

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
               :language: cpp
               :fragment: compile_model_default_gpu

   .. tab-item:: specific GPU

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
               :language: Python
               :fragment: compile_model_gpu_with_id

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
               :language: cpp
               :fragment: compile_model_gpu_with_id

   .. tab-item:: specific tile

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
               :language: Python
               :fragment: compile_model_gpu_with_id_and_tile

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
               :language: cpp
               :fragment: compile_model_gpu_with_id_and_tile

Supported Inference Data Types
#######################################

The GPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:

  - f32
  - f16

- Quantized data types:

  - u8
  - i8
  - u1

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
The ``u1``/``u8``/``i8`` data types are used for quantized operations only, which means that they are not selected automatically for non-quantized operations.
For more details on how to get a quantized model, refer to the :doc:`Model Optimization guide <../../model-optimization>`.

Floating-point precision of a GPU primitive is selected based on operation precision in the OpenVINO IR, except for the :doc:`<compressed f16 OpenVINO IR form <../../../openvino-workflow/model-preparation/conversion-parameters>`, which is executed in the ``f16`` precision.

.. note::

   The newer generation Intel Iris Xe and Xe MAX GPUs provide accelerated performance for i8/u8 models. Hardware acceleration for ``i8``/``u8`` precision may be unavailable on older generation platforms. In such cases, a model is executed in the floating-point precision taken from IR.
   Hardware support of ``u8``/``i8`` acceleration can be queried via the ``ov::device::capabilities`` property.

:doc:`Hello Query Device C++ Sample <../../../get-started/learn-openvino/openvino-samples/hello-query-device>` can be used to print out the supported data types for all detected devices.


Supported Features
#######################################

The GPU plugin supports the following features:

Automatic Device Selection
+++++++++++++++++++++++++++++++++++++++

If a system has multiple GPUs (for example, an integrated and a discrete Intel GPU), then any supported model can be executed on all GPUs simultaneously.
It is done by specifying ``AUTO:GPU.1,GPU.0`` as a target device, and adding the ``CUMULATIVE_THROUGHPUT`` parameter.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
         :language: Python
         :fragment: compile_model_auto

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
         :language: cpp
         :fragment: compile_model_auto


For more details, see the :doc:`Automatic Device Selection <auto-device-selection>`.

Automatic Batching
+++++++++++++++++++++++++++++++++++++++

The GPU plugin is capable of reporting ``ov::max_batch_size`` and ``ov::optimal_batch_size`` metrics with respect to the current hardware
platform and model. Therefore, automatic batching is enabled by default when ``ov::optimal_batch_size`` is ``> 1`` and ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)`` is set.
Alternatively, it can be enabled explicitly via the device notion, for example ``BATCH:GPU``.


.. tab-set::

   .. tab-item:: Batching via BATCH plugin

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
               :language: Python
               :fragment: compile_model_batch_plugin

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
               :language: cpp
               :fragment: compile_model_batch_plugin

   .. tab-item:: Batching via throughput hint

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.py
               :language: Python
               :fragment: compile_model_auto_batch

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/compile_model_gpu.cpp
               :language: cpp
               :fragment: compile_model_auto_batch


For more details, see the :doc:`Automatic batching <automatic-batching>`.

Multi-stream Execution
+++++++++++++++++++++++++++++++++++++++

If either the ``ov::num_streams(n_streams)`` with ``n_streams > 1`` or the ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)`` property is set for the GPU plugin,
multiple streams are created for the model. In the case of GPU plugin each stream has its own host thread and an associated OpenCL queue
which means that the incoming infer requests can be processed simultaneously.

.. note::

   Simultaneous scheduling of kernels to different queues does not mean that the kernels are actually executed in parallel on the GPU device.
   The actual behavior depends on the hardware architecture and in some cases the execution may be serialized inside the GPU driver.

When multiple inferences of the same model need to be executed in parallel, the multi-stream feature is preferred to multiple instances of the model or application.
The reason for this is that the implementation of streams in the GPU plugin supports weight memory sharing across streams, thus, memory consumption may be lower, compared to the other approaches.

For more details, see the :doc:`optimization guide <../optimize-inference>`.

Dynamic Shapes
+++++++++++++++++++++++++++++++++++++++


.. note::

   Currently, dynamic shape support for GPU is a preview feature and has the following limitations:

   - It mainly supports NLP models (Natural Language Processing). Not all operations and optimization passes support dynamic shapes.
     As a result, a given model may crash or experience significant performance drops.
   - Due to the dominant runtime overhead on the host device, dynamic shapes may perform worse than static shapes on a discrete GPU.
   - Dynamic rank is not supported.

The general description of what dynamic shapes are and how they are used can be found in
:doc:`dynamic shapes guide <../model-input-output/dynamic-shapes>`.
To support dynamic shape execution, the following basic infrastructures are implemented:

- Runtime shape inference: infers output shapes of each primitive for a new input shape at runtime.
- Shape agnostic kernels: new kernels that can run arbitrary shapes. If a shape-agnostic kernel is not available,
  the required kernel is compiled at runtime for each shape.
- Asynchronous kernel compilation: even when a shape-agnostic kernel is available,
  the GPU plugin compiles an optimal kernel for the given shape and preserves it in the in-memory cache for future use.
- In-memory cache: preserves kernels compiled at runtime and weights reordered for the specific kernels.

Bounded dynamic batch
-----------------------------------------------------------

It is worth noting that the internal behavior differs in the case of bounded-batch dynamic shapes,
which means that only the batch dimension is dynamic and it has a fixed upper bound.

While general dynamic shapes can run on one compiled model, for the bounded dynamic batch the GPU plugin creates ``log2(N)``
low-level execution graphs in batch sizes equal to the powers of 2, to emulate the dynamic behavior (``N`` - is the upper bound for the batch dimension here).
As a result, the incoming infer request with a specific batch size is executed via the minimal combination of internal networks.
For example, a batch size of 33 may be executed via two internal networks with batch sizes of 32 and 1.
This approach is adopted for performance reasons, but it requires more memory and increased compilation time for multiple copies of internal networks.

The code snippet below demonstrates examples of a bounded dynamic batch:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/dynamic_batch.py
         :language: Python
         :fragment: dynamic_batch

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/dynamic_batch.cpp
         :language: cpp
         :fragment: dynamic_batch


Notes for performance and memory consumption in dynamic shapes
--------------------------------------------------------------

- Extra CPU utilization during inference:

  - Shape inference for new input shapes
  - Kernel compilation in runtime for optimal kernel
  - Unfusion of the fused subgraph when fusing is not allowed for a runtime shape

- Higher memory consumption for in-memory cache

  - Optimal kernels and weights from the previously used shapes are preserved in in-memory cache for future use


Recommendations for performance improvement
-----------------------------------------------------------

- Use static shapes whenever possible

  - Static models can benefit from more aggressive optimizations, such as, constant propagation, fusing, and reorder optimization.
    If the same shape is used for a dynamic and a static model, performance is worse in the dynamic one.
    It is, therefore, recommended to reshape dynamic models to static ones, if the scenario allows.

- Use bounded dynamic shapes whenever possible

  - The GPU plugin needs to reallocate memory if the current shape is larger than the maximum of the previous shapes, which causes additional overhead.
  - Using a bounded dynamic shape will help to reduce such overhead. For example, use ``{ov::Dimension(1, 10), ov::Dimension(1, 384)}``
    instead of ``{ov::Dimension(-1), ov::Dimension(-1)}``.
  - Note that a bounded dynamic *batch* is handled differently as mentioned above.

- Use permanent cache, e.g., OpenVino model_cache, to reduce the runtime re-compilation overhead

  - GPU plugin deploys in-memory cache to store compiled kernels for previously used shapes,
    but the size of such an in-memory cache is limited. Therefore, it is recommended to use
    a permanent cache such as OpenVino model_cache. For more details, See
    :doc:`Model caching overview <../optimize-inference/optimizing-latency/model-caching-overview>`.

- The longer the inference sequence, the better throughput can be obtained, because it can
  leverage more compilation time during inference.

  - If the primitive has a shape-agnostic kernel and the static shape kernel for the current
    shape does not exist in the in-memory cache, the shape-agnostic kernel is used. Then, as
    mentioned above, optimal kernels for the current shapes are also asynchronously compiled
    in parallel for future use. If the application process removes the CompiledModel object
    and the GPU plugin is unusable, any not-yet-started compilation tasks for optimal kernels
    will be canceled. However, if the application process allows enough time for the enqueued
    asynchronous compilation tasks, the more optimal kernels become available, enabling better
    throughput. For example, running 200 inputs of
    ``{[1, 1], ..., [1, 50], [1, 1], ... , [1, 50], [1, 1], ..., [1, 50], [1, 1], ..., [1, 50]}``
    may achieve better throughput than running 100 inputs of ``{[1, 1], ..., [1, 50], [1, 1], ... , [1,50]}``.


Preprocessing Acceleration
+++++++++++++++++++++++++++++++++++++++

The GPU plugin has the following additional preprocessing options:

- The ``ov::intel_gpu::memory_type::surface`` and ``ov::intel_gpu::memory_type::buffer`` values for the ``ov::preprocess::InputTensorInfo::set_memory_type()`` preprocessing method. These values are intended to be used to provide a hint for the plugin on the type of input Tensors that will be set in runtime to generate proper kernels.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes.py
         :language: Python
         :fragment: init_preproc

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/gpu/preprocessing_nv12_two_planes.cpp
         :language: cpp
         :fragment: init_preproc


With such preprocessing, GPU plugin will expect ``ov::intel_gpu::ocl::ClImage2DTensor`` (or derived) to be passed for each NV12 plane via ``ov::InferRequest::set_tensor()`` or ``ov::InferRequest::set_tensors()`` methods.

For usage examples, refer to the :doc:`RemoteTensor API <gpu-device/remote-tensor-api-gpu-plugin>`.

For more details, see the :doc:`preprocessing API <../optimize-inference/optimize-preprocessing>`.

Model Caching
+++++++++++++++++++++++++++++++++++++++

Model Caching helps reduce application startup delays by exporting and reusing
the compiled model automatically. The cache for the GPU plugin may be enabled
via the common OpenVINO ``ov::cache_dir`` property.

This means that all plugin-specific model transformations are executed on each ``ov::Core::compile_model()``
call, regardless of the ``ov::cache_dir`` option. Still, since kernel compilation is a bottleneck in the model
loading process, a significant load time reduction can be achieved.
Currently, GPU plugin implementation fully supports static models only. For dynamic models,
kernel caching is used instead and multiple ‘.cl_cache’ files are generated along with the ‘.blob’ file.

For more details, see the :doc:`Model caching overview <../optimize-inference/optimizing-latency/model-caching-overview>`.

Extensibility
+++++++++++++++++++++++++++++++++++++++

For information on this subject, see the :doc:`GPU Extensibility <../../../documentation/openvino-extensibility/custom-gpu-operations>`.

GPU Context and Memory Sharing via RemoteTensor API
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For information on this subject, see the :doc:`RemoteTensor API of GPU Plugin <gpu-device/remote-tensor-api-gpu-plugin>`.

Supported Properties
#######################################

The plugin supports the properties listed below.

Read-write properties
+++++++++++++++++++++++++++++++++++++++

All parameters must be set before calling ``ov::Core::compile_model()`` in order to take effect or passed as additional argument to ``ov::Core::compile_model()``.

- ``ov::cache_dir``
- ``ov::enable_profiling``
- ``ov::hint::model_priority``
- ``ov::hint::performance_mode``
- ``ov::hint::execution_mode``
- ``ov::hint::num_requests``
- ``ov::hint::inference_precision``
- ``ov::num_streams``
- ``ov::compilation_num_threads``
- ``ov::device::id``
- ``ov::intel_gpu::hint::host_task_priority``
- ``ov::intel_gpu::hint::queue_priority``
- ``ov::intel_gpu::hint::queue_throttle``
- ``ov::intel_gpu::enable_loop_unrolling``
- ``ov::intel_gpu::disable_winograd_convolution``

Read-only Properties
+++++++++++++++++++++++++++++++++++++++

- ``ov::supported_properties``
- ``ov::available_devices``
- ``ov::range_for_async_infer_requests``
- ``ov::range_for_streams``
- ``ov::optimal_batch_size``
- ``ov::max_batch_size``
- ``ov::device::full_name``
- ``ov::device::type``
- ``ov::device::gops``
- ``ov::device::capabilities``
- ``ov::intel_gpu::device_total_mem_size``
- ``ov::intel_gpu::uarch_version``
- ``ov::intel_gpu::execution_units_count``
- ``ov::intel_gpu::memory_statistics``

Limitations
#######################################

In some cases, the GPU plugin may implicitly execute several primitives on CPU using internal implementations, which may lead to an increase in CPU utilization.
Below is a list of such operations:

- Proposal
- NonMaxSuppression
- DetectionOutput

The behavior depends on specific parameters of the operations and hardware configuration.

.. important::

   While working on a fine tuned model, inference may give an inaccuracy and performance drop
   on GPU if winograd convolutions are selected. This issue can be fixed by disabling winograd
   convolutions:

   .. code:: bash

      compiled_model = core.compile_model(ov_model, device_name=devStr1, config={ "GPU_DISABLE_WINOGRAD_CONVOLUTION": True })

GPU Performance Checklist: Summary
#######################################

Since OpenVINO relies on the OpenCL kernels for the GPU implementation, many general OpenCL tips apply:

-	Prefer ``FP16`` inference precision over ``FP32``, as Model Conversion API can generate both variants, and the ``FP32`` is the default. To learn about optimization options, see :doc:`Optimization Guide <../../model-optimization>`.
- Try to group individual infer jobs by using :doc:`automatic batching <automatic-batching>`.
-	Consider :doc:`caching <../optimize-inference/optimizing-latency/model-caching-overview>` to minimize model load time.
-	If your application performs inference on the CPU alongside the GPU, or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. :doc:`CPU configuration options <cpu-device>` can be used to limit the number of inference threads for the CPU plugin.
-	Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-loop polling for completion. If CPU load is a concern, consider the dedicated ``queue_throttle`` property mentioned previously. Note that this option may increase inference latency, so consider combining it with multiple GPU streams or :doc:`throughput performance hints <../optimize-inference/high-level-performance-hints>`.
- When operating media inputs, consider :doc:`remote tensors API of the GPU Plugin <gpu-device/remote-tensor-api-gpu-plugin>`.


Additional Resources
#######################################

* `Working with GPUs in OpenVINO™ Notebook <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device/>`__
* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`.
* :doc:`Optimization guide <../optimize-inference>`.
* `GPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md>`__


