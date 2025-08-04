CPU Device
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   cpu-device/performance-hint-and-thread-scheduling

.. meta::
   :description: Explore the capabilities, performance features, and configuration options of the OpenVINO™ CPU plugin, including support for threading, precision control, caching, sparse weights, and runtime optimizations on Intel® x86-64 and Arm® CPUs.


The CPU plugin is a part of the Intel® OpenVINO™ toolkit. It enables high-performance
inference of neural networks on Intel® x86-64 and Arm® CPUs. The newer 11th generation and
later Intel® CPUs provide even further performance boost, especially for INT8 models.
For more detailed description of CPU plugin, see:

- `CPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/docs>`__.
- `OpenVINO Runtime CPU plugin source files <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/>`__.

.. note::

   Features and optimizations of the CPU plugin on Arm® may differ from Intel® x86-64.
   If the limitation is not mentioned explicitly, the feature is supported for
   all CPU architectures. **CPU inference on ARM64 is not supported for Windows.**


Device Name
###########################################################

The ``CPU`` device name is used for the CPU plugin. Even if a platform has multiple physical
sockets, OpenVINO lists it as a single CPU device.
On multi-socket platforms, load balancing and memory usage distribution between NUMA nodes are
handled automatically. To use CPU for inference, pass the device name to
the ``ov::Core::compile_model()`` method:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_cpu.py
         :language: py
         :fragment: [compile_model_default]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_cpu.cpp
         :language: cpp
         :fragment: [compile_model_default]


Supported Model Precision
#########################

CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:

  - ``FP32`` (Intel® x86-64, Arm®)
  - ``BF16`` (Intel® x86-64)
  - ``FP16`` (Intel® x86-64, Arm®)
  - :ref:`MXFP4 <mxfp4_support>` (Intel® x86-64)

- Integer data types:

  - ``INT32`` (Intel® x86-64, Arm®)

- Quantized data types:

  - ``uINT8`` (Intel® x86-64)
  - ``INT8`` (Intel® x86-64)
  - ``uINT1`` (Intel® x86-64)

Use :doc:`Hello Query Device C++ Sample <../../../get-started/learn-openvino/openvino-samples/hello-query-device>`
to print out supported data types for all detected devices.


Quantized Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Selected precision of each primitive depends on the operation precision in IR,
quantization primitives, and available hardware capabilities. The ``u1/u8/i8`` data types are
used for quantized operations only. The types are not selected automatically for non-quantized operations.

For more details on how to get a quantized model, see the :doc:`low-precision optimization guide <../../model-optimization>`.

.. note::

   Arm® platforms execute quantized models in simulation mode: the whole model, including quantization operations, is executed in floating-point precision.


Floating-Point Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports the following floating-point data types as inference precision of internal primitives:

- ``f32`` (Intel® x86-64, Arm®)
- ``bf16`` (Intel® x86-64)
- ``f16`` (Intel® x86-64, Arm®)

The default floating-point precision of a CPU primitive is ``f32``. To support the ``f16``
OpenVINO IR on platforms that do not natively support ``float16``, the plugin internally converts
all the ``f16`` values to ``f32``, and all calculations run using the native ``f32`` precision.
On platforms that support half-precision calculations (bfloat16 or float16), the plugin uses
the half-precision type (``bf16`` or ``f16``) automatically instead of ``f32`` to improve performance
(see the `Execution Mode Hint <#execution-mode-hint>`__).
No special steps are required to run a model with ``bf16`` or ``f16`` inference precision.

.. important::

   The ``bf16`` floating-point precision may affect inference accuracy in LLM models.
   For more details, refer to the :ref:`Precision Control article <limited_inference_precision>`.

Using the half-precision provides the following performance benefits:

- ``bfloat16`` and ``float16`` enable Intel® Advanced Matrix Extension (AMX)
  on 4th-gen and newer generation Intel® Xeon® Scalable Processors, resulting in significantly faster
  computations on the corresponding hardware compared to AVX512 or AVX2 for
  many deep learning operations.
- ``float16`` enables the ``armv8.2-a+fp16`` extension on ARM64 CPUs, significantly improving
  performance through doubled vector capacity.
- Memory footprint is reduced, as most weight and activation tensors are stored in half-precision.

For more details on the ``bfloat16`` format, see
the `BFLOAT16 – Hardware Numerics Definition white paper <https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf>`__.
To check if the CPU device supports half-precision types, use the :doc:`query device properties interface <query-device-properties>`
to query ``ov::device::capabilities`` property. If supported, ``FP16`` or ``BF16`` will appear in the list of CPU capabilities:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference.py
         :language: py
         :fragment: [part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference0.cpp
         :language: cpp
         :fragment: [part0]


Inference Precision Hint
-----------------------------------------------------------

If a model has been converted to half-precision (``bf16`` or ``f16``), the ``ov::hint::inference_precision``
is set to ``ov::element::f16`` or ``ov::element::bf16`` and can be checked via
the ``ov::CompiledModel::get_property`` call. The following code shows how to get the element type:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference.py
         :language: py
         :fragment: [part1]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference1.cpp
         :language: cpp
         :fragment: [part1]

To infer the model in ``f32`` precision instead of half-precision (``bf16`` or ``f16``) on
targets with native half-precision support, set the ``ov::hint::inference_precision`` to ``ov::element::f32``.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference.py
         :language: py
         :fragment: [part2]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/Bfloat16Inference2.cpp
         :language: cpp
         :fragment: [part2]


The ``Bfloat16`` software simulation mode is available on CPUs with Intel® AVX-512 instruction set that do not support the
native ``avx512_bf16`` instruction. This mode is used for development purposes and does not guarantee optimal performance.
To enable simulation, explicitly set ``ov::hint::inference_precision`` to ``ov::element::bf16``.

.. note::

   If ``ov::hint::inference_precision`` is set to ``ov::element::bf16`` on a CPU without native
   ``bfloat16`` support or ``bfloat16`` simulation mode, an exception is thrown.

.. note::

   Due to reduced mantissa size of half-precision types (``bfloat16`` or ``float16``),
   the resulting half-precision inference accuracy may differ from the ``f32`` inference,
   especially for models not trained in half-precision. If accuracy is not acceptable,
   it is recommended to switch to the ``f32`` precision. You can also balance performance and accuracy
   using the ``ov::hint::execution_mode`` hint. See the `Execution Mode Hint <#execution-mode-hint>`__ for details.

Execution Mode Hint
-----------------------------------------------------------
If ``ov::hint::inference_precision`` is not explicitly set, you can use ``ov::hint::execution_mode``
hint to direct the runtime optimizations toward either accuracy or performance.
When ``ov::hint::execution_mode`` is set to ``ov::hint::ExecutionMode::PERFORMANCE`` (the default),
and the platform supports half-precision (``bfloat16`` or ``float16``), the plugin
automatically uses ``bf16`` or ``f16``  instead of ``f32`` for better performance.
If accuracy is not acceptable, set ``ov::hint::execution_mode`` to ``ov::hint::ExecutionMode::ACCURACY``
to enforce ``f32`` precision.

For more details and code examples, see :doc:`Precision Control <../optimize-inference/precision-control>`.

Supported Features
###########################################################

Automatic Device Selection
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Any supported model can run on all available supported devices simultaneously.
For example, with a CPU and an integrated GPU, set ``AUTO:CPU,GPU.0`` as the target device,
and add the ``CUMULATIVE_THROUGHPUT`` parameter.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_cpu.py
         :language: py
         :fragment: [compile_model_auto]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_cpu.cpp
         :language: cpp
         :fragment: [compile_model_auto]


For more details, see the :doc:`Automatic Device Selection <auto-device-selection>`.

.. _multi_stream_execution:

Multi-stream Execution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The CPU plugin creates multiple streams for the model when either ``ov::num_streams(n_streams)`` with ``n_streams > 1``
or ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)`` is set.
Each stream has its own host thread, enabling simultaneous processing of inference requests.
Each stream is pinned to its own group of physical cores, aligned with NUMA nodes physical memory
usage, to minimize overhead on data transfer between NUMA nodes.

For more details, see the :doc:`optimization guide <../optimize-inference>` and :doc:`thread scheduling introduction <cpu-device/performance-hint-and-thread-scheduling>`.

.. note::

   On multi-socket platforms, running only one stream may increase the latency of
   data transfers between NUMA nodes. To reduce this overhead, use the ``ov::hint::PerformanceMode::LATENCY`` hint.
   For more details, see the :doc:`performance hints <../optimize-inference/high-level-performance-hints>` overview.

Dynamic Shapes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU provides full functional support for models with dynamic shapes in terms of the opset coverage.

.. note::

   The CPU plugin does not support tensors with dynamically changing rank. Inferring a model with such tensors will cause an exception.

Some runtime optimizations work better when model shapes are known in advance. If the input shape does
not changed between inference calls, it is recommended to use a model with static shapes or
reshape the existing model to a static input shape for better performance.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/dynamic_shape.py
         :language: py
         :fragment: [static_shape]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/dynamic_shape.cpp
         :language: cpp
         :fragment: [static_shape]


For more details, see the :doc:`dynamic shapes guide <../model-input-output/dynamic-shapes>`.

Preprocessing Acceleration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports the full set of preprocessing operations, providing high performance implementations for them.
For more details, see :doc:`preprocessing API guide <../optimize-inference/optimize-preprocessing>`.


.. dropdown:: The CPU plugin supports tensor precision conversion only for the following ov::element types:

   * ``bf16``
   * ``f16``
   * ``f32``
   * ``f64``
   * ``i8``
   * ``i16``
   * ``i32``
   * ``i64``
   * ``u8``
   * ``u16``
   * ``u32``
   * ``u64``
   * ``boolean``


Model Caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports Import/Export model capability. When model caching is enabled via the
common OpenVINO™ ``ov::cache_dir`` property, the plugin automatically creates a cached blob
during model compilation. This blob contains a partially optimized version of the model,
including common runtime optimizations and low-precision transformations.

On the next compilation, the plugin loads the cached version instead of the original OpenVINO IR.
This enables the plugin to skip the time-consuming transformation steps,
reducing model compilation time and first inference latency (FIL).

For more details, see the :doc:`model caching <../optimize-inference/optimizing-latency/model-caching-overview>` overview.

Extensibility
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports fallback on the ``ov::Op`` reference implementation if the plugin does not
have a native implementation for such operation. This means the :doc:`OpenVINO™ Extensibility Mechanism <../../../documentation/openvino-extensibility>`
can be used for the plugin extension as well.
To enable fallback for a custom operation, override the ``ov::Op::evaluate`` method in the derived operation
class. For more details, see :doc:`custom OpenVINO™ operations <../../../documentation/openvino-extensibility/custom-openvino-operations>`.

Stateful Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The CPU plugin supports stateful models without any limitations.

For details, see the :doc:`stateful models guide <../inference-request/stateful-models>`.

Supported Properties
###########################################################

The plugin supports the following properties:

.. tab-set::

   .. tab-item:: Read-write properties

      All parameters must be set or passed as additional arguments
      before calling the ``ov::Core::compile_model()`` method:

      - ``ov::enable_profiling``
      - ``ov::hint::inference_precision``
      - ``ov::hint::performance_mode``
      - ``ov::hint::execution_mode``
      - ``ov::hint::num_request``
      - ``ov::hint::scheduling_core_type``
      - ``ov::hint::enable_hyper_threading``
      - ``ov::hint::enable_cpu_reservation``
      - ``ov::hint::enable_cpu_pinning``
      - ``ov::num_streams``
      - ``ov::inference_num_threads``
      - ``ov::cache_dir``
      - ``ov::intel_cpu::denormals_optimization``
      - ``ov::intel_cpu::sparse_weights_decompression_rate``

   .. tab-item:: Read-only properties

      - ``ov::supported_properties``
      - ``ov::available_devices``
      - ``ov::range_for_async_infer_requests``
      - ``ov::range_for_streams``
      - ``ov::device::full_name``
      - ``ov::device::capabilities``


External Dependencies
###########################################################

For some performance-critical DL operations, the CPU plugin uses third-party libraries:

- `oneDNN <https://github.com/oneapi-src/oneDNN>`__ (Intel® x86-64, Arm®)
- `Compute Library <https://github.com/ARM-software/ComputeLibrary>`__ (Arm®)


Optimization guide
###########################################################

Multi-Threading Optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU inference will run one or multiple inputs in parallel on multiple logical processors.

For more details, see the :doc:`thread scheduling introduction <cpu-device/performance-hint-and-thread-scheduling>`.


Denormals Optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Denormal numbers (denormals) are non-zero, finite float numbers very close to zero, i.e. the numbers
in (0, 1.17549e-38) and (0, -1.17549e-38). The normalized-number encoding format cannot represent
such numbers, causing underflow. Calculations involving denormals are much slower on most hardware.

A common way to speed up computations is to treat denormals as zero, as they are extremely close to zero.
However, this optimization does not comply with IEEE 754 standard and may reduce accuracy.
To control this behavior, use the ``denormals_optimization`` property:

* If your use case includes denormals and accuracy remains acceptable, set  ``denormals_optimization`` to `True` to improve performance. Otherwise set it to ``False``.
* If the property is not set and the application does not optimize denormals, the optimization is turned off by default.

When ``denormals_optimization`` is enabled, OpenVINO applies a safe, cross-platform optimization
to handle denormals efficiently.

Some applications already optimize denormals by setting the FTZ (Flush-To-Zero) and DAZ (Denormals-As-Zero)
flags in the MXCSR register at the beginning of the thread where OpenVINO is called. In this case,
OpenVINO inherits these settings in the same thread and sub-thread,
so you do not need to set the ``denormals_optimization`` property. However, you should ensure
that the settings are effective and safe.

.. note::

   The ``denormals_optimization`` property must be set before calling ``compile_model()``.

To enable denormals optimization in the application, set the ``denormals_optimization``  to ``True``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_denormals.py
         :language: python
         :fragment: [ov:intel_cpu:denormals_optimization:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_denormals.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:denormals_optimization:part0]


Sparse weights decompression (Intel® x86-64)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

*Sparse weights* are weights where most elements are zero. The *sparse rate* is the ratio of the number of zero elements
to the number of all elements, so *sparse weights* have a high sparse rate. For *sparse weights*,
only non-zero values are stored in memory using special storage structures. This reduces memory usage
and can improve performance, especially in high memory-bound workloads (throughput scenarios).

*Sparse weights decompression feature* enables packing weights for Matrix Multiplication operations directly
in the CPU plugin at the model compilation stage and storing non-zero values in a special packed format.
During the model execution, these weights are unpacked and used in the computational kernel. Because
weights are loaded from DDR/L3 cache in the packed format, memory consumption significantly decreases,
leading to faster inference performance.

To enable this feature, use the property ``sparse_weights_decompression_rate``, which accepts
values from the interval \[0, 1\]. This value defines sparse rate threshold: only operations
with a higher sparse rate will be executed using sparse weights decompression. The default value is ``1``,
meaning the option is disabled.

.. note::
   The ``sparse_weights_decompression_rate`` property must be set **before** calling ``compile_model()``.
   Sparse weights decompression feature is **disabled by default**, since overall speed-up highly depends on
   particular workload and may introduce performance degradations.

Code examples of how to use ``sparse_weights_decompression_rate``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_sparse_weights_decompression.py
         :language: python
         :fragment: [ov:intel_cpu:sparse_weights_decompression:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_sparse_weights_decompression.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:sparse_weights_decompression:part0]



You can check which layers use the sparse weights decompression feature in the performance counters log.
The "exec type" field will contain the implementation type with the word "sparse".
For example, brgemm_avx512_amx_sparse_I8:

.. code-block:: sh

   MatMul_1800         EXECUTED         layerType: FullyConnected         execType: brgemm_avx512_amx_sparse_I8 realTime (ms): 0.050000  cpuTime (ms): 0.050000

Limitations
-----------------------------------------------------------

Currently, the ``sparse weights decompression feature`` is supported with the following limitations:

1. Model should be quantized to int8 precision.
2. Feature is only supported for Matrix Multiplication operations.
3. HW target must have Intel AMX extension support (for example, Intel® 4th Generation Xeon® processors (code name Sapphire Rapids)).
4. The number of input and output channels of the weights must be a multiple of 64.

Additional Resources
###########################################################

* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`
* :doc:`Optimization guide <../optimize-inference>`
* `CPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/README.md>`__




