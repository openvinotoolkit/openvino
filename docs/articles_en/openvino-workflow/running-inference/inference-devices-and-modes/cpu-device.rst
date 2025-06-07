CPU Device
===============================================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   cpu-device/performance-hint-and-thread-scheduling

.. meta::
   :description: The CPU plugin in the Intel® Distribution of OpenVINO™ toolkit
                 is developed to achieve high performance inference of neural
                 networks on Intel® x86-64 and Arm® CPUs.


The CPU plugin is developed to achieve high performance inference of neural networks
on Intel® x86-64 and Arm® CPUs. The newer 11th generation and later Intel® CPUs
provide even further performance boost, especially with INT8 models.
For an in-depth description of CPU plugin, see:

- `CPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/docs>`__,
- `OpenVINO Runtime CPU plugin source files <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/>`__.

.. note::

   * | Features and optimizations of the CPU plugin on Arm® may differ from those on
       Intel® x86-64.
     | If limitation is not mentioned explicitly, the feature is supported for
       all CPU architectures.
   * **CPU inference on ARM64 is not supported for Windows.**


Device Name
###############################################################################################

The ``CPU`` device name is used for the CPU plugin. OpenVINO lists only one device of a type,
regardless of the number of physical sockets on a platform.
On multi-socket platforms, load balancing and memory distribution between NUMA nodes are
handled automatically. To use CPU for inference, pass the device name to the
``ov::Core::compile_model()`` method:

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
###############################################################################################

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

:doc:`Hello Query Device C++ Sample <../../../get-started/learn-openvino/openvino-samples/hello-query-device>`
can be used to print out supported data types for all detected devices.

Quantized Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Selected precision of each primitive depends on the operation precision in IR,
quantization primitives, and available hardware capabilities.
The ``u1/u8/i8`` data types are used for quantized operations only, that is,
the types are not selected automatically for non-quantized operations.

For more details on how to get a quantized model, see the
:doc:`low-precision optimization guide <../../model-optimization>`.

.. note::

   Arm® platforms execute quantized models in simulation mode: a whole model
   (including quantization operations) is executed in floating-point precision.


Floating Point Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports the following floating-point data types as inference precision of
internal primitives:

- ``f32`` (Intel® x86-64, Arm®)
- ``bf16`` (Intel® x86-64)
- ``f16`` (Intel® x86-64, Arm®)

The default floating-point precision of a CPU primitive is ``f32``. To support the
``f16`` OpenVINO IR on platforms that do not natively support ``float16``, the plugin
internally converts all the ``f16`` values to ``f32``, and all calculations are
performed using the native precision of ``f32``. On platforms that natively support
half-precision calculations (``bfloat16`` or ``float16``), the half-precision type
(``bf16`` or ``f16``) is automatically used instead of ``f32`` to achieve better performance
(see the `Execution Mode Hint <#execution-mode-hint>`__). Therefore, no special steps are
required to run a model with ``bf16`` or ``f16`` inference precision.

.. important::

   The ``bf16`` floating-point precision appears to impact the inference accuracy in LLM
   models. For more details, refer to this :ref:`article <limited_inference_precision>`.


Using the half-precision provides the following performance benefits:

- ``bfloat16`` and ``float16`` enable Intel® Advanced Matrix Extension (AMX)
  on 4+ generation Intel® Xeon® Scalable Processors, resulting in significantly faster
  computations on the corresponding hardware compared to AVX512 or AVX2.
- ``float16`` enables the ``armv8.2-a+fp16`` extension on ARM64 CPUs,
  which significantly improves performance due to the doubled vector capacity.
- Memory footprint is reduced since most weight and activation tensors are stored
  in half-precision.

For more details about the ``bfloat16`` format, see the
`BFLOAT16 - Hardware Numerics Definition white paper <https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf>`__.
To check if the CPU device supports the half-precision data type, use the
:doc:`query device properties interface <query-device-properties>`
to query ``ov::device::capabilities`` property for ``FP16`` or
``BF16``:

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
-----------------------------------------------------------------------------------------------

If the model has been converted to half-precision (``bf16`` or ``f16``), the
``ov::hint::inference_precision`` is set to ``ov::element::f16`` or ``ov::element::bf16``
and can be checked via the ``ov::CompiledModel::get_property`` call.
The code below demonstrates how to get the element type:

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

To infer the model in ``f32`` precision instead of half-precision (``bf16`` or ``f16``)
on targets with native half-precision support, set the ``ov::hint::inference_precision``
to ``ov::element::f32``.

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


The ``Bfloat16`` software simulation mode is available on CPUs with Intel® AVX-512
instruction set that do not support the native ``avx512_bf16`` instruction. This mode is
used for development purposes and it does not guarantee good performance.
To enable the simulation, set ``ov::hint::inference_precision`` to ``ov::element::bf16``.

.. note::

   * If ``ov::hint::inference_precision`` is set to ``ov::element::bf16`` on a CPU without
     native ``bfloat16`` support or ``bfloat16`` simulation mode, an exception is thrown.
   * Due to the reduced mantissa size of half-precision data types (``bfloat16`` or
     ``float16``), the resulting accuracy differs in the half-precision and the ``f32``
     inference, especially for models that were not trained using half-precision data types.
   * If the accuracy of half-precision inference is not acceptable,
     it is recommended to switch to the ``f32`` precision. Also, the performance/accuracy
     balance can be managed with the ``ov::hint::execution_mode`` hint,
     see the `Execution Mode Hint <#execution-mode-hint>`__.


Execution Mode Hint
-----------------------------------------------------------------------------------------------
If ``ov::hint::inference_precision`` is not explicitly set, you can use
``ov::hint::execution_mode`` hint to direct the run-time optimizations toward either
better accuracy or better performance.
When ``ov::hint::execution_mode`` is set to ``ov::hint::ExecutionMode::PERFORMANCE``
(default behavior) and the platform natively supports half-precision calculations
(``bfloat16`` or ``float16``), then ``bf16`` or ``f16`` is automatically used instead of
``f32`` to achieve better performance. If the accuracy in this mode is not good enough,
set ``ov::hint::execution_mode`` to ``ov::hint::ExecutionMode::ACCURACY`` to enforce the
plugin to use the ``f32`` precision in floating point calculations.

For more details and code examples, see the :doc:`Precision Control <../optimize-inference/precision-control>`.

Supported Features
###############################################################################################

Automatic Device Selection
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Inference of any supported model can be executed simultaneously on all supported devices
available on the system (for example, a CPU and an integrated GPU).
Simply specify ``AUTO:CPU,GPU.0`` as a target device and add the ``CUMULATIVE_THROUGHPUT``
parameter.

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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Multiple streams are created for the model when either ``ov::num_streams(n_streams)`` with
``n_streams > 1`` or ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)``
is set for CPU plugin. In case of CPU plugin, each stream has its own host thread, so the
incoming infer requests can be processed simultaneously. Each stream is pinned to its own
group of physical cores with respect to physical memory usage of NUMA nodes to minimize
overhead on data transfer between the NUMA nodes.

For more details, see the
:doc:`optimization guide <../optimize-inference>` and
:doc:`thread scheduling introduction <cpu-device/performance-hint-and-thread-scheduling>`.

.. important::

   Latency-wise, running only one stream on multi-socket platform may introduce additional
   overhead on data transfer between NUMA nodes. Therefore, it is better to use the
   ``ov::hint::PerformanceMode::LATENCY`` performance hint. For more details see the
   :doc:`performance hints <../optimize-inference/high-level-performance-hints>` overview.

Dynamic Shapes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU provides full functional support for models with dynamic shapes in terms of the
opset coverage.

.. note::

   The CPU plugin does not support tensors with dynamically changing rank. In case of
   an attempt to infer a model with such tensors, an exception will be thrown.

Some runtime optimizations work better if the model shapes are known in advance.
Therefore, if the input data shape is not changed between inference calls, it is
recommended to use a model with static shapes or reshape the existing model
with the static input shape to get the best performance.


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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports a full set of the preprocessing operations, providing high performance
implementations for them.
For more details, see
:doc:`preprocessing API guide <../optimize-inference/optimize-preprocessing>`.


The CPU plugin support for handling tensor precision conversion is limited to the following
``ov::element types``:

.. table::
   :widths: 20 20 50
   :align: left

   +---------------+------------+------------+
   | * ``bf16``    |  * ``i8``  |  * ``u8``  |
   | * ``f16``     |  * ``i16`` |  * ``u16`` |
   | * ``f32``     |  * ``i32`` |  * ``u32`` |
   | * ``f64``     |  * ``i64`` |  * ``u64`` |
   | * ``boolean`` |            |            |
   +---------------+------------+------------+

Model Caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If your :ref:`CPU supports import and export <import_export_device_capabilities>` of models,
you can use model caching by specifying the ``ov::cache_dir`` property. When the caching
is used during model compilation:

* the plugin automatically creates a cached blob inside the specified directory,
* the cached blob contains partial model representation with common runtime
  optimizations and low precision transformations applied,
* the cached model representation is loaded to the plugin instead of the initial OpenVINO IR,
  so the aforementioned transformations will not be reapplied during compilation,
* first inference latency (FIL) is reduced since the subsequent compilations of the model
  are significantly fast, due to the time-consuming transformations being skipped.

For more details, see the :doc:`model caching <../optimize-inference/optimizing-latency/model-caching-overview>`
overview.

Extensibility
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports fallback on implementation of ``ov::Op`` if the plugin does not have its
own implementation of such operation. It means that
:doc:`OpenVINO™ Extensibility Mechanism <../../../documentation/openvino-extensibility>`
can be used for the plugin extension as well. Enabling fallback on a custom operation
implementation is possible by overriding the ``ov::Op::evaluate`` method in the
derived operation class (see
:doc:`custom OpenVINO™ operations <../../../documentation/openvino-extensibility/custom-openvino-operations>`
for details).

Stateful Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The CPU plugin supports stateful models without any limitations.

For details, see :doc:`stateful models guide <../inference-request/stateful-models>`.

Supported Properties
###############################################################################################

The plugin supports the following properties:

.. tab-set::

   .. tab-item:: Read-write properties

      In order to take effect, all parameters must be set or passed as additional arguments
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
###############################################################################################

For some performance-critical DL operations, the CPU plugin uses third-party libraries:

- `oneDNN <https://github.com/oneapi-src/oneDNN>`__ (Intel® x86-64, Arm®)
- `Compute Library <https://github.com/ARM-software/ComputeLibrary>`__ (Arm®)

Optimization guide
###############################################################################################

Multi-Threading Optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU inference of a single or multiple inputs can be run in parallel on several
logical processors. For more details, see the
:doc:`thread scheduling introduction <cpu-device/performance-hint-and-thread-scheduling>`.


Optimization of Denormals
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Denormal numbers (denormals) are non-zero, finite float numbers that are very close to zero:
from (0, 1.17549e-38) to (0, -1.17549e-38). The computation involving such numbers is
usually extremely slow. Also, the numbers cannot be encoded by normalized number encoding
format, which results in underflow. To prevent it, you should treat denormals as
zero, which is a straightforward method to optimize computation of such numbers.
This method does not comply with IEEE 754 standard though, and may significantly impact
the accuracy.

To avoid any possible unacceptable accuracy degradation and to control such optimization,
you can enable the ``denormals_optimization`` property. By default, the property is disabled,
so you need to set the property to ``True``. Enable it to improve performance when denormals
are in use, and no or acceptable accuracy drop is seen. Using ``denormals_optimization`` will
provide a safe and cross operation system/compiler optimization on all supported platforms.

There are cases when low-level optimization of denormals is performed when the FTZ
(Flush-To-Zero) and DAZ (Denormals-As-Zero) flags are set in MXCSR register at the beginning
of the thread where ``ov:core`` is called. In such cases, the ``denormals_optimization``
property will automatically be inherited in the same thread and a sub-thread.
Keep in mind that the settings you provide influence the effectiveness and safety.

.. note::

   The ``denormals_optimization`` property must be set before calling ``compile_model()``.


To enable optimization of denormals, set the ``denormals_optimization`` property to ``True``:

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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Sparse weights are weights where most of the elements are zero. The weights usually have
a high sparse rate - the ratio of the number of zero elements to the number of all
elements. With sparse weights, you can store only non-zero values in memory using
special storage structures, which enables you to use memory more efficiently. In turn, this
can provide better performance in the high memory bound workloads (for example, throughput
scenario).

Sparse weights decompression feature enables packing weights for Matrix Multiplication
operations directly in the CPU plugin at the model compilation stage and storing non-zero
values in a special packed format. During the execution of the model, the weights are
unpacked and used in the computational kernel. Loading the weights from DDR/L3 cache
in the packed format significantly decreases memory consumption, thus improving
inference performance.

To use this feature, specify the ``sparse_weights_decompression_rate`` option,
which can take values from the \[0, 1\] interval. ``sparse_weights_decompression_rate``
defines sparse rate threshold: only operations with higher sparse rate will be
executed using sparse weights decompression feature. The default value is ``1``, which
means the option is disabled.

.. note::

   * Sparse weights decompression feature is disabled by default since overall speed-up
     highly depends on workload. In some cases, the feature may introduce
     performance degradations.
   * The ``sparse_weights_decompression_rate`` property must be set before calling
     ``compile_model()``.


Code examples below demonstrate how to use ``sparse_weights_decompression_rate``:

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


You can get information about the layers, in which the
sparse weights decompression feature was applied from performance counters log.
The *“exec type”* field will contain the implementation type with the *“sparse”* particle
(*“brgemm_avx512_amx_sparse_I8”* in the example below):

.. code-block:: sh

   MatMul_1800         EXECUTED         layerType: FullyConnected         execType: brgemm_avx512_amx_sparse_I8 realTime (ms): 0.050000  cpuTime (ms): 0.050000


Limitations
-----------------------------------------------------------------------------------------------

Currently, the sparse weights decompression feature has the following limitations:

1. A model should be quantized to the *int8* precision.
2. The feature is only supported for the Matrix Multiplication operations.
3. A target HW device must be supported by the Intel AMX extension (for example, Intel®
   4th Generation Xeon® processors, code-named *Sapphire Rapids*).
4. The number of input and output channels of the weights must be a multiple of 64.

Additional Resources
###############################################################################################

* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`
* :doc:`Optimization guide <../optimize-inference>`
* `CPU plugin developer documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/README.md>`__
