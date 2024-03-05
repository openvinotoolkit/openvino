.. {#openvino_docs_OV_UG_supported_plugins_CPU}

CPU Device
==========



.. meta::
   :description: The CPU plugin in the Intel® Distribution of OpenVINO™ toolkit
                 is developed to achieve high performance inference of neural
                 networks on Intel® x86-64 and Arm® CPUs.


The CPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit. It is developed to achieve high performance inference of neural networks on Intel® x86-64 and Arm® CPUs. The newer 11th generation and later Intel® CPUs provide even further performance boost, especially with INT8 models.
For an in-depth description of CPU plugin, see:

- `CPU plugin developers documentation <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md>`__.
- `OpenVINO Runtime CPU plugin source files <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/>`__.

.. note::
   The scope of the CPU plugin features and optimizations on Arm® may differ from Intel® x86-64. If the limitation is not mentioned explicitly, the feature is supported for all CPU architectures.


Device Name
###########################################################

The ``CPU`` device name is used for the CPU plugin. Even though there can be more than one physical socket on a platform, only one device of this kind is listed by OpenVINO.
On multi-socket platforms, load balancing and memory usage distribution between NUMA nodes are handled automatically.
In order to use CPU for inference, the device name should be passed to the ``ov::Core::compile_model()`` method:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/compile_model.py
         :language: py
         :fragment: [compile_model_default]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/compile_model.cpp
         :language: cpp
         :fragment: [compile_model_default]


Supported Inference Data Types
###########################################################

CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:

  - ``f32`` (Intel® x86-64, Arm®)
  - ``bf16`` (Intel® x86-64)
- Integer data types:

  - ``i32`` (Intel® x86-64, Arm®)
- Quantized data types:

  - ``u8`` (Intel® x86-64)
  - ``i8`` (Intel® x86-64)
  - ``u1`` (Intel® x86-64)

:doc:`Hello Query Device C++ Sample <openvino_sample_hello_query_device>` can be used to print out supported data types for all detected devices.


Quantized Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
The ``u1/u8/i8`` data types are used for quantized operations only, i.e., those are not selected automatically for non-quantized operations.

For more details on how to get a quantized model see the :doc:`low-precision optimization guide <openvino_docs_model_optimization_guide>`.

.. note::

   Arm® platforms execute quantized models in simulation mode: the whole model (including quantization operations) is executed in floating-point precision.


Floating Point Data Types Specifics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports the following floating-point data types as inference precision of internal primitives:

- ``f32`` (Intel® x86-64, Arm®)
- ``bf16`` (Intel® x86-64)

The default floating-point precision of a CPU primitive is ``f32``. To support the ``f16`` OpenVINO IR the plugin internally converts
all the ``f16`` values to ``f32`` and all the calculations are performed using the native precision of ``f32``.
On platforms that natively support ``bfloat16`` calculations (have the ``AVX512_BF16`` or ``AMX`` extension), the ``bf16`` type is automatically used instead
of ``f32`` to achieve better performance (see the `Execution Mode Hint <#execution-mode-hint>`__).
Thus, no special steps are required to run a ``bf16`` model. For more details about the ``bfloat16`` format, see
the `BFLOAT16 – Hardware Numerics Definition white paper <https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf>`__.

Using the ``bf16`` precision provides the following performance benefits:

- ``bfloat16`` data type allows using Intel® Advanced Matrix Extension (AMX), which provides dramatically faster computations on corresponding hardware in comparison with AVX512 or AVX2 instructions in many DL operation implementations.
- Reduced memory consumption since ``bfloat16`` data half the size of 32-bit float.

To check if the CPU device can support the ``bfloat16`` data type, use the :doc:`query device properties interface <openvino_docs_OV_UG_query_api>`
to query ``ov::device::capabilities`` property, which should contain ``BF16`` in the list of CPU capabilities:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference.py
         :language: py
         :fragment: [part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference0.cpp
         :language: cpp
         :fragment: [part0]


Inference Precision Hint
-----------------------------------------------------------

If the model has been converted to ``bf16``, the ``ov::hint::inference_precision`` is set to ``ov::element::bf16`` and can be checked via
the ``ov::CompiledModel::get_property`` call. The code below demonstrates how to get the element type:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference.py
         :language: py
         :fragment: [part1]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference1.cpp
         :language: cpp
         :fragment: [part1]

To infer the model in ``f32`` precision instead of ``bf16`` on targets with native ``bf16`` support, set the ``ov::hint::inference_precision`` to ``ov::element::f32``.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference.py
         :language: py
         :fragment: [part2]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/Bfloat16Inference2.cpp
         :language: cpp
         :fragment: [part2]


The ``Bfloat16`` software simulation mode is available on CPUs with Intel® AVX-512 instruction set that do not support the
native ``avx512_bf16`` instruction. This mode is used for development purposes and it does not guarantee good performance.
To enable the simulation, the ``ov::hint::inference_precision`` has to be explicitly set to ``ov::element::bf16``.

.. note::

   If ``ov::hint::inference_precision`` is set to ``ov::element::bf16`` on a CPU without native bfloat16 support or bfloat16 simulation mode, an exception is thrown.

.. note::

   Due to the reduced mantissa size of the ``bfloat16`` data type, the resulting ``bf16`` inference accuracy may differ from the ``f32`` inference,
   especially for models that were not trained using the ``bfloat16`` data type. If the ``bf16`` inference accuracy is not acceptable,
   it is recommended to switch to the ``f32`` precision. Also, the performance/accuracy balance can be managed using the ``ov::hint::execution_mode`` hint,
   see the `Execution Mode Hint <#execution-mode-hint>`__.

Execution Mode Hint
-----------------------------------------------------------
In case ``ov::hint::inference_precision`` is not explicitly set, one can use ``ov::hint::execution_mode`` hint to direct the run-time optimizations toward either better accuracy or better performance.
If ``ov::hint::execution_mode`` is set to ``ov::hint::ExecutionMode::PERFORMANCE`` (default behavior) and the platform natively supports ``bfloat16``
calculations (has the ``AVX512_BF16`` or ``AMX`` extension) then ``bf16`` type is automatically used instead of ``f32`` to achieve better performance.
If the accuracy in this mode is not good enough, then set ``ov::hint::execution_mode`` to ``ov::hint::ExecutionMode::ACCURACY`` to enforce the plugin to
use the ``f32`` precision in floating point calculations.

For more details and code examples, see the :doc:`Precision Control <openvino_docs_OV_UG_Precision_Control>`.

Supported Features
###########################################################

Multi-device Execution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If a system includes OpenVINO-supported devices other than the CPU (e.g. an integrated GPU), then any supported model can be executed on all the devices simultaneously.
This can be achieved by specifying ``MULTI:CPU,GPU.0`` as a target device in case of simultaneous usage of CPU and GPU.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/compile_model.py
         :language: py
         :fragment: [compile_model_multi]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/compile_model.cpp
         :language: cpp
         :fragment: [compile_model_multi]


For more details, see the :doc:`Multi-device execution <openvino_docs_OV_UG_Running_on_multiple_devices>` article.

Multi-stream Execution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If either ``ov::num_streams(n_streams)`` with ``n_streams > 1`` or ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)``
property is set for CPU plugin, then multiple streams are created for the model. In case of CPU plugin, each stream has its own
host thread, which means that incoming infer requests can be processed simultaneously. Each stream is pinned to its own group of
physical cores with respect to NUMA nodes physical memory usage to minimize overhead on data transfer between NUMA nodes.

For more details, see the :doc:`optimization guide <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`.

.. note::

   When it comes to latency, be aware that running only one stream on multi-socket platform may introduce additional overheads
   on data transfer between NUMA nodes. In that case it is better to use the ``ov::hint::PerformanceMode::LATENCY`` performance hint.
   For more details see the :doc:`performance hints <openvino_docs_OV_UG_Performance_Hints>` overview.

.. note::

   Multi-stream execution is not supported on Arm® platforms. Latency and throughput hints have identical behavior and use only one stream for inference.


Dynamic Shapes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU provides full functional support for models with dynamic shapes in terms of the opset coverage.

.. note::

   The CPU plugin does not support tensors with dynamically changing rank. In case of an attempt to infer a model with such tensors, an exception will be thrown.

Some runtime optimizations work better if the model shapes are known in advance. Therefore, if the input data shape is
not changed between inference calls, it is recommended to use a model with static shapes or reshape the existing model
with the static input shape to get the best performance.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/dynamic_shape.py
         :language: py
         :fragment: [static_shape]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/dynamic_shape.cpp
         :language: cpp
         :fragment: [static_shape]


For more details, see the :doc:`dynamic shapes guide <openvino_docs_OV_UG_DynamicShapes>`.

Preprocessing Acceleration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports a full set of the preprocessing operations, providing high performance implementations for them.
For more details, see :doc:`preprocessing API guide <openvino_docs_OV_UG_Preprocessing_Overview>`.


.. dropdown:: The CPU plugin support for handling tensor precision conversion is limited to the following ov::element types:

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

CPU supports Import/Export network capability. If model caching is enabled via the common OpenVINO™ ``ov::cache_dir`` property,
the plugin automatically creates a cached blob inside the specified directory during model compilation. This cached blob contains
partial representation of the network, having performed common runtime optimizations and low precision transformations.
The next time the model is compiled, the cached representation will be loaded to the plugin instead of the initial OpenVINO IR,
so the aforementioned transformation steps will be skipped. These transformations take a significant amount of time during
model compilation, so caching this representation reduces time spent for subsequent compilations of the model, thereby reducing
first inference latency (FIL).

For more details, see the :doc:`model caching <openvino_docs_OV_UG_Model_caching_overview>` overview.

Extensibility
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU plugin supports fallback on ``ov::Op`` reference implementation if the plugin does not have its own implementation for such operation.
That means that :doc:`OpenVINO™ Extensibility Mechanism <openvino_docs_Extensibility_UG_Intro>` can be used for the plugin extension as well.
Enabling fallback on a custom operation implementation is possible by overriding the ``ov::Op::evaluate`` method in the derived operation
class (see :doc:`custom OpenVINO™ operations <openvino_docs_Extensibility_UG_add_openvino_ops>` for details).

Stateful Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The CPU plugin supports stateful models without any limitations.

For details, see :doc:`stateful models guide <openvino_docs_OV_UG_stateful_models_intro>`.

Supported Properties
###########################################################

The plugin supports the following properties:

Read-write Properties
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

All parameters must be set before calling ``ov::Core::compile_model()`` in order to take effect or passed as additional argument to ``ov::Core::compile_model()``

- ``ov::enable_profiling``
- ``ov::hint::inference_precision``
- ``ov::hint::performance_mode``
- ``ov::hint::execution_mode``
- ``ov::hint::num_request``
- ``ov::hint::scheduling_core_type``
- ``ov::hint::enable_hyper_threading``
- ``ov::hint::enable_cpu_pinning``
- ``ov::num_streams``
- ``ov::affinity``
- ``ov::inference_num_threads``
- ``ov::cache_dir``
- ``ov::intel_cpu::denormals_optimization``
- ``ov::intel_cpu::sparse_weights_decompression_rate``

Read-only properties
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- ``ov::supported_properties``
- ``ov::available_devices``
- ``ov::range_for_async_infer_requests``
- ``ov::range_for_streams``
- ``ov::device::full_name``
- ``ov::device::capabilities``

.. note::
   ``ov::affinity`` is replaced by ``ov::hint::enable_cpu_pinning``. As such, it is deprecated in the 2024.0 release and will be removed in the 2025 release.

External Dependencies
###########################################################

For some performance-critical DL operations, the CPU plugin uses third-party libraries:

- `oneDNN <https://github.com/oneapi-src/oneDNN>`__ (Intel® x86-64, Arm®)
- `Compute Library <https://github.com/ARM-software/ComputeLibrary>`__ (Arm®)


Optimization guide
###########################################################

Multi-Threading Optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU inference will infer an input or multiple inputs in parallel on multiple logical processors.

User can use the following properties to limit available CPU resource for model inference. If the platform or operating system can support this behavior, OpenVINO Runtime will perform multi-threading scheduling based on limited available CPU resources.

- ``ov::inference_num_threads`` limits number of logical processors used for CPU inference.
  If the number set by the user is greater than the number of logical processors on the platform, multi-threading scheduler only uses the platform number for CPU inference.
- ``ov::hint::scheduling_core_type`` limits the type of CPU cores for CPU inference when user runs inference on a hybird platform that includes both Performance-cores (P-cores) with Efficient-cores (E-cores).
  If user platform only has one type of CPU cores, this property has no effect, and CPU inference always uses this unique core type.
- ``ov::hint::enable_hyper_threading`` limits the use of one or two logical processors per CPU core when platform has CPU hyperthreading enabled.
  If there is only one logical processor per CPU core, such as Efficient-cores, this property has no effect, and CPU inference uses all logical processors.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/multi_threading.py
         :language: python
         :fragment: [ov:intel_cpu:multi_threading:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/multi_threading.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:multi_threading:part0]


.. note::

   ``ov::hint::scheduling_core_type`` and ``ov::hint::enable_hyper_threading`` only support Intel® x86-64 CPU on Linux and Windows in current release.

In some use cases, OpenVINO Runtime will enable CPU threads pinning by default for better performance. User can also turn it on or off using property ``ov::hint::enable_cpu_pinning``. Disable threads pinning might be beneficial in complex applications with several workloads executed in parallel. The following table describes the default setting for ``ov::hint::enable_cpu_pinning`` in different use cases.

==================================================== ================================
 Use Case                                             Default Setting of CPU Pinning 
==================================================== ================================
 All use cases with Windows OS                        False
 Stream contains both Pcore and Ecore with Linux OS   False
 Stream only contains Pcore or Ecore with Linux OS    True
 All use cases with Mac OS                            False
==================================================== ================================

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/multi_threading.py
         :language: python
         :fragment: [ov:intel_cpu:multi_threading:part1]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/multi_threading.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:multi_threading:part1]


user can check the :doc:`optimization guide <openvino_docs_deployment_optimization_guide_tput_advanced>` for details on multi-stream execution

.. note::

   ``ov::hint::enable_cpu_pinning`` is not supported on multi-socket platforms with Windows OS.

Denormals Optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Denormal numbers (denormals) are non-zero, finite float numbers that are very close to zero, i.e. the numbers
in (0, 1.17549e-38) and (0, -1.17549e-38). In such cases, normalized-number encoding format does not have a capability
to encode the number and underflow will happen. The computation involving such numbers is extremely slow on much hardware.

As a denormal number is extremely close to zero, treating a denormal directly as zero is a straightforward
and simple method to optimize computation of denormals. This optimization does not comply with IEEE 754 standard.
If it causes unacceptable accuracy degradation, the ``denormals_optimization`` property is introduced to control this behavior.
If there are denormal numbers in use cases, and no or acceptable accuracy drop is seen, set the property to `True`
to improve performance, otherwise set it to ``False``. If it is not set explicitly by the property and the application
does not perform any denormals optimization as well, the optimization is disabled by default. After enabling
the ``denormals_optimization`` property, OpenVINO will provide a cross operation system/ compiler and safe optimization
on all platform when applicable.

There are cases when the application in which OpenVINO is used also performs this low-level denormals optimization.
If it is optimized by setting the FTZ(Flush-To-Zero) and DAZ(Denormals-As-Zero) flags in MXCSR register at the beginning
of the thread where OpenVINO is called, OpenVINO will inherit this setting in the same thread and sub-thread,
so there is no need to set the ``denormals_optimization`` property. In such cases, you are responsible for the
effectiveness and safety of the settings.

.. note::

   The ``denormals_optimization`` property must be set before calling ``compile_model()``.

To enable denormals optimization in the application, the ``denormals_optimization`` property must be set to ``True``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_denormals.py
         :language: python
         :fragment: [ov:intel_cpu:denormals_optimization:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_denormals.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:denormals_optimization:part0]


Sparse weights decompression (Intel® x86-64)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``Sparse weights`` are weights where most of the elements are zero. The ratio of the number of zero elements
to the number of all elements is called ``sparse rate``. Thus, we assume that ``sparse weights`` are weights
with a high sparse rate. In case of ``sparse weights``, we can store only non-zero values in memory using
special storage structures, which allows us to use memory more efficiently. In turn, this can give us better
performance in the high memory bound workloads (e.g., throughput scenario).

``Sparse weights decompression feature`` allows to pack weights for Matrix Multiplication operations directly
in the CPU plugin at the model compilation stage and store non-zero values in a special packed format. Then,
during the execution of the model, the weights are unpacked and used in the computational kernel. Since the
weights are loaded from DDR/L3 cache in the packed format this significantly decreases memory consumption
and as a consequence improve inference performance.

To use this feature, the user is provided with property ``sparse_weights_decompression_rate``, which can take
values from the interval \[0, 1\]. ``sparse_weights_decompression_rate`` defines sparse rate threshold: only operations
with higher sparse rate will be executed using ``sparse weights decompression feature``. The default value is ``1``,
which means the option is disabled.

.. note::

   ``Sparse weights decompression feature`` is disabled by default since overall speed-up highly depends on
   particular workload and for some cases the feature may introduce performance degradations.

Code examples of how to use ``sparse_weights_decompression_rate``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/cpu/ov_sparse_weights_decompression.py
         :language: python
         :fragment: [ov:intel_cpu:sparse_weights_decompression:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/cpu/ov_sparse_weights_decompression.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:sparse_weights_decompression:part0]


.. note::

   The ``sparse_weights_decompression_rate`` property must be set before calling ``compile_model()``.

Information about the layers in which the ``sparse weights decompression feature`` was applied can be obtained
from perf counters log. The "exec type" field will contain the implementation type with the "sparse" particle
("brgemm_avx512_amx_sparse_I8" in the example below):

.. code-block:: sh

   MatMul_1800         EXECUTED         layerType: FullyConnected         execType: brgemm_avx512_amx_sparse_I8 realTime (ms): 0.050000  cpuTime (ms): 0.050000

Limitations
-----------------------------------------------------------

Currently, the ``sparse weights decompression feature`` is supported with the following limitations:

1. Model should be quantized to int8 precision.
2. Feature is only supported for Matrix Multiplication operations.
3. HW target must have Intel AMX extension support (e.g., Intel® 4th Generation Xeon® processors (code name Sapphire Rapids)).
4. The number of input and output channels of the weights must be a multiple of 64.

Additional Resources
###########################################################

* :doc:`Inference Devices and Modes <openvino_docs_Runtime_Inference_Modes_Overview>`
* :doc:`Optimization guide <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`
* `CPU plugin developers documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/README.md>`__




