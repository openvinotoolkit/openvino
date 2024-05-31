.. {#openvino_docs_OV_UG_supported_plugins_CPU_Hints_Threading}

Performance Hints and Threads Scheduling 
========================================

.. meta::
   :description: The Threads Scheduling of CPU plugin in OpenVINO™ Runtime
                 detects CPU architecture and sets low-level properties based
                 on performance hints automatically.

While all supported devices in OpenVINO offer low-level performance settings, it is advisable not to widely use these settings unless targeting specific platforms and models. The recommended approach is configuring performance in OpenVINO Runtime using high-level performance hints property ``ov::hint::performance_mode``. Performance hints ensure optimal portability and scalability of the applications across various platforms and models.

To simplify the configuration of hardware devices, OpenVINO offers two performance hints: the latency hint ``ov::hint::PerformanceMode::LATENCY`` and the throughput hint ``ov::hint::PerformanceMode::THROUGHPUT``.

- ``ov::inference_num_threads`` limits number of logical processors used for CPU inference.
  If the number set by the user is greater than the number of logical processors on the platform, multi-threading scheduler only uses the platform number for CPU inference.
- ``ov::num_streams`` limits number of infer requests that can be run in parallel.
  If the number set by the user is greater than the number of inference threads, multi-threading scheduler only uses the number of inference threads to ensure that there is at least one thread per stream.
- ``ov::hint::scheduling_core_type`` limits the type of CPU cores for CPU inference when user runs inference on a hybird platform that includes both Performance-cores (P-cores) with Efficient-cores (E-cores).
  If user platform only has one type of CPU cores, this property has no effect, and CPU inference always uses this unique core type.
- ``ov::hint::enable_hyper_threading`` limits the use of one or two logical processors per CPU core when platform has CPU hyperthreading enabled.
  If there is only one logical processor per CPU core, such as Efficient-cores, this property has no effect, and CPU inference uses all logical processors.
- ``ov::hint::enable_cpu_pinning`` enable CPU pinning during CPU inference. 
  If user enable this property but inference scenario cannot support it, this property will be disabled during model compilation. 

For additional details on the above configurations, refer to:

- `Multi-stream Execution <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#multi-stream-execution>`__

Latency Hint
###################################

In this scenario, the default setting of ``ov::hint::scheduling_core_type`` is determined by the model precision and the ratio of P-cores and E-cores.

.. note::

    P-cores is short for Performance-cores and E-cores is for Efficient-cores. These are available after 12th Gen Intel® Core™ Processor. 

.. _Core Type Table of Latency Hint:
+----------------------------+---------------------+---------------------+
|                            | INT8 model          | FP32 model          |
+============================+=====================+=====================+
| E-cores / P-cores < 2      | P-cores             | P-cores             |
+----------------------------+---------------------+---------------------+
| 2 <= E-cores / P-cores < 4 | P-cores             | P-cores and E-cores |
+----------------------------+---------------------+---------------------+
| 4 <= E-cores / P-cores     | P-cores and E-cores | P-cores and E-cores |
+----------------------------+---------------------+---------------------+

.. note::

   Both P-cores and E-cores may be used for any configuration starting from 14th Gen Intel® Core™ Processor on Windows.

Then the default settings of low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| Property                             | Windows                                                        | Linux                                                          |
+======================================+================================================================+================================================================+
| ``ov::num_streams``                  | 1                                                              | 1                                                              |
+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| ``ov::inference_num_threads``        | is equal to number of P-cores or P-cores+E-cores on one socket | is equal to number of P-cores or P-cores+E-cores on one socket |
+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| ``ov::hint::scheduling_core_type``   | `Core Type Table of Latency Hint`_                             | `Core Type Table of Latency Hint`_                             |
+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                                                             | No                                                             |
+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No / Not Supported                                             | Yes except using P-cores and E-cores together                  |
+--------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+

.. note::

    - ``ov::hint::scheduling_core_type`` might be adjusted for particular inferred model on particular platform based on internal heuristics to guarantee best performance.
    - Both P-cores and E-cores are used for the Latency Hint on Intel® Core™ Ultra Processors on Windows, except in the case of large language models.
    - In case hyper-threading is enabled, two logical processors share hardware resource of one CPU core. OpenVINO do not expect to use both logical processors in one stream for one infer request. So ``ov::hint::enable_hyper_threading`` is ``No`` in this scenario.
    - ``ov::hint::enable_cpu_pinning`` is disabled by default on Windows/Mac, and enabled on Linux. Such default settings are aligned with typical workloads running in corresponding environment to guarantee better OOB performance.

Throughput Hint
######################################

In this scenario, thread scheduling first evaluates the memory pressure of the model being inferred on the current platform, and determines the number of threads per stream, as shown below.

+-----------------+-----------------------+
| Memory Pressure | Threads per stream    |
+=================+=======================+
| low             | 1 P-core or 2 E-cores |
+-----------------+-----------------------+
| medium          | 2                     |
+-----------------+-----------------------+
| high            | 3 or 4 or 5           |
+-----------------+-----------------------+

Then the value of ``ov::num_streams`` is calculated as ``ov::inference_num_threads`` divided by the number of threads per stream. The default settings of low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+-------------------------------+-------------------------------+
| Property                             | Windows                       | Linux                         |
+======================================+===============================+===============================+
| ``ov::num_streams``                  | Calculate as above            | Calculate as above            |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::inference_num_threads``        | Number of P-cores and E-cores | Number of P-cores and E-cores |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::scheduling_core_type``   | P-cores and E-cores           | P-cores and E-cores           |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::enable_hyper_threading`` | Yes / No                      | Yes / No                      |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No                            | Yes                           |
+--------------------------------------+-------------------------------+-------------------------------+

.. note::

    - By default, different core types are not mixed within single stream in this scenario. And cores from different numa nodes are not mixed within single stream.

Multi-Threading Optimization
##############################################

User can use the following properties to limit available CPU resource for model inference. If the platform or operating system can support this behavior, OpenVINO Runtime will perform multi-threading scheduling based on limited available CPU.

- ``ov::inference_num_threads`` 
- ``ov::hint::scheduling_core_type`` 
- ``ov::hint::enable_hyper_threading`` 

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/multi_threading.py
         :language: python
         :fragment: [ov:intel_cpu:multi_threading:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/multi_threading.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:multi_threading:part0]


.. note::

   ``ov::hint::scheduling_core_type`` and ``ov::hint::enable_hyper_threading`` only support Intel® x86-64 CPU on Linux and Windows in current release.

In some use cases, OpenVINO Runtime will enable CPU threads pinning by default for better performance. User can also turn it on or off using property ``ov::hint::enable_cpu_pinning``. Disable threads pinning might be beneficial in complex applications with several workloads executed in parallel.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/multi_threading.py
         :language: python
         :fragment: [ov:intel_cpu:multi_threading:part1]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/multi_threading.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:multi_threading:part1]


For details on multi-stream execution check the
:doc:`optimization guide <../../optimize-inference/optimizing-throughput/advanced_throughput_options>`.