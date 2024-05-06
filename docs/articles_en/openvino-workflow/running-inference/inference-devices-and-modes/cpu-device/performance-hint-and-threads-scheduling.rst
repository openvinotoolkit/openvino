.. {#openvino_docs_OV_UG_supported_plugins_CPU_Hints_Threading}

Performance Hints and Threads Scheduling 
=======================================

.. meta::
   :description: The Threads Scheduling of CPU plugin in OpenVINO™ Runtime
                 detects CPU architecture and sets low-level properties based
                 on performance hints automatically.

While all supported devices in OpenVINO offer low-level performance settings, it is advisable not to widely use these settings unless targeting specific platforms and models. The recommended approach is configuring performance in OpenVINO Runtime using high-level performance hints property ``ov::hint::performance_mode``. Performance hints ensure optimal portability and scalability of the applications across various platforms and models.

To simplify the configuration of hardware devices, OpenVINO offers two performance hints: the latency hint ``ov::hint::PerformanceMode::LATENCY`` and the throughput hint ``ov::hint::PerformanceMode::THROUGHPUT``.

Using the CPU as an example, employing these two performance hints automatically configures the following low-level performance properties on the threads scheduling side. These configuration details may vary between releases to ensure optimal performance on various platforms and a wide set of models. Overall performance is usually measured with a GEOMean calculation of performance differences across hundreds of models of various sizes and precisions.

- ``ov::num_streams``
- ``ov::inference_num_threads``
- ``ov::hint::scheduling_core_type``
- ``ov::hint::enable_hyper_threading``
- ``ov::hint::enable_cpu_pinning``

For additional details on the above configurations, refer to:

- `Multi-stream Execution <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#multi-stream-execution>`__
- `Multi-Threading Optimization <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#multi-threading-optimization>`__

Latency Hint on Hybrid Core Systems
###################################

In this scenario, the default setting of ``ov::hint::scheduling_core_type`` is determined by the model precision and the ratio of P-cores and E-cores.

.. note::

    P-cores is short for Performance-cores and E-cores is for Efficient-cores. These are available in Intel 12th, 13th, and 14th generations of Intel® Core™ processor (code name: Alder Lake, Raptor Lake, Raptor Lake refresh), and Intel® Core™ Ultra Processors (code name Meteor Lake). 

+----------------------------+---------------------+---------------------+
|                            | INT8 model          | FP32 model          |
+============================+=====================+=====================+
| E-cores / P-cores < 2      | P-cores             | P-cores             |
+----------------------------+---------------------+---------------------+
| 2 <= E-cores / P-cores < 4 | P-cores             | P-cores and E-cores |
+----------------------------+---------------------+---------------------+
| 4 <= E-cores / P-cores     | P-cores and E-cores | P-cores and E-cores |
+----------------------------+---------------------+---------------------+

Then the default settings of low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+-----------------------------------+-----------------------------------------------+
| Property                             | Windows                           | Linux                                         |
+======================================+===================================+===============================================+
| ``ov::num_streams``                  | 1                                 | 1                                             |
+--------------------------------------+-----------------------------------+-----------------------------------------------+
| ``ov::inference_num_threads``        | Dependent on scheduling_core_type | Dependent on scheduling_core_type             |
+--------------------------------------+-----------------------------------+-----------------------------------------------+
| ``ov::hint::scheduling_core_type``   | Previous table                    | Previous table                                |
+--------------------------------------+-----------------------------------+-----------------------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                                | No                                            |
+--------------------------------------+-----------------------------------+-----------------------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No                                | Yes except using P-cores and E-cores together |
+--------------------------------------+-----------------------------------+-----------------------------------------------+

.. note::

    Both P-cores and E-cores are used for the Latency Hint on Intel® Core™ Ultra Processors on Windows, except in the case of large language models.

Throughput Hint on Hybrid Core Systems
######################################

In this scenario, thread scheduling first evaluates the memory pressure of the model being inferred on the current platform, and determines the number of threads per stream, as shown below.

+-----------------+-----------------------+
| Memory Pressure | Threads per stream    |
+=================+=======================+
| least           | 1 P-core or 2 E-cores |
+-----------------+-----------------------+
| less            | 2                     |
+-----------------+-----------------------+
| normal          | 3 or 4 or 5           |
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
| ``ov::hint::enable_hyper_threading`` | Yes                           | Yes                           |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No                            | Yes                           |
+--------------------------------------+-------------------------------+-------------------------------+

Latency Hint on Non-Hybrid Core Systems or Single-Socket XEON Platforms
#######################################################################

In this case, the logic is identical to the case where ``ov::hint::scheduling_core_type`` is P-cores in the `Latency Hint on Hybrid Core Systems <#latency-hint-on-hybrid-core-systems>`__.

Throughput Hint on Non-Hybrid Core Systems or Single-Socket XEON Platforms
##########################################################################

In this case, the logic is identical to the case where ``ov::hint::scheduling_core_type`` is P-cores in the `Throughput Hint on Hybrid Core Systems <#throughput-hint-on-hybrid-core-systems>`__.

Latency Hint on Dual-Socket XEON Platforms
##########################################

In this scenario, thread scheduling only creates one stream, currently pinned to one socket. The default settings of low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+-------------------------------+-------------------------------+
| Property                             | Windows                       | Linux                         |
+======================================+===============================+===============================+
| ``ov::num_streams``                  | 1                             | 1                             |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::inference_num_threads``        | Number of cores on one socket | Number of cores on one socket |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::scheduling_core_type``   | P-cores or E-cores            | P-cores or E-cores            |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                            | No                            |
+--------------------------------------+-------------------------------+-------------------------------+
| ``ov::hint::enable_cpu_pinning``     | Not Supported                 | Yes                           |
+--------------------------------------+-------------------------------+-------------------------------+

Throughput Hint on Dual-Socket XEON Platforms
##############################################

In this scenario, thread scheduling first evaluates the memory pressure of the model being inferred on the current platform, and determines the number of threads per stream, as shown below.

+-----------------+--------------------+
| Memory Pressure | Threads per stream |
+=================+====================+
| least           | 1                  |
+-----------------+--------------------+
| less            | 2                  |
+-----------------+--------------------+
| normal          | 3 or 4 or 5        |
+-----------------+--------------------+

Then the value of ``ov::num_streams`` is calculated as ``ov::inference_num_threads`` divided by the number of threads per stream. The default settings of low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+---------------------------------+---------------------------------+
| Property                             | Windows                         | Linux                           |
+======================================+=================================+=================================+
| ``ov::num_streams``                  | Calculate as above              | Calculate as above              |
+--------------------------------------+---------------------------------+---------------------------------+
| ``ov::inference_num_threads``        | Number of cores on dual sockets | Number of cores on dual sockets |
+--------------------------------------+---------------------------------+---------------------------------+
| ``ov::hint::scheduling_core_type``   | P-cores or E-cores              | P-cores or E-cores              |
+--------------------------------------+---------------------------------+---------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                              | No                              |
+--------------------------------------+---------------------------------+---------------------------------+
| ``ov::hint::enable_cpu_pinning``     | Not Supported                   | Yes                             |
+--------------------------------------+---------------------------------+---------------------------------+
