Performance Hints and Thread Scheduling
========================================

.. meta::
   :description: Thread Scheduling of the CPU plugin in OpenVINO™ Runtime
                 detects CPU architecture and sets low-level properties based
                 on performance hints automatically.

To simplify the configuration of hardware devices, it is recommended to use the
ov::hint::PerformanceMode::LATENCY and ov::hint::PerformanceMode::THROUGHPUT
:doc:`high-level performance hints <../../optimize-inference/high-level-performance-hints>`. Both performance hints ensure optimal portability
and scalability of applications across various platforms and models.

- ``ov::inference_num_threads`` limits the number of logical processors used for CPU inference.
  If the number set by the user is greater than the number of logical processors on the platform,
  the multi-threading scheduler only uses the platform number for CPU inference.
- ``ov::num_streams`` limits the number of infer requests that can be run in parallel.
  If the number set by the user is greater than the number of inference threads, multi-threading
  scheduler only uses the number of inference threads to ensure that there is at least
  one thread per stream.
- ``ov::hint::scheduling_core_type`` specifies the type of CPU cores for CPU inference when
  the user runs inference on a hybird platform that includes both Performance-cores (P-cores)
  and Efficient-cores (E-cores). If the user platform only has one type of CPU core, this
  property has no effect, and CPU inference always uses this unique core type.
- ``ov::hint::enable_hyper_threading`` limits the use of one or two logical processors per CPU
  core when the platform has CPU hyperthreading enabled.
  If there is only one logical processor per CPU core, such as Efficient-cores, this
  property has no effect, and CPU inference uses all logical processors.
- ``ov::hint::enable_cpu_pinning`` enables CPU pinning during CPU inference.
  If the user enables this property but the inference scenario does not support it, this
  property will be disabled during model compilation.

For additional details on the above configurations, refer to
:ref:`Multi-stream Execution <multi_stream_execution>`.

Latency Hint
#####################

In this scenario, the default setting of ``ov::hint::scheduling_core_type`` is determined by
the model precision and the ratio of P-cores and E-cores.

.. note::

    P-cores is short for Performance-cores and E-cores stands for Efficient-cores. These
    types of cores are available starting with the 12th Gen Intel® Core™ processors.

.. _core_type_latency:

+----------------------------+---------------------+---------------------+
|                            | INT8 Model          | FP32 Model          |
+============================+=====================+=====================+
| E-cores / P-cores < 2      | P-cores             | P-cores             |
+----------------------------+---------------------+---------------------+
| 2 <= E-cores / P-cores < 4 | P-cores             | P-cores and E-cores |
+----------------------------+---------------------+---------------------+
| 4 <= E-cores / P-cores     | P-cores and E-cores | P-cores and E-cores |
+----------------------------+---------------------+---------------------+

.. note::

   Both P-cores and E-cores may be used for any configuration starting with 14th Gen Intel®
   Core™ processors on Windows.

Then the default settings for low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+
| Property                             | Windows                                                            | Linux                                                              |
+======================================+====================================================================+====================================================================+
| ``ov::num_streams``                  | 1                                                                  | 1                                                                  |
+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+
| ``ov::inference_num_threads``        | is equal to the number of P-cores or P-cores+E-cores on one socket | is equal to the number of P-cores or P-cores+E-cores on one socket |
+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+
| ``ov::hint::scheduling_core_type``   | :ref:`Core Type Table of Latency Hint <core_type_latency>`         | :ref:`Core Type Table of Latency Hint <core_type_latency>`         |
+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                                                                 | No                                                                 |
+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No / Not Supported                                                 | Yes except using P-cores and E-cores together                      |
+--------------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------+

.. note::

    - ``ov::hint::scheduling_core_type`` may be adjusted for a particular inferred model on a
      specific platform based on internal heuristics to guarantee optimal performance.
    - Both P-cores and E-cores are used for the Latency Hint on Intel® Core™ Ultra Processors
      on Windows, except in the case of large language models.
    - In case hyper-threading is enabled, two logical processors share the hardware resources
      of one CPU core. OpenVINO does not expect to use both logical processors in one stream
      for a single infer request. So ``ov::hint::enable_hyper_threading`` is set to
      ``No`` in this scenario.
    - ``ov::hint::enable_cpu_pinning`` is disabled by default on Windows and macOS, and
      enabled on Linux. Such default settings are aligned with typical workloads running
      in the corresponding environments to guarantee better out-of-the-box (OOB) performance.

.. note::

   Starting from 5th Gen Intel Xeon Processors, new microarchitecture enabled new sub-NUMA clusters
   feature. A sub-NUMA cluster (SNC) can create two or more localization domains (numa nodes)
   within a socket by BIOS configuration. 
   By default OpenVINO with latency hint uses single socket for inference. Although such
   behavior allows to achive best performance for most of the models, there might be corner
   cases which require manual tuning of ``ov::num_streams`` and ``ov::hint::enable_hyper_threading parameters``.
   Please find more detail about `Sub-NUMA Clustering <https://www.intel.com/content/www/us/en/developer/articles/technical/xeon-processor-scalable-family-technical-overview.html>`__ 

Throughput Hint
#####################

In this scenario, thread scheduling first evaluates the memory pressure of the model being
inferred on the current platform, and determines the number of threads per stream, as shown below.

+-----------------+-----------------------+
| Memory Pressure | Threads per Stream    |
+=================+=======================+
| low             | 1 P-core or 2 E-cores |
+-----------------+-----------------------+
| medium          | 2                     |
+-----------------+-----------------------+
| high            | 3 or 4 or 5           |
+-----------------+-----------------------+

Then the value of ``ov::num_streams`` is calculated by dividing ``ov::inference_num_threads``
by the number of threads per stream. The default settings for low-level performance
properties on Windows and Linux are as follows:

+--------------------------------------+-------------------------------+-------------------------------+
| Property                             | Windows                       | Linux                         |
+======================================+===============================+===============================+
| ``ov::num_streams``                  | Calculated as above           | Calculated as above           |
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

   By default, different core types are not mixed within a single stream in this scenario.
   The cores from different NUMA nodes are not mixed within a single stream.

Multi-Threading Optimization
############################

The following properties can be used to limit the available CPU resources for model inference.
If the platform or operating system supports this behavior, the OpenVINO Runtime will
perform multi-threading scheduling based on the limited available CPU.

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

   ``ov::hint::scheduling_core_type`` and ``ov::hint::enable_hyper_threading`` only support
   Intel® x86-64 CPU on Linux and Windows in the current release.

In some use cases, OpenVINO Runtime will enable CPU thread pinning by default for better performance.
Users can also turn this feature on or off using the property ``ov::hint::enable_cpu_pinning``.
Disabling thread pinning may be beneficial in complex applications where several workloads
are executed in parallel.

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

.. _Composability_of_different_threading_runtimes:

Composability of different threading runtimes
#############################################

OpenVINO is by default built with the `oneTBB <https://github.com/oneapi-src/oneTBB/>`__ threading library,
oneTBB has a feature `worker_wait`, similar to `OpenMP <https://www.openmp.org/>`__ `busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__, which makes OpenVINO inference
threads wait actively for a while after a task done. The intention is to avoid CPU inactivity in the
transition time between inference tasks. 

In the pipeline that runs OpenVINO inferences on the CPU along with other sequential application logic, using different threading runtimes (e.g., OpenVINO inferences use oneTBB,
while other application logic uses OpenMP) will cause both to occupy CPU cores for additional time after the task done, leading to overhead. 

Recommended solutions:

- The most effective way is to use oneTBB for all computations made in the pipeline.
- Rebuild OpenVINO with OpenMP if other application logic uses OpenMP.
- Limit the number of threads for OpenVINO and other parts and let OS do the scheduling.
- If other application logic uses OpenMP, set the environment variable `OMP_WAIT_POLICY <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html>`__ to `PASSIVE` to disable OpenMP `busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__.
