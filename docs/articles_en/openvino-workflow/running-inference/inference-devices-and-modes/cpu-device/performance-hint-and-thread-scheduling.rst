Performance Hints and Thread Scheduling
===============================================================================================

.. meta::
   :description: Thread Scheduling of the CPU plugin in OpenVINO™ Runtime
                 detects CPU architecture and sets low-level properties based
                 on performance hints automatically.

To simplify the configuration of hardware devices, it is recommended to use the
``ov::hint::PerformanceMode::LATENCY`` and ``ov::hint::PerformanceMode::THROUGHPUT``
:doc:`high-level performance hints <../../optimize-inference/high-level-performance-hints>`.
Both performance hints ensure optimal portability
and scalability of applications across various platforms and models.

- ``ov::inference_num_threads`` limits the number of logical processors used for CPU
  inference. The multi-threading scheduler will use only the total number of logical
  processors on platform when the specified number exceeds it.
- ``ov::num_streams`` limits the number of infer requests that can be run in parallel.
  If the specified number is greater than the number of inference threads,
  multi-threading scheduler uses only the number of inference threads to ensure that there
  is at least one thread per stream.
- ``ov::hint::scheduling_core_type`` specifies the type of CPU cores for inference when
  run on a hybrid platform that includes both Performance-cores (P-cores)
  and Efficient-cores (E-cores). If the platform has one type of CPU core, the
  property has no effect, and this unique core type is used for inference.
- ``ov::hint::enable_hyper_threading`` limits the use of one or two logical processors per CPU
  core when CPU hyper-threading is enabled on the platform.
  If there is only one logical processor per CPU core, such as Efficient-core, the
  property has no effect, and all logical processors are used for inference.
- ``ov::hint::enable_cpu_pinning`` enables CPU pinning during inference.
  If the inference scenario does not support it, this property will be disabled during
  model compilation

For additional details on the above configurations, refer to
:ref:`Multi-stream Execution <multi_stream_execution>`.

Latency Hint
###############################################################################################

In this scenario, the default setting of ``ov::hint::scheduling_core_type`` is determined by
the model precision and the ratio of P-cores to E-cores.

.. note::

    Performance-cores (P-cores) and Efficient-cores (E-cores) are available starting
    with the 12th Gen Intel® Core™ processors.

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

The default settings for low-level performance properties on Windows and Linux are as follows:

+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
| Property                             | Windows                                                            | Linux                                                                     |
+======================================+====================================================================+===========================================================================+
| ``ov::num_streams``                  | 1                                                                  | 1                                                                         |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``ov::inference_num_threads``        | is equal to the number of P-cores or P-cores+E-cores on one socket | is equal to the number of P-cores or P-cores+E-cores on one socket        |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``ov::hint::scheduling_core_type``   | :ref:`Core Type Table of Latency Hint <core_type_latency>`         | :ref:`Core Type Table of Latency Hint <core_type_latency>`                |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
|                                      | ``ov::hint::scheduling_core_type`` *may be adjusted for a particular model inferred on a*                                                      |
|                                      | *specific platform based on internal heuristics to guarantee optimal performance.*                                                             |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``ov::hint::enable_hyper_threading`` | No                                                                 | No                                                                        |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
|                                      | *If* ``ov::hint::enable_hyper_threading`` *is enabled, two logical processors share the hardware resources of one CPU core. The use of both*   |
|                                      | *logical processors in one stream for a single infer request is not likely to happen, so the property is set to* ``No`` *by default.*          |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``ov::hint::enable_cpu_pinning``     | No / Not Supported                                                 | Yes except using P-cores and E-cores together                             |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+
|                                      | ``ov::hint::enable_cpu_pinning`` *is disabled by default on Windows and macOS, but enabled on Linux. The default settings are aligned with*    |
|                                      | *typical workloads running in the corresponding environments to guarantee better out-of-the-box (OOB) performance.*                            |
+--------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------------+

.. note::

   - Both P-cores and E-cores are used with latency hint on Intel® Core™ Ultra Processors
     on Windows, except the case of large language models.
   - By default, a single socket is used for inference when latency hint is enabled.
     Although such behavior gives best performance for most of the models, there might be
     corner cases, which require manual tuning of ``ov::num_streams`` and
     ``ov::hint::enable_hyper_threading parameters``.
   - Starting from 5th Gen Intel Xeon Processors, new microarchitecture introduced sub-NUMA
     clusters feature. A sub-NUMA cluster (SNC) can create two or more localization
     domains (numa nodes) within a socket by BIOS configuration. Refer to
     `Sub-NUMA Clustering <https://www.intel.com/content/www/us/en/developer/articles/technical/xeon-processor-scalable-family-technical-overview.html>`__
     for more details.


Throughput Hint
###############################################################################################

In this scenario, thread scheduling first evaluates the memory pressure of the model being
inferred on the current platform, and determines the number of threads per stream,
as shown below:

+-----------------+-----------------------+
| Memory Pressure | Threads per Stream    |
+=================+=======================+
| low             | 1 P-core or 2 E-cores |
+-----------------+-----------------------+
| medium          | 2                     |
+-----------------+-----------------------+
| high            | 3 or 4 or 5           |
+-----------------+-----------------------+

Then, the value of ``ov::num_streams`` is calculated by dividing ``ov::inference_num_threads``
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
###############################################################################################

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

In some use cases, OpenVINO Runtime will enable CPU thread pinning by default for better
performance. You can also turn this feature on or off, using the
``ov::hint::enable_cpu_pinning`` property. Disabling thread pinning may be beneficial
in complex applications where several workloads are executed in parallel.

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


For details on multi-stream execution, check the
:doc:`optimization guide <../../optimize-inference/optimizing-throughput/advanced_throughput_options>`.

.. _Composability_of_different_threading_runtimes:

Composability of different threading runtimes
###############################################################################################

OpenVINO is built with the `oneTBB <https://github.com/oneapi-src/oneTBB/>`__ threading
library by default. oneTBB has the `worker_wait` feature, similar to
`OpenMP <https://www.openmp.org/>`__
`busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__, which makes
inference threads in OpenVINO wait actively for a while after a task is done. The intention is
to avoid CPU inactivity in between inference tasks.

When inference tasks along with other sequential application logic are run in OpenVINO
on the CPU, using different threading runtimes (for example, oneTBB is used for inference
in OpenVINO, and OpenMP is used for other application logic) will cause all operations to
occupy CPU cores for additional time after completion, leading to overhead.

**Recommended solutions:**

- Use oneTBB for all computations in the pipeline for the best effect.
- Rebuild OpenVINO with OpenMP if other application logic uses it.
- Set the
  `OMP_WAIT_POLICY <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html>`__
  environment variable
  to `PASSIVE` to disable
  `busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__
  when using OpenMP.
- Limit the number of threads for OpenVINO and other parts and let OS do the scheduling.
