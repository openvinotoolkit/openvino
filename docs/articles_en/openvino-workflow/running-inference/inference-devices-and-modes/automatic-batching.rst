Automatic Batching
==================


.. meta::
   :description: The Automatic Batching Execution mode in OpenVINO Runtime
                 performs automatic batching to improve device utilization
                 by grouping inference requests.


The Automatic Batching Execution mode (or Auto-batching for short) performs automatic batching on-the-fly to improve device utilization by grouping inference requests together, without programming effort from the user.
With Automatic Batching, gathering the input and scattering the output from the individual inference requests required for the batch happen transparently, without affecting the application code.

Auto Batching can be used :ref:`directly as a virtual device <auto-batching-as-device>` or as an :ref:`option for inference on CPU/GPU/NPU <auto-batching-as-option>` (by means of configuration/hint). These 2 ways are provided for the user to enable the BATCH devices **explicitly** or **implicitly**, with the underlying logic remaining the same. An example of the difference is that the CPU device doesn’t support implicitly to enable BATCH device, commands such as ``./benchmark_app -m <model> -d CPU -hint tput`` will not apply BATCH device **implicitly**, but ``./benchmark_app -m <model> -d "BATCH:CPU(16)`` can **explicitly** load BATCH device.

Auto-batching primarily targets the existing code written for inferencing many requests, each instance with the batch size 1. To get corresponding performance improvements, the application **must be running multiple inference requests simultaneously**.
Auto-batching can also be used via a particular *virtual* device.

This article provides a preview of the Automatic Batching function, including how it works, its configurations, and testing performance.

How Automatic Batching Works
############################

.. tab-set::

   .. tab-item:: Enabling Automatic Batching
      :sync: enabling-automatic-batching

      Batching is a straightforward way of leveraging the compute power of GPU and saving on communication overheads. Automatic Batching is "implicitly" triggered on the GPU when ``ov::hint::PerformanceMode::THROUGHPUT`` is specified for the ``ov::hint::performance_mode`` property for the ``compile_model`` or ``set_property`` calls.

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
               :language: Python
               :fragment: [compile_model]

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
               :language: cpp
               :fragment: [compile_model]

      To enable Auto-batching in the legacy apps not akin to the notion of performance hints, you need to use the **explicit** device notion, such as ``BATCH:GPU``.

   .. tab-item:: Disabling Automatic Batching
      :sync: disabling-automatic-batching

      Auto-Batching can be disabled (for example, for the GPU device) to prevent being triggered by ``ov::hint::PerformanceMode::THROUGHPUT``. To do that, set ``ov::hint::allow_auto_batching`` to **false** in addition to the ``ov::hint::performance_mode``, as shown below:

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
               :language: Python
               :fragment: [compile_model_no_auto_batching]

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
               :language: cpp
               :fragment: [compile_model_no_auto_batching]


Configuring Automatic Batching
++++++++++++++++++++++++++++++

Following the OpenVINO naming convention, the *batching* device is assigned the label of *BATCH*. The configuration options are as follows:

+----------------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter name             | Parameter description                                                                                | Examples                                                                                                                                                                                                                                         |
+============================+======================================================================================================+==================================================================================================================================================================================================================================================+
| ``AUTO_BATCH_DEVICE``      | The name of the device to apply Automatic batching,  with the optional batch size value in brackets. | ``BATCH:GPU`` triggers the automatic batch size selection. ``BATCH:GPU(4)`` directly specifies the batch size.                                                                                                                                   |
+----------------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ov::auto_batch_timeout`` | The timeout value, in ms. (1000 by default)                                                          | You can reduce the timeout value to avoid performance penalty when the data arrives too unevenly. For example, set it to "100", or the contrary, i.e., make it large enough to accommodate input preparation (e.g. when it is a serial process). |
+----------------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Automatic Batch Size Selection
++++++++++++++++++++++++++++++

In both the THROUGHPUT hint and the explicit BATCH device cases, the optimal batch size is selected automatically, as the implementation queries the ``ov::optimal_batch_size`` property from the device and passes the model graph as the parameter. The actual value depends on the model and device specifics, for example, the on-device memory for dGPUs.
The support for Auto-batching is not limited to GPU. However, if a device does not support ``ov::optimal_batch_size`` yet, to work with Auto-batching, an explicit batch size must be specified, e.g., ``BATCH:<device>(16)``.

This "automatic batch size selection" works on the presumption that the application queries ``ov::optimal_number_of_infer_requests`` to create the requests of the returned number and run them simultaneously:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: Python
         :fragment: [query_optimal_num_requests]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [query_optimal_num_requests]


.. _limiting-batch-size:

Optimizing Performance by Limiting Batch Size
---------------------------------------------

If not enough inputs were collected, the ``timeout`` value makes the transparent execution fall back to the execution of individual requests. This value can be configured via the ``AUTO_BATCH_TIMEOUT`` property.
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid this, when your parallel slack is bounded, provide OpenVINO with an additional hint.

For example, when the application processes only 4 video streams, there is no need to use a batch larger than 4. The most future-proof way to communicate the limitations on the parallelism is to equip the performance hint with the optional ``ov::hint::num_requests`` configuration key set to 4. This will limit the batch size for the GPU and the number of inference streams for the CPU, hence each device uses ``ov::hint::num_requests`` while converting the hint to the actual device configuration options:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: Python
         :fragment: [hint_num_requests]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [hint_num_requests]


For the *explicit* usage, you can limit the batch size by using ``BATCH:GPU(4)``, where 4 is the number of requests running in parallel.


.. _auto-batching-as-device:

Automatic Batching as an explicit device
++++++++++++++++++++++++++++++++++++++++

The below examples show how AUTO Batching can be used in the form of device that the user can apply to perform inference directly:

.. code-block:: sh

   ./benchmark_app -m <model> -d "BATCH:GPU"
   ./benchmark_app -m <model> -d "BATCH:GPU(16)"
   ./benchmark_app -m <model> -d "BATCH:CPU(16)"


* ``BATCH`` -- load BATCH device explicitly,
* ``:GPU(16)`` -- BATCH devices configuration, which tell BATCH device to apply GPU device with batch size = 16.

.. _auto-batching-as-option:

Automatic Batching as underlying device configured to other devices
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In the following example, BATCH device will be configured to another device in case of ``tput/ctput mode``.

.. code-block:: sh

   ./benchmark_app -m <model> -d GPU -hint tput
   ./benchmark_app -m <model> -d AUTO -hint tput
   ./benchmark_app -m <model> -d AUTO -hint ctput
   ./benchmark_app -m <model> -d AUTO:GPU -hint ctput

.. note::
   If you run ``./benchmark_app``, do not set ``batch_size`` by ``-b <batch_size>``, otherwise AUTO mode will not be applied.

Other Performance Considerations
################################

To achieve the best performance with Automatic Batching, the application should:

- Operate inference requests of the number that represents the multiple of the batch size. In the example from :ref:`Optimizing Performance by Limiting Batch Size section <limiting-batch-size>` -- for batch size 4, the application should operate 4, 8, 12, 16, etc. requests.
- Use the requests that are grouped by the batch size together. For example, the first 4 requests are inferred, while the second group of the requests is being populated. Essentially, Automatic Batching shifts the asynchronicity from the individual requests to the groups of requests that constitute the batches.
- Balance the ``timeout`` value vs. the batch size. For example, in many cases, having a smaller ``timeout`` value/batch size may yield better performance than having a larger batch size with a ``timeout`` value that is not large enough to accommodate the full number of the required requests.
- When Automatic Batching is enabled, the ``timeout`` property of ``ov::CompiledModel`` can be changed anytime, even after the loading/compilation of the model. For example, setting the value to 0 disables Auto-batching effectively, as the collection of requests would be omitted.
- Carefully apply Auto-batching to the pipelines. For example, in the conventional "video-sources -> detection -> classification" flow, it is most beneficial to do Auto-batching over the inputs to the detection stage. The resulting number of detections is usually fluent, which makes Auto-batching less applicable for the classification stage.

Limitations
+++++++++++

The following are limitations of the current AUTO Batching implementations:

- The dynamic model is not supported by ``BATCH`` device.
- ``BATCH`` device can only support ``tput/ctput mode``. The ``latency/none mode`` is not supported.
- Supported are only models with ``batch dimension = 1``.
- The input/output tensor should come from ``inferRequest``, otherwise the user-created tensor will trigger a memory copying.
- The ``OPTIMAL_BATCH_SIZE`` should be greater than ``2``. In case it's not, user needs to specify a batch size which depends on model and device (CPU does not support this property).
- ``BATCH`` device supports GPU by default, while CPU will not trigger ``auto_batch`` in ``tput`` mode.
- ``AUTO_BATCH`` will bring much more compilation latency.
- Although it is less critical for the throughput-oriented scenarios, the load time with Auto-batching increases by almost double.
- Certain networks are not safely reshapable by the "batching" dimension (specified as ``N`` in the layout terms). Besides, if the batching dimension is not zeroth, Auto-batching will not be triggered "implicitly" by the throughput hint.
-  The "explicit" notion, for example, ``BATCH:GPU``, using the relaxed dimensions tracking, often makes Auto-batching possible. For example, this method unlocks most **detection networks**.
- When *forcing* Auto-batching via the "explicit" device notion, make sure that you validate the results for correctness.
- Performance improvements happen at the cost of the growth of memory footprint. However, Auto-batching queries the available memory (especially for dGPU) and limits the selected batch size accordingly.

.. note::
   ``BATCH`` device supports GPU by default, but GPU still may not trigger ``auto_batch`` in ``tput`` mode if model or GPU memory size are not allowed. Which means it is required to check ``supported_properties`` of GPU ``tput`` mode ``compiled_model`` before doing any actions (set/get) with ``ov::auto_batch_timeout`` property.<br/>
   To make sure ``BATCH`` device supports GPU by default, ``ov::model`` is required for ``core.compile_model``. A string of model file path to ``core.compile_model`` will be passed to GPU Plugin directly due to performance consideration and without involving ``BATCH``.


Testing Performance with Benchmark_app
######################################

Using the :doc:`benchmark_app sample <../../../get-started/learn-openvino/openvino-samples/benchmark-tool>` is the best way to evaluate the performance of Automatic Batching:

- The most straightforward way is using the performance hints:

  - benchmark_app **-hint tput** -d GPU -m 'path to your favorite model'
- You can also use the "explicit" device notion to override the strict rules of the implicit reshaping by the batch dimension:

  - benchmark_app **-hint none -d BATCH:GPU** -m 'path to your favorite model'
- or override the automatically deduced batch size as well:

  - $benchmark_app -hint none -d **BATCH:GPU(16)** -m 'path to your favorite model'
  - This example also applies to CPU or any other device that generally supports batch execution.
  - Keep in mind that some shell versions (e.g. ``bash``) may require adding quotes around complex device names, i.e. ``-d "BATCH:GPU(16)"`` in this example.


Note that Benchmark_app performs a warm-up run of a *single* request. As Auto-Batching requires significantly more requests to execute in batch, this warm-up run hits the default timeout value (1000 ms), as reported in the following example:

.. code-block:: sh

   [ INFO ] First inference took 1000.18ms

This value also exposed as the final execution statistics on the ``benchmark_app`` exit:

.. code-block:: sh

   [ INFO ] Latency:
   [ INFO ]  Max:      1000.18 ms

This is NOT the actual latency of the batched execution, so you are recommended to refer to other metrics in the same log, for example, "Median" or "Average" execution.

Additional Resources
####################

* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`



