High-level Performance Hints
============================


.. meta::
   :description: OpenVINO Runtime offers two dedicated high-level performance
                 hints, namely throughput and latency, that help to configure
                 an inference device.


Even though all :doc:`supported devices <../../../about-openvino/compatibility-and-support/supported-devices>` in OpenVINO™ offer low-level performance settings, utilizing them is not recommended outside of very few cases.
The preferred way to configure performance in OpenVINO Runtime is using performance hints. This is a future-proof solution fully compatible with the :doc:`automatic device selection inference mode <../inference-devices-and-modes/auto-device-selection>` and designed with *portability* in mind.

The hints also set the direction of the configuration in the right order. Instead of mapping the application needs to the low-level performance settings, and keeping an associated application logic to configure each possible device separately, the hints express a target scenario with a single config key and let the *device* configure itself in response.

Previously, a certain level of automatic configuration was the result of the *default* values of the parameters. For example, the number of CPU streams was deduced from the number of CPU cores, when ``ov::streams::AUTO`` was set. However, the resulting number of streams did not account for actual compute requirements of the model to be inferred.
The hints, in contrast, respect the actual model, so the parameters for optimal throughput are calculated for each model individually (based on its compute versus memory bandwidth requirements and capabilities of the device).

Performance Hints: Latency and Throughput
#########################################

As discussed in the :doc:`Optimization Guide <../optimize-inference>` there are a few different metrics associated with inference speed. Latency and throughput are some of the most widely used metrics that measure the overall performance of an application.

Therefore, in order to ease the configuration of the device, OpenVINO offers two dedicated hints, namely ``ov::hint::PerformanceMode::LATENCY`` **(default)** and ``ov::hint::PerformanceMode::THROUGHPUT``.

For more information on conducting performance measurements with the ``benchmark_app``, refer to the last section in this document.

Keep in mind that a typical model may take significantly more time to load with the ``ov::hint::PerformanceMode::THROUGHPUT`` and consume much more memory, compared to the ``ov::hint::PerformanceMode::LATENCY``. Also, the `THROUGHPUT` and `LATENCY` hints only improve performance in an asynchronous inference pipeline. For information on asynchronous inference, see the :ref:`Prefer Async API <prefer-async-api>` section of this document.

Performance Hints: How It Works
###############################

Internally, every device "translates" the value of the hint to the actual performance settings.
For example, the ``ov::hint::PerformanceMode::THROUGHPUT`` selects the number of CPU or GPU streams.
Additionally, the optimal batch size is selected for the GPU and the :doc:`automatic batching <../inference-devices-and-modes/automatic-batching>` is applied whenever possible. To check whether the device supports it, refer to the :doc:`Supported devices <../../../about-openvino/compatibility-and-support/supported-devices>` article.

The resulting (device-specific) settings can be queried back from the instance of the ``ov:Compiled_Model``.
Be aware that the ``benchmark_app`` outputs the actual settings for the ``THROUGHPUT`` hint. See the example of the output below:

.. code-block:: sh

   $benchmark_app -hint tput -d CPU -m 'path to your favorite model'
   ...
   [Step 8/11] Setting optimal runtime parameters
   [ INFO ] Device: CPU
   [ INFO ]   { PERFORMANCE_HINT , THROUGHPUT }
   ...
   [ INFO ]   { OPTIMAL_NUMBER_OF_INFER_REQUESTS , 4 }
   [ INFO ]   { NUM_STREAMS , 4 }
   ...


Using the Performance Hints: Basic API
######################################

In the example code snippet below, ``ov::hint::PerformanceMode::THROUGHPUT`` is specified for the ``ov::hint::performance_mode`` property for ``compile_model``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: python
         :fragment: [compile_model]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [compile_model]


Additional (Optional) Hints from the App
########################################

For an application that processes 4 video streams, the most future-proof way to communicate the limitation of the parallel slack is to equip the performance hint with the optional ``ov::hint::num_requests`` configuration key set to 4.
As mentioned earlier, this will limit the batch size for the GPU and the number of inference streams for the CPU. Thus, each device uses the ``ov::hint::num_requests`` while converting the hint to the actual device configuration options:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: python
         :fragment: [hint_num_requests]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [hint_num_requests]


Optimal Number of Inference Requests
####################################

The hints are used on the presumption that the application queries ``ov::optimal_number_of_infer_requests`` to create and run the returned number of requests simultaneously:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: python
         :fragment: [query_optimal_num_requests]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [query_optimal_num_requests]


While an application is free to create more requests if needed (for example to support asynchronous inputs population) **it is very important to at least run the** ``ov::optimal_number_of_infer_requests`` **of the inference requests in parallel**. It is recommended for efficiency, or device utilization, reasons.

Keep in mind that ``ov::hint::PerformanceMode::LATENCY`` does not necessarily imply using single inference request. For example, multi-socket CPUs can deliver as many requests at the same minimal latency as the number of NUMA nodes in the system.
To make your application fully scalable, make sure to query the ``ov::optimal_number_of_infer_requests`` directly.

.. _prefer-async-api:

Prefer Async API
################

The API of the inference requests offers Sync and Async execution. The ``ov::InferRequest::infer()`` is inherently synchronous and simple to operate (as it serializes the execution flow in the current application thread). The Async "splits" the ``infer()`` into ``ov::InferRequest::start_async()`` and ``ov::InferRequest::wait()`` (or callbacks). For more information on synchronous and asynchronous modes, refer to the :doc:`OpenVINO Inference Request <../integrate-openvino-with-your-application/inference-request>`.

Although the synchronous API can be easier to start with, it is recommended to use the asynchronous (callbacks-based) API in production code. It is the most general and scalable way to implement the flow control for any possible number of requests. The ``THROUGHPUT`` and ``LATENCY`` performance hints automatically configure the Asynchronous pipeline to use the optimal number of processing streams and inference requests.

.. note::

   **Important:** Performance Hints only work when asynchronous execution mode is used. They do not affect the performance of a synchronous pipeline.

Combining the Hints and Individual Low-Level Settings
#####################################################

While sacrificing the portability to some extent, it is possible to combine the hints with individual device-specific settings.
For example, use ``ov::hint::PerformanceMode::THROUGHPUT`` to prepare a general configuration and override any of its specific values:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.py
         :language: python
         :fragment: [hint_plus_low_level]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [hint_plus_low_level]


Testing Performance of the Hints with the Benchmark_App
#######################################################

Using the :doc:`benchmark_app sample <../../../learn-openvino/openvino-samples/benchmark-tool>` is the best way to evaluate the functionality of the performance hints for a particular device:

* benchmark_app **-hint tput** -d 'device' -m 'path to your model'
* benchmark_app **-hint latency** -d 'device' -m 'path to your model'

Disabling the hints to emulate the pre-hints era (highly recommended before trying the individual low-level settings, such as the number of streams as below, threads, etc):

* benchmark_app **-hint none -nstreams 1**  -d 'device' -m 'path to your model'
