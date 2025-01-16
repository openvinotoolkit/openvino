Optimizing for Throughput
=========================


.. meta::
   :description: Throughput-oriented approaches in OpenVINO involve
                 execution of a large number of inference requests
                 simultaneously which improves the device utilization.


.. toctree::
   :maxdepth: 1
   :hidden:

   Advanced Throughput Options <optimizing-throughput/advanced_throughput_options>


As described in the section on the :doc:`latency-specific optimizations <optimizing-latency>`, one of the possible use cases is delivering every single request with minimal delay.
Throughput, on the other hand, is about inference scenarios in which potentially **large numbers of inference requests are served simultaneously to improve resource use**.

The associated increase in latency is not linearly dependent on the number of requests executed in parallel.
A trade-off between overall throughput and serial performance of individual requests can be achieved with the right performance configuration of OpenVINO.

Basic and Advanced Ways of Leveraging Throughput
################################################

There are two ways of leveraging throughput with individual devices:

* **Basic (high-level)** flow with :doc:`OpenVINO performance hints <high-level-performance-hints>` which is inherently **portable and future-proof**.
* **Advanced (low-level)** approach of explicit  **batching** and **streams**. For more details, see the :doc:`Advanced Throughput Options <optimizing-throughput/advanced_throughput_options>`

In both cases, the application should be designed to execute multiple inference requests in parallel, as described in the following section.

.. _throughput_app_design:

Throughput-Oriented Application Design
######################################

In general, most throughput-oriented inference applications should:

* Expose substantial amounts of *input* parallelism (e.g. process multiple video- or audio- sources, text documents, etc).
* Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel:

  * Setup the configuration for the *device* (for example, as parameters of the ``ov::Core::compile_model``) via either previously introduced :doc:`low-level explicit options <./optimizing-throughput/advanced_throughput_options>` or :doc:`OpenVINO performance hints <high-level-performance-hints>` (**preferable**):

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

  * Query the ``ov::optimal_number_of_infer_requests`` from the ``ov::CompiledModel`` (resulted from a compilation of the model for the device) to create the number of the requests required to saturate the device.

* Use the Async API with callbacks, to avoid any dependency on the completion order of the requests and possible device starvation, as explained in the :doc:`common-optimizations section <general-optimizations>`.

Multi-Device Execution
######################

OpenVINO offers the automatic, scalable :doc:`multi-device inference mode <../../../documentation/legacy-features/multi-device>`, which is a simple *application-transparent* way to improve throughput. There is no need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance inference requests between devices, etc. For the application using it, multi-device is like any other device, as it manages all processes internally.
Just like with other throughput-oriented scenarios, there are several major pre-requisites for optimal multi-device performance:

* Using the :ref:`Asynchronous API <async_api>` and :doc:`callbacks <../integrate-openvino-with-your-application/inference-request>` in particular.
* Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the "requests" (outermost) level to minimize the scheduling overhead.

Keep in mind that the resulting performance is usually a fraction of the "ideal" (plain sum) value, when the devices compete for certain resources such as the memory-bandwidth, which is shared between CPU and iGPU.

.. note::

   While the legacy approach of optimizing the parameters of each device separately works, the :doc:`Automatic Device Selection <../inference-devices-and-modes/auto-device-selection>` allow configuring all devices (that are part of the specific multi-device configuration) at once.

