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

.. note::

   The :doc:`Automatic Device Selection <../inference-devices-and-modes/auto-device-selection>` allows configuration of all devices at once.

