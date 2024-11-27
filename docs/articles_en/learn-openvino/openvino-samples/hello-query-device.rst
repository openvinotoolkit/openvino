Hello Query Device Sample
=========================


.. meta::
   :description: Learn how to show metrics and default
                 configuration values of inference devices using Query
                 Device API feature (Python, C++).


This sample demonstrates how to show OpenVINO™ Runtime devices and prints their
metrics and default configuration values using :doc:`Query Device API feature <../../openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties>`.
To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
section in "Get Started with Samples" guide.

How It Works
####################

The sample queries all available OpenVINO™ Runtime devices and prints their
supported metrics and plugin configuration parameters.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. scrollbox::

         .. doxygensnippet:: samples/python/hello_query_device/hello_query_device.py
            :language: python


   .. tab-item:: C++
      :sync: cpp

      .. scrollbox::

         .. doxygensnippet:: samples/cpp/hello_query_device/main.cpp
            :language: cpp


Running
####################

The sample has no command-line parameters. To see the report, run the following command:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python hello_query_device.py

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         hello_query_device



Sample Output
####################

The application prints all available devices with their supported metrics and
default values for configuration parameters.
For example:


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         [ INFO ] Available devices:
         [ INFO ] CPU :
         [ INFO ]        SUPPORTED_PROPERTIES:
         [ INFO ]                AVAILABLE_DEVICES:
         [ INFO ]                FULL_DEVICE_NAME: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
         [ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, FP16, INT8, BIN
         [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 1, 1
         [ INFO ]                RANGE_FOR_STREAMS: 1, 8
         [ INFO ]                IMPORT_EXPORT_SUPPORT: True
         [ INFO ]                CACHE_DIR:
         [ INFO ]                ENABLE_CPU_PINNING: NO
         [ INFO ]                INFERENCE_NUM_THREADS: 0
         [ INFO ]                NUM_STREAMS: 1
         [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT:
         [ INFO ]                INFERENCE_PRECISION_HINT: f32
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
         [ INFO ]                PERFORMANCE_HINT:
         [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS: 0
         [ INFO ]                PERF_COUNT: NO


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>
         [ INFO ]
         [ INFO ] Available devices:
         [ INFO ] CPU
         [ INFO ]        SUPPORTED_PROPERTIES:
         [ INFO ]                AVAILABLE_DEVICES : [  ]
         [ INFO ]                FULL_DEVICE_NAME : Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
         [ INFO ]                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
         [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
         [ INFO ]                RANGE_FOR_STREAMS : { 1, 8 }
         [ INFO ]                IMPORT_EXPORT_SUPPORT : true
         [ INFO ]                CACHE_DIR : ""
         [ INFO ]                ENABLE_CPU_PINNING : NO
         [ INFO ]                INFERENCE_NUM_THREADS : 0
         [ INFO ]                NUM_STREAMS : 1
         [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT : ""
         [ INFO ]                INFERENCE_PRECISION_HINT : f32
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
         [ INFO ]                PERFORMANCE_HINT : ""
         [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS : 0
         [ INFO ]                PERF_COUNT : NO


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
- :doc:`Get Started with Samples <get-started-demos>`
- :doc:`Using OpenVINO™ Toolkit Samples <../openvino-samples>`
- `Hello Query Device Python Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_query_device/README.md>`__
- `Hello Query Device C++ Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_query_device/README.md>`__
