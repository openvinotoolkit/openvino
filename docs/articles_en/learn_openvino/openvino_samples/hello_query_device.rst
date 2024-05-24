.. {#openvino_sample_hello_query_device}

Hello Query Device Sample
=========================


.. meta::
   :description: Learn how to show metrics and default
                 configuration values of inference devices using Query
                 Device API feature (Python, C++).


This sample demonstrates how to show OpenVINO™ Runtime devices and prints their
metrics and default configuration values using :doc:`Query Device API feature <openvino_docs_OV_UG_query_api>`.
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
         [ INFO ]        SUPPORTED_METRICS:
         [ INFO ]                AVAILABLE_DEVICES:
         [ INFO ]                FULL_DEVICE_NAME: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
         [ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, FP16, INT8, BIN
         [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 1, 1
         [ INFO ]                RANGE_FOR_STREAMS: 1, 8
         [ INFO ]                IMPORT_EXPORT_SUPPORT: True
         [ INFO ]
         [ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
         [ INFO ]                CACHE_DIR:
         [ INFO ]                CPU_BIND_THREAD: NO
         [ INFO ]                CPU_THREADS_NUM: 0
         [ INFO ]                CPU_THROUGHPUT_STREAMS: 1
         [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT:
         [ INFO ]                ENFORCE_BF16: NO
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
         [ INFO ]                PERFORMANCE_HINT:
         [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS: 0
         [ INFO ]                PERF_COUNT: NO
         [ INFO ]
         [ INFO ] GNA :
         [ INFO ]        SUPPORTED_METRICS:
         [ INFO ]                AVAILABLE_DEVICES: GNA_SW
         [ INFO ]                OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
         [ INFO ]                FULL_DEVICE_NAME: GNA_SW
         [ INFO ]                GNA_LIBRARY_FULL_VERSION: 3.0.0.1455
         [ INFO ]                IMPORT_EXPORT_SUPPORT: True
         [ INFO ]
         [ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
         [ INFO ]                GNA_COMPACT_MODE: YES
         [ INFO ]                GNA_COMPILE_TARGET:
         [ INFO ]                GNA_DEVICE_MODE: GNA_SW_EXACT
         [ INFO ]                GNA_EXEC_TARGET:
         [ INFO ]                GNA_FIRMWARE_MODEL_IMAGE:
         [ INFO ]                GNA_FIRMWARE_MODEL_IMAGE_GENERATION:
         [ INFO ]                GNA_LIB_N_THREADS: 1
         [ INFO ]                GNA_PRECISION: I16
         [ INFO ]                GNA_PWL_MAX_ERROR_PERCENT: 1.000000
         [ INFO ]                GNA_PWL_UNIFORM_DESIGN: NO
         [ INFO ]                GNA_SCALE_FACTOR: 1.000000
         [ INFO ]                GNA_SCALE_FACTOR_0: 1.000000
         [ INFO ]                LOG_LEVEL: LOG_NONE
         [ INFO ]                PERF_COUNT: NO
         [ INFO ]                SINGLE_THREAD: YES

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>
         [ INFO ]
         [ INFO ] Available devices:
         [ INFO ] CPU
         [ INFO ]        SUPPORTED_METRICS:
         [ INFO ]                AVAILABLE_DEVICES : [  ]
         [ INFO ]                FULL_DEVICE_NAME : Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
         [ INFO ]                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
         [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
         [ INFO ]                RANGE_FOR_STREAMS : { 1, 8 }
         [ INFO ]                IMPORT_EXPORT_SUPPORT : true
         [ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
         [ INFO ]                CACHE_DIR : ""
         [ INFO ]                CPU_BIND_THREAD : NO
         [ INFO ]                CPU_THREADS_NUM : 0
         [ INFO ]                CPU_THROUGHPUT_STREAMS : 1
         [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT : ""
         [ INFO ]                ENFORCE_BF16 : NO
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
         [ INFO ]                PERFORMANCE_HINT : ""
         [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS : 0
         [ INFO ]                PERF_COUNT : NO
         [ INFO ]
         [ INFO ] GNA
         [ INFO ]        SUPPORTED_METRICS:
         [ INFO ]                AVAILABLE_DEVICES : [ GNA_SW_EXACT ]
         [ INFO ]                OPTIMAL_NUMBER_OF_INFER_REQUESTS : 1
         [ INFO ]                FULL_DEVICE_NAME : GNA_SW_EXACT
         [ INFO ]                GNA_LIBRARY_FULL_VERSION : 3.0.0.1455
         [ INFO ]                IMPORT_EXPORT_SUPPORT : true
         [ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
         [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
         [ INFO ]                GNA_COMPACT_MODE : YES
         [ INFO ]                GNA_COMPILE_TARGET : ""
         [ INFO ]                GNA_DEVICE_MODE : GNA_SW_EXACT
         [ INFO ]                GNA_EXEC_TARGET : ""
         [ INFO ]                GNA_FIRMWARE_MODEL_IMAGE : ""
         [ INFO ]                GNA_FIRMWARE_MODEL_IMAGE_GENERATION : ""
         [ INFO ]                GNA_LIB_N_THREADS : 1
         [ INFO ]                GNA_PRECISION : I16
         [ INFO ]                GNA_PWL_MAX_ERROR_PERCENT : 1.000000
         [ INFO ]                GNA_PWL_UNIFORM_DESIGN : NO
         [ INFO ]                GNA_SCALE_FACTOR : 1.000000
         [ INFO ]                GNA_SCALE_FACTOR_0 : 1.000000
         [ INFO ]                LOG_LEVEL : LOG_NONE
         [ INFO ]                PERF_COUNT : NO
         [ INFO ]                SINGLE_THREAD : YES

Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- `Hello Query Device Python Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_query_device/README.md>`__
- `Hello Query Device C++ Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_query_device/README.md>`__
