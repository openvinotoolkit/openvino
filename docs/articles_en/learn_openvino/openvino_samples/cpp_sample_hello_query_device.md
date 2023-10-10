# Hello Query Device C++ Sample {#openvino_inference_engine_samples_hello_query_device_README}

@sphinxdirective

.. meta::
   :description: Learn how to show metrics and default 
                 configuration values of inference devices using Query 
                 Device (C++) API feature.


This sample demonstrates how to execute an query OpenVINO™ Runtime devices, prints their metrics and default configuration values, using :doc:`Properties API <openvino_docs_OV_UG_query_api>`.

.. tab-set::

   .. tab-item:: Requirements 

      +----------------------------------------+----------------------------------------------------------------------------------------------+
      | Options                                | Values                                                                                       |
      +========================================+==============================================================================================+
      | Supported devices                      | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                         |
      +----------------------------------------+----------------------------------------------------------------------------------------------+
      | Other language realization             | :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README>` |
      +----------------------------------------+----------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API

      The following C++ API is used in the application:

      +----------------------------------------+---------------------------------------+-------------------------------------------------------------------+
      | Feature                                | API                                   | Description                                                       |
      +========================================+=======================================+===================================================================+
      | Available Devices                      | ``ov::Core::get_available_devices``,  | Get available devices information and configuration for inference |
      |                                        | ``ov::Core::get_property``            |                                                                   |
      +----------------------------------------+---------------------------------------+-------------------------------------------------------------------+

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_inference_engine_samples_hello_classification_README>`.
   
   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/cpp/hello_query_device/main.cpp 
         :language: cpp

How It Works
############

The sample queries all available OpenVINO™ Runtime devices, prints their supported metrics and plugin configuration parameters.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

To see quired information, run the following:

.. code-block:: console
   
   hello_query_device

Sample Output
#############

The application prints all available devices with their supported metrics and default values for configuration parameters:

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

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`

@endsphinxdirective

