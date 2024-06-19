.. {#openvino_2_0_configure_devices}

Configuring Devices
===================


.. meta::
   :description: Openvino Runtime API 2.0 has introduced properties that unify 
                 metrics and configuration key concepts, which simplifies the 
                 configuration of inference devices.


The Inference Engine API provides the ability to configure devices with configuration keys and obtain device-specific metrics. The values retrived from `InferenceEngine::Core::GetConfig <namespaceInferenceEngine.html#doxid-namespace-inference-engine-1aff2231f886c9f8fc9c226fd343026789>`__ are requested by the string name, while the return type is `InferenceEngine::Parameter <namespaceInferenceEngine.html#doxid-namespace-inference-engine-1aff2231f886c9f8fc9c226fd343026789>`__ , which results in users not knowing what the actual type is stored in this parameter.

API 2.0 solves these issues by introducing :doc:`properties <openvino_docs_OV_UG_query_api>`, which unify metrics and configuration key concepts. The main advantage is that they have the C++ type:

.. code-block:: sh

   static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};


where the property can be requested from an inference device as:


.. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
    :language: cpp
    :fragment: core_get_ro_property


The snippets in the following sections demonstrate the device configurations for migrating from Inference Engine to API 2.0.

.. note::

   The Inference Engine API is a **legacy solution** and it is recomended to use API 2.0. If you want to learn more about Inference Engine API, its configuration and how to obtain device-specific metrics from it, check the following `article <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html>`__ from the 2021.4 version of OpenVINO documentation.

Setting Configuration Values
############################

**Inference Engine API**


.. tab-set::

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_set_config

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_load_network

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: executable_network_set_config

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_set_config

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_load_network

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: executable_network_set_config

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_set_config

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_load_network

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: executable_network_set_config



**API 2.0**


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_set_property

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_compile_model

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: compiled_model_set_property

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_set_property

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_compile_model

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: compiled_model_set_property

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Devices
             :sync: devices

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_set_property

          .. tab-item:: Model Loading
             :sync: model-loading

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_compile_model

          .. tab-item:: Execution
             :sync: execution

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: compiled_model_set_property


Getting Information
###################

**Inference Engine API**


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_get_config

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_get_metric

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: executable_network_set_config

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: executable_network_get_metric

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_get_config

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_get_metric

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: executable_network_set_config

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: executable_network_get_metric

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_get_config

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_get_metric

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: executable_network_set_config

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: executable_network_get_metric


**API 2.0**


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_get_rw_property

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: core_get_ro_property

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: compiled_model_get_rw_property

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.py
                 :language: python
                 :fragment: compiled_model_get_ro_property

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_get_rw_property

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: core_get_ro_property

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: compiled_model_get_rw_property

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
                 :language: cpp
                 :fragment: compiled_model_get_ro_property

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-configuration

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_get_rw_property

          .. tab-item:: Device metrics
             :sync: device-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: core_get_ro_property

          .. tab-item:: Execution config
             :sync: execution-config

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: compiled_model_get_rw_property

          .. tab-item:: Execution metrics
             :sync: execution-metrics

             .. doxygensnippet:: docs/snippets/ov_properties_migration.c
                 :language: c
                 :fragment: compiled_model_get_ro_property

Configuration to property mapping
#################################

+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| Inference Engine Configuration        | API 2.0 C++                                      | API 2.0 Python                                     |
+=======================================+==================================================+====================================================+
| CPU_DENORMALS_OPTIMIZATION            | ov::intel_cpu::denormals_optimization            | props.intel_cpu.denormals_optimization             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE | ov::intel_cpu::sparse_weights_decompression_rate | props.intel_cpu.sparse_weights_decompression_rate  |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| GPU_HOST_TASK_PRIORITY                | ov::intel_gpu::hint::host_task_priority          | props.intel_gpu.hint.host_task_priority            |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| GPU_ENABLE_LOOP_UNROLLING             | ov::intel_gpu::enable_loop_unrolling             | props.intel_gpu.host_task_priority                 |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| GPU_THROUGHPUT_STREAMS                | ov::num_streams                                  | props.streams.num                                  |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| MULTI_DEVICE_PRIORITIES               | ov::device::priorities                           | props.device.priorities                            |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| MODEL_PRIORITY                        | ov::hint::model_priority                         | props.hint.model_priority                          |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| PERFORMANCE_HINT                      | ov::hint::performance_mode                       | props.hint.performance_mode                        |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| PERFORMANCE_HINT_NUM_REQUESTS         | ov::hint::num_requests                           | props.hint.num_requests                            |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| ALLOW_AUTO_BATCHING                   | ov::hint::allow_auto_batching                    | props.hint.allow_auto_batching                     |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| AUTO_BATCH_DEVICE_CONFIG              | ov::device::priorities                           | props.device.priorities                            |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| AUTO_BATCH_TIMEOUT                    | ov::auto_batch_timeout                           | props.auto_batch_timeout                           |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| CPU_THREADS_NUM                       | ov::inference_num_threads                        | props.inference_num_threads                        |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| CPU_BIND_THREAD                       | ov::hint::enable_cpu_pinning                     | props.hint.enable_cpu_pinning                      |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| CPU_THROUGHPUT_STREAMS                | ov::num_streams                                  | props.streams.num                                  |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| PERF_COUNT                            | ov::enable_profiling                             | props.enable_profiling                             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| LOG_LEVEL                             | ov::log::level                                   | props.log.level                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| DEVICE_ID                             | ov::device::id                                   | props.device.id                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| ENFORCE_BF16                          | ov::hint::inference_precision(ov::element::bf16) | props.hint.inference_precision(openvino.Type.bf16) |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| CACHE_DIR                             | ov::cache_dir                                    | props.cache_dir                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| FORCE_TBB_TERMINATE                   | ov::force_tbb_terminate                          | props.force_tbb_terminate                          |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+

.. note:: 
   API 2.0 Python "import openvino.properties as props"

More references
+++++++++++++++

InferenceEngine Engine Configuration
------------------------------------
- `src/inference/include/ie/ie_plugin_config.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/ie/ie_plugin_config.hpp>`__
- `src/inference/include/ie/cpu/cpu_config.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/ie/cpu/cpu_config.hpp>`__
- `src/inference/include/ie/gpu/gpu_config.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/ie/gpu/gpu_config.hpp>`__

API 2.0 C++ properties
----------------------
- `src/inference/include/openvino/runtime/properties.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/openvino/runtime/properties.hpp>`__
- `src/inference/include/openvino/runtime/intel_cpu/properties.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/openvino/runtime/intel_cpu/properties.hpp>`__
- `src/inference/include/openvino/runtime/intel_gpu/properties.hpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/inference/include/openvino/runtime/intel_gpu/properties.hpp>`__

API 2.0 Python properties
-------------------------
- `src/bindings/python/src/pyopenvino/core/properties/properties.cpp <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/bindings/python/src/pyopenvino/core/properties/properties.cpp>`__
- `src/bindings/python/tests/test_runtime/test_properties.py <https://github.com/openvinotoolkit/openvino/blob/releases/2023/3/src/bindings/python/tests/test_runtime/test_properties.py>`__

Ask questions
-------------
- Go to "Forum", "Support", "Github"
