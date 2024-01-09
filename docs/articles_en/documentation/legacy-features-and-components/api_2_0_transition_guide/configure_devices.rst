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


