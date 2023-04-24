# Configuring Devices {#openvino_2_0_configure_devices}

@sphinxdirective


Inference Engine API provides the `ability to configure devices <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html>`__ via configuration keys and `get device specific metrics <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html#getmetric>`__. The values taken from `InferenceEngine::Core::GetConfig <namespaceInferenceEngine.html#doxid-namespace-inference-engine-1aff2231f886c9f8fc9c226fd343026789>`__ are requested by the string name, while the return type is `InferenceEngine::Parameter <namespaceInferenceEngine.html#doxid-namespace-inference-engine-1aff2231f886c9f8fc9c226fd343026789>`__, making users lost on what the actual type is stored in this parameter.

API 2.0 solves these issues by introducing :doc:`properties <openvino_docs_OV_UG_query_api>`, which unify metrics and configuration key concepts. The main advantage is that they have the C++ type:

.. code-block::

   static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};


where the property can be requested from an inference device as:


.. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
    :language: cpp
    :fragment: core_get_ro_property


The snippets in the following sections demonstrate the device configurations for migrating from Inference Engine to API 2.0.

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


## Getting Information

**Inference Engine API**


.. tab-set::

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-config

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

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-config

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

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-config

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
             :sync: device-config

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

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-config

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

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: Device Configuration
             :sync: device-config

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


@endsphinxdirective
