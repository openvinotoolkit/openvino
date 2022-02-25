# Configure devices {#openvino_2_0_configure_devices}

### Introduction

Inference Engine API provides an [ability to configure devices](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html) via configuration keys and [get device specific metrics](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html#getmetric). The values taken from `InferenceEngine::Core::GetConfig` are requested by its string name, while return type is `InferenceEngine::Parameter` and users don't know what is the actual type is stored in this parameter.

OpenVINO Runtime API 2.0 solves these issues by introducing [properties](../supported_plugins/config_properties.md), which unify metrics and configuration keys, but the main advantage of properties - they have C++ type:

```
static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};
```

Which can be extracted from an inference device as:

@snippet ov_properties_migration.cpp core_get_ro_property

The snippets below show how to migrate from Inference Engine device configuration to OpenVINO Runtime API 2.0 steps.

### Set configuration values

Inference Engine API:

@sphinxdirective

.. tab:: Core::SetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_set_config]

.. tab:: Core::LoadNetwork

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_load_network]

.. tab:: ExecutableNetwork::SetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [executable_network_set_config]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: Core::SetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_set_property]

.. tab:: Core::LoadNetwork

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_compile_model]

.. tab:: ExecutableNetwork::SetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [compiled_model_set_property]

@endsphinxdirective

### Get information

Inference Engine API:

@sphinxdirective

.. tab:: Core::GetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_get_config]

.. tab:: Core::GetMetric

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_get_metric]

.. tab:: ExecutableNetwork::GetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [executable_network_get_metric]

.. tab:: ExecutableNetwork::GetMetric

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [executable_network_get_config]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: Core::GetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_get_rw_property]

.. tab:: Core::GetMetric

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [core_get_ro_property]

.. tab:: ExecutableNetwork::GetConfig

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [compiled_model_get_rw_property]

.. tab:: ExecutableNetwork::GetMetric

    .. doxygensnippet:: docs/snippets/ov_properties_migration.cpp
       :language: cpp
       :fragment: [compiled_model_get_ro_property]

@endsphinxdirective
