# Configuring Devices {#openvino_2_0_configure_devices}

### Introduction

Inference Engine API provides the [ability to configure devices](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html) via configuration keys and [get device specific metrics](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html#getmetric). The values taken from `InferenceEngine::Core::GetConfig` are requested by the string name, while the return type is `InferenceEngine::Parameter`, making users lost on what the actual type is stored in this parameter.

OpenVINO Runtime API 2.0 solves these issues by introducing [properties](../supported_plugins/config_properties.md), which unify metrics and configuration key concepts. Their main advantage is that they have the C++ type:

```
static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};
```

The property can be requested from an inference device as:

@snippet ov_properties_migration.cpp core_get_ro_property

The snippets below explain how to migrate from an Inference Engine device configuration to OpenVINO Runtime API 2.0.

### Set configuration values

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{Devices}

@snippet docs/snippets/ov_properties_migration.cpp core_set_config

@endsphinxtab

@sphinxtab{Model Loading}

@snippet docs/snippets/ov_properties_migration.cpp core_load_network

@endsphinxtab

@sphinxtab{Execution}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_set_config

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{Devices}

@snippet docs/snippets/ov_properties_migration.py core_set_config

@endsphinxtab

@sphinxtab{Model Loading}

@snippet docs/snippets/ov_properties_migration.py core_load_network

@endsphinxtab

@sphinxtab{Execution}

@snippet docs/snippets/ov_properties_migration.py executable_network_set_config

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{Devices}

@snippet docs/snippets/ov_properties_migration.cpp core_set_property

@endsphinxtab

@sphinxtab{Model Loading}

@snippet docs/snippets/ov_properties_migration.cpp core_compile_model

@endsphinxtab

@sphinxtab{Execution}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_set_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{Devices}

@snippet docs/snippets/ov_properties_migration.py core_set_property

@endsphinxtab

@sphinxtab{Model Loading}

@snippet docs/snippets/ov_properties_migration.py core_compile_model

@endsphinxtab

@sphinxtab{Execution}

@snippet docs/snippets/ov_properties_migration.py compiled_model_set_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

### Get Information

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{Device Configuration}

@snippet docs/snippets/ov_properties_migration.cpp core_get_config

@endsphinxtab

@sphinxtab{Device metrics}

@snippet docs/snippets/ov_properties_migration.cpp core_get_metric

@endsphinxtab

@sphinxtab{Execution config}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_get_config

@endsphinxtab

@sphinxtab{Execution metrics}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_get_metric

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{Device Configuration}

@snippet docs/snippets/ov_properties_migration.py core_get_config

@endsphinxtab

@sphinxtab{Device metrics}

@snippet docs/snippets/ov_properties_migration.py core_get_metric

@endsphinxtab

@sphinxtab{Execution config}

@snippet docs/snippets/ov_properties_migration.py executable_network_get_config

@endsphinxtab

@sphinxtab{Execution metrics}

@snippet docs/snippets/ov_properties_migration.py executable_network_get_metric

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{Device Configuration}

@snippet docs/snippets/ov_properties_migration.cpp core_get_rw_property

@endsphinxtab

@sphinxtab{Device metrics}

@snippet docs/snippets/ov_properties_migration.cpp core_get_ro_property

@endsphinxtab

@sphinxtab{Execution config}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_get_rw_property

@endsphinxtab

@sphinxtab{Execution metrics}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_get_ro_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{Device Configuration}

@snippet docs/snippets/ov_properties_migration.py core_get_rw_property

@endsphinxtab

@sphinxtab{Device metrics}

@snippet docs/snippets/ov_properties_migration.py core_get_ro_property

@endsphinxtab

@sphinxtab{Execution config}

@snippet docs/snippets/ov_properties_migration.py compiled_model_get_rw_property

@endsphinxtab

@sphinxtab{Execution metrics}

@snippet docs/snippets/ov_properties_migration.py compiled_model_get_ro_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset
