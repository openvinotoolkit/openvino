# 配置设备 {#openvino_2_0_configure_devices_zh_CN}

推理引擎 API 提供通过配置密钥来[配置设备](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html)及[获取设备特定指标](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_InferenceEngine_QueryAPI.html#getmetric)的功能。按字符串名称请求从 `InferenceEngine::Core::GetConfig` 中提取的值，而返回类型为 `InferenceEngine::Parameter`，导致用户无法确定此参数中存储的到底是哪种类型。

API 2.0 通过引入[属性](../../../OV_Runtime_UG/supported_plugins/config_properties.md)来统一指标和配置密钥概念，从而解决了这些问题。主要优势在于它们具有 C++ 类型：

```
static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};
```

其中，可以从推理设备请求属性，如下所示：

@snippet ov_properties_migration.cpp core_get_ro_property

以下部分中的代码片段演示了用于从推理引擎迁移到 API 2.0 的设备配置。

## 设置配置值

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{设备}

@snippet docs/snippets/ov_properties_migration.cpp core_set_config

@endsphinxtab

@sphinxtab{模型加载}

@snippet docs/snippets/ov_properties_migration.cpp core_load_network

@endsphinxtab

@sphinxtab{执行}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_set_config

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{设备}

@snippet docs/snippets/ov_properties_migration.py core_set_config

@endsphinxtab

@sphinxtab{模型加载}

@snippet docs/snippets/ov_properties_migration.py core_load_network

@endsphinxtab

@sphinxtab{执行}

@snippet docs/snippets/ov_properties_migration.py executable_network_set_config

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{设备}

@snippet docs/snippets/ov_properties_migration.cpp core_set_property

@endsphinxtab

@sphinxtab{模型加载}

@snippet docs/snippets/ov_properties_migration.cpp core_compile_model

@endsphinxtab

@sphinxtab{执行}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_set_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{设备}

@snippet docs/snippets/ov_properties_migration.py core_set_property

@endsphinxtab

@sphinxtab{模型加载}

@snippet docs/snippets/ov_properties_migration.py core_compile_model

@endsphinxtab

@sphinxtab{执行}

@snippet docs/snippets/ov_properties_migration.py compiled_model_set_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

## 获取信息

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{设备配置}

@snippet docs/snippets/ov_properties_migration.cpp core_get_config

@endsphinxtab

@sphinxtab{设备指标}

@snippet docs/snippets/ov_properties_migration.cpp core_get_metric

@endsphinxtab

@sphinxtab{执行配置}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_get_config

@endsphinxtab

@sphinxtab{执行指标}

@snippet docs/snippets/ov_properties_migration.cpp executable_network_get_metric

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{设备配置}

@snippet docs/snippets/ov_properties_migration.py core_get_config

@endsphinxtab

@sphinxtab{设备指标}

@snippet docs/snippets/ov_properties_migration.py core_get_metric

@endsphinxtab

@sphinxtab{执行配置}

@snippet docs/snippets/ov_properties_migration.py executable_network_get_config

@endsphinxtab

@sphinxtab{执行指标}

@snippet docs/snippets/ov_properties_migration.py executable_network_get_metric

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{设备配置}

@snippet docs/snippets/ov_properties_migration.cpp core_get_rw_property

@endsphinxtab

@sphinxtab{设备指标}

@snippet docs/snippets/ov_properties_migration.cpp core_get_ro_property

@endsphinxtab

@sphinxtab{执行配置}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_get_rw_property

@endsphinxtab

@sphinxtab{执行指标}

@snippet docs/snippets/ov_properties_migration.cpp compiled_model_get_ro_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{设备配置}

@snippet docs/snippets/ov_properties_migration.py core_get_rw_property

@endsphinxtab

@sphinxtab{设备指标}

@snippet docs/snippets/ov_properties_migration.py core_get_ro_property

@endsphinxtab

@sphinxtab{执行配置}

@snippet docs/snippets/ov_properties_migration.py compiled_model_get_rw_property

@endsphinxtab

@sphinxtab{执行指标}

@snippet docs/snippets/ov_properties_migration.py compiled_model_get_ro_property

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset
