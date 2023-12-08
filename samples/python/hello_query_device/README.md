# Hello Query Device Python Sample

This sample demonstrates how to show OpenVINO™ Runtime devices and prints their metrics and default configuration values using [Query Device API feature](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_query_api.html).

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html)

## Requirements

| Options                     | Values                                                                                                  |
| ----------------------------| --------------------------------------------------------------------------------------------------------|
| Supported devices           | [All](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)     |
| Other language realization  | [C++](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_hello_query_device_README.html) |

The following Python API is used in the application:

| Feature       | API                                                                                                                                                                                     | Description                            |
| --------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Basic         | [openvino.runtime.Core](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html)                                                                      | Common API                             |
| Query Device  | [openvino.runtime.Core.available_devices](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.available_devices) ,          | Get device properties                  |
|               | [openvino.runtime.Core.get_metric](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_metric) ,  |                                        |
|               | [openvino.runtime.Core.get_config](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config)    |                                        |

