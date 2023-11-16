# Hello Query Device Python Sample

This sample demonstrates how to show OpenVINO™ Runtime devices and prints their metrics and default configuration values using [Query Device API feature](..\..\..\docs\articles_en\openvino_workflow\openvino_intro\Device_Plugins\config_properties.md).

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_query_device.md)

## Requirements

| Options                     | Values                                                                                            |
| ----------------------------| --------------------------------------------------------------------------------------------------|
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)    |
| Other language realization  | [C++](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_query_device.md) |

The following Python API is used in the application:

| Feature       | API                                                                                                                                                                                     | Description                            |
| --------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Basic         | [openvino.runtime.Core](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html)                                                                      | Common API                             |
| Query Device  | [openvino.runtime.Core.available_devices](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.available_devices) ,          | Get device properties                  |
|               | [openvino.runtime.Core.get_metric](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_metric) ,  |                                        |
|               | [openvino.runtime.Core.get_config](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config)    |                                        |

