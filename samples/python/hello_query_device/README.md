# Hello Query Device Python Sample

This sample demonstrates how to show OpenVINO™ Runtime devices and prints their metrics and default configuration values using [Query Device API feature](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html).

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-query-device.html)

## Requirements

| Options                     | Values                                                                                                  |
| ----------------------------| --------------------------------------------------------------------------------------------------------|
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)     |
| Other language realization  | [C++](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-query-device.html)                          |

The following Python API is used in the application:

| Feature       | API                                                                                                                                                                                     | Description                            |
| --------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Basic         | [openvino.runtime.Core](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Core.html)                                                                      | Common API                             |
| Query Device  | [openvino.runtime.Core.available_devices](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.available_devices) ,          | Get device properties                  |

