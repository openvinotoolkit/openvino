# Hello Query Device C++ Sample

This sample demonstrates how to execute an query OpenVINO™ Runtime devices, prints their metrics and default configuration values, using [Properties API](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html).

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-query-device.html)

## Requirements

| Options                       | Values                                                                                                                      |
| ------------------------------| ----------------------------------------------------------------------------------------------------------------------------|
| Supported devices             | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                         |
| Other language realization    | [Python](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-query-device.html)                                           |

The following C++ API is used in the application:

| Feature                  | API                                   | Description                                                       |
| -------------------------| --------------------------------------|-------------------------------------------------------------------|
| Available Devices        | ``ov::Core::get_available_devices``,  | Get available devices information and configuration for inference |
|                          | ``ov::Core::get_property``            |                                                                   |

Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).
