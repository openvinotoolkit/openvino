# Hello Query Device C++ Sample

This sample demonstrates how to execute an query OpenVINO™ Runtime devices, prints their metrics and default configuration values, using :doc:`Properties API <openvino_docs_OV_UG_query_api>`.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_query_device.md)

## Requirements

| Options                       | Values                                                                                                  |
| ------------------------------| --------------------------------------------------------------------------------------------------------|
| Supported devices             | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)          |
| Other language realization    | [Python](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_query_device.md) |

The following C++ API is used in the application:

| Feature                  | API                                   | Description                                                       |
| -------------------------| --------------------------------------|-------------------------------------------------------------------|
| Available Devices        | ``ov::Core::get_available_devices``,  | Get available devices information and configuration for inference |
|                          | ``ov::Core::get_property``            |                                                                   |

Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_classification.md).
