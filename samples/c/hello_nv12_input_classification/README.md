# Hello NV12 Input Classification C Sample

This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

Hello NV12 Input Classification C Sample demonstrates how to use the NV12 automatic input pre-processing API in your applications.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_nv12_input_classification.md)

## Requirements

| Options                     | Values                                                                                                       |
| ----------------------------| -------------------------------------------------------------------------------------------------------------|
| Validated Models            | [alexnet](https://docs.openvino.ai/2023.2/omz_models_model_alexnet.html)                                     |
| Model Format                | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                               |
| Validated images            | An uncompressed image in the NV12 color format - \*.yuv                                                      |
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)               |
| Other language realization  | [C++](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_nv12_input_classification.md) |

The following C++ API is used in the application:

| Feature                   | API                                                       | Description                                            |
| --------------------------| ----------------------------------------------------------|--------------------------------------------------------|
| Node Operations           | ``ov_port_get_any_name``                                  | Get a layer name                                       |
| Infer Request Operations  | ``ov_infer_request_set_tensor``,                          | Operate with tensors                                   |
|                           | ``ov_infer_request_get_output_tensor_by_index``           |                                                        |
| Preprocessing             | ``ov_preprocess_input_tensor_info_set_color_format``,     | Change the color format of the input data              |
|                           | ``ov_preprocess_preprocess_steps_convert_element_type``,  |                                                        |
|                           | ``ov_preprocess_preprocess_steps_convert_color``          |                                                        |


Basic Inference Engine API is covered by [Hello Classification C sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_classification.md).


