# Hello NV12 Input Classification C Sample

This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

Hello NV12 Input Classification C Sample demonstrates how to use the NV12 automatic input pre-processing API in your applications.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README.html)

## Requirements

| Options                     | Values                                                                                                               |
| ----------------------------| ---------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [alexnet](https://docs.openvino.ai/2023.2/omz_models_model_alexnet.html)                                             |
| Model Format                | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                       |
| Validated images            | An uncompressed image in the NV12 color format - \*.yuv                                                              |
| Supported devices           | [All](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)                  |
| Other language realization  | [C++](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_hello_nv12_input_classification_README.html) |

The following C++ API is used in the application:

| Feature                   | API                                                       | Description                                            |
| --------------------------| ----------------------------------------------------------|--------------------------------------------------------|
| Node Operations           | ``ov_port_get_any_name``                                  | Get a layer name                                       |
| Infer Request Operations  | ``ov_infer_request_set_tensor``,                          | Operate with tensors                                   |
|                           | ``ov_infer_request_get_output_tensor_by_index``           |                                                        |
| Preprocessing             | ``ov_preprocess_input_tensor_info_set_color_format``,     | Change the color format of the input data              |
|                           | ``ov_preprocess_preprocess_steps_convert_element_type``,  |                                                        |
|                           | ``ov_preprocess_preprocess_steps_convert_color``          |                                                        |


Basic Inference Engine API is covered by [Hello Classification C sample](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_c_samples_hello_classification_README.html).


