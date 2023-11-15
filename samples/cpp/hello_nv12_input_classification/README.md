# Hello NV12 Input Classification C++ Sample

This sample demonstrates how to execute an inference of image classification models with images in NV12 color format using Synchronous Inference Request API.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_nv12_input_classification.md)

## Requirements

| Options                     | Values                                                                                                     |
| ----------------------------| -----------------------------------------------------------------------------------------------------------|
| Validated Models            | [alexnet <omz_models_model_alexnet](https://docs.openvino.ai/2023.2/omz_models_model_alexnet.html)         |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                            |
| Validated images            | An uncompressed image in the NV12 color format - \*.yuv                                                    |
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)             |
| Other language realization  | [C](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_nv12_input_classification.md) |


The following C++ API is used in the application:

| Feature                  | API                                                         | Description                               |
| -------------------------| ------------------------------------------------------------|-------------------------------------------|
| Node Operations          | ``ov::Output::get_any_name``                                | Get a layer name                          |
| Infer Request Operations | ``ov::InferRequest::set_tensor``,                           | Operate with tensors                      |
|                          | ``ov::InferRequest::get_tensor``                            |                                           |
| Preprocessing            | ``ov::preprocess::InputTensorInfo::set_color_format``,      | Change the color format of the input data |
|                          | ``ov::preprocess::PreProcessSteps::convert_element_type``,  |                                           |
|                          | ``ov::preprocess::PreProcessSteps::convert_color``          |                                           |


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_classification.md).

