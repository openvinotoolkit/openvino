# Hello Reshape SSD C++ Sample

This sample demonstrates how to do synchronous inference of object detection models using [input reshape feature](https://docs.openvino.ai/2024/openvino-workflow/running-inference/changing-input-shape.html).
Models with only one input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-reshape-ssd.html)

## Requirements

| Options                     | Values                                                                                                                                   |
| ----------------------------| -----------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [person-detection-retail-0013](https://docs.openvino.ai/2024/omz_models_model_person_detection_retail_0013.html)                         |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                          |
| Supported devices           | [All](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html)                                     |
| Other language realization  | [Python](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-reshape-ssd.html)                                           |

The following C++ API is used in the application:

| Feature                  | API                                                         | Description                                    |
| -------------------------| ------------------------------------------------------------|------------------------------------------------|
| Node operations          | ``ov::Node::get_type_info``,                                | Get a node info                                |
|                          | ``ov::op::DetectionOutput::get_type_info_static``,          |                                                |
|                          | ``ov::Output::get_any_name``,                               |                                                |
|                          | ``ov::Output::get_shape``                                   |                                                |
| Model Operations         | ``ov::Model::get_ops``,                                     | Get model nodes, reshape input                 |
|                          | ``ov::Model::reshape``                                      |                                                |
| Tensor Operations        | ``ov::Tensor::data``                                        | Get a tensor data                              |
| Preprocessing            | ``ov::preprocess::PreProcessSteps::convert_element_type``,  | Model input preprocessing                      |
|                          | ``ov::preprocess::PreProcessSteps::convert_layout``         |                                                |


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-classification.html).
