# Hello Reshape SSD C++ Sample

This sample demonstrates how to do synchronous inference of object detection models using [input reshape feature](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-input-output/changing-input-shape.html).
Models with only one input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-reshape-ssd.html)

## Requirements

| Options                     | Values                                                                                                                                   |
| ----------------------------| -----------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013)  |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                          |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                                      |
| Other language realization  | [Python](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-reshape-ssd.html)                                           |

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


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).
