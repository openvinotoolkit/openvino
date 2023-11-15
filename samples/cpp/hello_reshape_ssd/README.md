# Hello Reshape SSD C++ Sample

This sample demonstrates how to do synchronous inference of object detection models using [input reshape feature](..\..\..\docs\articles_en\openvino_workflow\openvino_intro\ShapeInference.md).
Models with only one input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_reshape_ssd.md)

## Requirements

| Options                     | Values                                                                                                               |
| ----------------------------| ---------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [person-detection-retail-0013](https://docs.openvino.ai/nightly/omz_models_model_person_detection_retail_0013.html)  |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                      |
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)                       |
| Other language realization  | [Python](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_reshape_ssd.md)               |

The following C++ API is used in the application:

| Feature                  | API                                                         | Description                                    |
| -------------------------| ------------------------------------------------------------|------------------------------------------------|
| Node operations          | ``ov::Node::get_type_info``,                                | Get a node info                                |
|                          | ``ngraph::op::DetectionOutput::get_type_info_static``,      |                                                |
|                          | ``ov::Output::get_any_name``,                               |                                                |
|                          | ``ov::Output::get_shape``                                   |                                                |
| Model Operations         | ``ov::Model::get_ops``,                                     | Get model nodes, reshape input                 |
|                          | ``ov::Model::reshape``                                      |                                                |
| Tensor Operations        | ``ov::Tensor::data``                                        | Get a tensor data                              |
| Preprocessing            | ``ov::preprocess::PreProcessSteps::convert_element_type``,  | Model input preprocessing                      |
|                          | ``ov::preprocess::PreProcessSteps::convert_layout``         |                                                |


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_classification.md).