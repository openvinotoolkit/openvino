# Hello Reshape SSD C++ Sample

This sample demonstrates how to do synchronous inference of object detection models using [input reshape feature](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_ShapeInference.html).
Models with only one input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_hello_reshape_ssd_README.html)

## Requirements

| Options                     | Values                                                                                                                                   |
| ----------------------------| -----------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [person-detection-retail-0013](https://docs.openvino.ai/nightly/omz_models_model_person_detection_retail_0013.html)                      |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                          |
| Supported devices           | [All](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)                                      |
| Other language realization  | [Python](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README.html)               |

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


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_hello_classification_README.html).