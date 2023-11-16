# Hello Reshape SSD Python Sample

This sample demonstrates how to do synchronous inference of object detection models using [Shape Inference feature](..\..\..\docs\articles_en\openvino_workflow\openvino_intro\ShapeInference.md).  

Models with only 1 input and output are supported.

## Requirements

| Options                     | Values                                                                                            |
| ----------------------------| --------------------------------------------------------------------------------------------------|
| Validated Models            | [mobilenet-ssd](https://docs.openvino.ai/2023.2/omz_models_model_mobilenet_ssd.html)                     |
| Validated Layout            | NCHW                                                                      |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx) |
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)      |
| Other language realization  | [C++](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_reshape_ssd.md)   |

The following Python API is used in the application:

| Feature          | API                                                                                                                                                                        | Description          |
| -----------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| Model Operations | [openvino.runtime.Model.reshape](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape),               | Managing of model    |
|                  | [openvino.runtime.Model.input](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input),                   |                      |
|                  | [openvino.runtime.Output.get_any_name](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Output.html#openvino.runtime.Output.get_any_name),  |                      |
|                  | [openvino.runtime.PartialShape](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.PartialShape.html)                                         |                      |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_classification.md).
