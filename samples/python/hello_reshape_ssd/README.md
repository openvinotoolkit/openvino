# Hello Reshape SSD Python Sample

This sample demonstrates how to do synchronous inference of object detection models using [Shape Inference feature](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_ShapeInference.html).  

Models with only 1 input and output are supported.

## Requirements

| Options                     | Values                                                                                                   |
| ----------------------------| ---------------------------------------------------------------------------------------------------------|
| Validated Models            | [mobilenet-ssd](https://docs.openvino.ai/2023.2/omz_models_model_mobilenet_ssd.html)                     |
| Validated Layout            | NCHW                                                                                                     |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                                |
| Supported devices           | [All](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)      |
| Other language realization  | [C++](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_hello_reshape_ssd_README.html)   |

The following Python API is used in the application:

| Feature          | API                                                                                                                                                                        | Description          |
| -----------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| Model Operations | [openvino.runtime.Model.reshape](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape),               | Managing of model    |
|                  | [openvino.runtime.Model.input](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input),                   |                      |
|                  | [openvino.runtime.Output.get_any_name](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Output.html#openvino.runtime.Output.get_any_name),  |                      |
|                  | [openvino.runtime.PartialShape](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.PartialShape.html)                                         |                      |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_hello_classification_README.html).
