# Hello Reshape SSD Python Sample

This sample demonstrates how to do synchronous inference of object detection models using [Shape Inference feature](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-input-output/changing-input-shape.html).

Models with only 1 input and output are supported.

## Requirements

| Options                     | Values                                                                                                   |
| ----------------------------| ---------------------------------------------------------------------------------------------------------|
| Validated Layout            | NCHW                                                                                                     |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                                |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)      |
| Other language realization  | [C++](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-reshape-ssd.html)                            |

The following Python API is used in the application:

| Feature          | API                                                                                                                                                                        | Description          |
| -----------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| Model Operations | [openvino.runtime.Model.reshape](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape),               | Managing of model    |
|                  | [openvino.runtime.Model.input](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input),                   |                      |
|                  | [openvino.runtime.Output.get_any_name](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Output.html#openvino.runtime.Output.get_any_name),  |                      |
|                  | [openvino.runtime.PartialShape](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.PartialShape.html)                                         |                      |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).
