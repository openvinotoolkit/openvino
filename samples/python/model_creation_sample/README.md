# Model Creation Python Sample

This sample demonstrates how to run inference using a [model](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html) built on the fly that uses weights from the LeNet classification model, which is known to work well on digit classification tasks. You do not need an XML file, the model is created from the source code on the fly.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/model-creation.html)

## Requirements

| Options                     | Values                                                                                                      |
| ----------------------------| ------------------------------------------------------------------------------------------------------------|
| Validated Models            | LeNet                                                                                                       |
| Model Format                | Model weights file (\*.bin)                                                                                 |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)         |
| Other language realization  | [C++](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/model-creation.html)                                  |

The following OpenVINO Python API is used in the application:

| Feature           | API                                                                                                                                                       | Description                                               |
| ------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Model Operations  | [openvino.runtime.Model](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Model.html) ,                                    | Managing of model                                         |
|                   | [openvino.runtime.set_batch](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.set_batch.html) ,                            |                                                           |
|                   | [openvino.runtime.Model.input](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input)   |                                                           |
| Opset operations  | [openvino.runtime.op.Parameter](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.op.Parameter.html),                       | Description of a model topology using OpenVINO Python API |
|                   | [openvino.runtime.op.Constant](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.op.Constant.html) ,                        |                                                           |
|                   | [openvino.runtime.opset8.convolution](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.convolution.html) ,          |                                                           |
|                   | [openvino.runtime.opset8.add](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.add.html) ,                          |                                                           |
|                   | [openvino.runtime.opset1.max_pool](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset1.max_pool.html) ,                |                                                           |
|                   | [openvino.runtime.opset8.reshape](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.reshape.html) ,                  |                                                           |
|                   | [openvino.runtime.opset8.matmul](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.matmul.html) ,                    |                                                           |
|                   | [openvino.runtime.opset8.relu](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.relu.html) ,                        |                                                           |
|                   | [openvino.runtime.opset8.softmax](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.opset8.softmax.html)                    |                                                           |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).
