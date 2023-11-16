# Hello Classification Python Sample

This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API. 

Models with only 1 input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_classification.md)

## Requirements

| Options                     | Values                                                                                                |
| ----------------------------| ------------------------------------------------------------------------------------------------------|
| Validated Models            | [alexnet](https://docs.openvino.ai/2023.2/omz_models_model_alexnet.html),                             |
|                             | [googlenet-v1](https://docs.openvino.ai/2023.2/omz_models_model_googlenet_v1.html)                    |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                             |
| Supported devices           | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)        |
| Other language realization  | [C++](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_classification.md),  |
|                             | [C](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_classification.md)       |

The following Python API is used in the application:

| Feature           | API                                                                                                                                                                                                                                   | Description                                                                                                     |
| ------------------| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Basic Infer Flow  | [openvino.runtime.Core](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html) ,                                                                                                                  |                                                                                                                 |
|                   | [openvino.runtime.Core.read_model](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.read_model),                                                                       |                                                                                                                 |
|                   | [openvino.runtime.Core.compile_model](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.compile_model)                                                                  | Common API to do inference                                                                                      |
| Synchronous Infer | [openvino.runtime.CompiledModel.infer_new_request](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.infer_new_request),                              | Do synchronous inference                                                                                        |
| Model Operations  | [openvino.runtime.Model.inputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.inputs),                                                                            | Managing of model                                                                                               |
|                   | [openvino.runtime.Model.outputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.outputs),                                                                          |                                                                                                                 |
| Preprocessing     | [openvino.preprocess.PrePostProcessor](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html),                                                                                     | Set image of the original size as input for a model with other input size.                                      |
|                   | [openvino.preprocess.InputTensorInfo.set_element_type](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_element_type),                 | Resize and layout conversions will be performed automatically by the corresponding plugin just before inference |
|                   | [openvino.preprocess.InputTensorInfo.set_layout](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_layout),                             |                                                                                                                 |
|                   | [openvino.preprocess.InputTensorInfo.set_spatial_static_shape](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_spatial_static_shape), |                                                                                                                 |
|                   | [openvino.preprocess.PreProcessSteps.resize](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html#openvino.preprocess.PreProcessSteps.resize),                                     |                                                                                                                 |
|                   | [openvino.preprocess.InputModelInfo.set_layout](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputModelInfo.html#openvino.preprocess.InputModelInfo.set_layout),                                |                                                                                                                 |
|                   | [openvino.preprocess.OutputTensorInfo.set_element_type](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.OutputTensorInfo.html#openvino.preprocess.OutputTensorInfo.set_element_type),              |                                                                                                                 |
|                   | [openvino.preprocess.PrePostProcessor.build](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html#openvino.preprocess.PrePostProcessor.build)                                     |                                                                                                                 |

