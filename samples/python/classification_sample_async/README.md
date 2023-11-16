# Image Classification Async Python Sample

This sample demonstrates how to do inference of image classification models using Asynchronous Inference Request API.

Models with only 1 input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_image_classification_async.md)

## Requirements

| Options                    | Values                                                                                                    |
| ---------------------------| ----------------------------------------------------------------------------------------------------------|
| Validated Models           | [alexnet](https://docs.openvino.ai/2023.2/omz_models_model_alexnet.html)                                  |
| Model Format               | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                                 |
| Supported devices          | [All](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)            |
| Other language realization | [C++](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_image_classification_async.md) |

The following Python API is used in the application:

| Feature            | API                                                                                                                                                                                                   | Description               |
| -------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| Asynchronous Infer | [openvino.runtime.AsyncInferQueue](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html),                                                             | Do asynchronous inference |
|                    | [openvino.runtime.AsyncInferQueue.set_callback](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html#openvino.runtime.AsyncInferQueue.set_callback),  |                           |
|                    | [openvino.runtime.AsyncInferQueue.start_async](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html#openvino.runtime.AsyncInferQueue.start_async),    |                           |
|                    | [openvino.runtime.AsyncInferQueue.wait_all](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html#openvino.runtime.AsyncInferQueue.wait_all),          |                           |
|                    | [openvino.runtime.InferRequest.results](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.results)                      |                           |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python Sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_hello_classification.md).
