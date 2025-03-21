# Sync Benchmark Python Sample

This sample demonstrates how to estimate performance of a model using Synchronous Inference Request API. It makes sense to use synchronous inference only in latency oriented scenarios. Models with static input shapes are supported. Unlike [demos](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/sync-benchmark.html)

## Requirements

| Options                     | Values                                                                                                                |
| ----------------------------| ----------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [yolo-v3-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf),                 |
|                             | [face-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-0200) |
| Model Format                | OpenVINO™ toolkit Intermediate Representation                                                                         |
|                             | (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                     |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                   |
| Other language realization  | [C++](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/sync-benchmark.html)                              |

The following Python API is used in the application:

| Feature                   | API                                             | Description                                  |
| --------------------------| ------------------------------------------------|----------------------------------------------|
| OpenVINO API Version      | [openvino.\_\_version__]                          | Get Openvino API version.                    |
| Basic Infer Flow          | [openvino.runtime.Core],                        | Common API to do inference: compile a model, |
|                           | [openvino.runtime.Core.compile_model],          | configure input tensors.                     |
|                           | [openvino.runtime.InferRequest.get_tensor]      |                                              |
| Synchronous Infer         | [openvino.runtime.InferRequest.infer],          | Do synchronous inference.                    |
| Model Operations          | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
| Tensor Operations         | [openvino.runtime.Tensor.get_shape],            | Get a tensor shape and its data.             |
|                           | [openvino.runtime.Tensor.data]                  |                                              |
