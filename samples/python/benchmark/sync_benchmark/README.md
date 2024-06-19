# Sync Benchmark Python Sample

This sample demonstrates how to estimate performance of a model using Synchronous Inference Request API. It makes sense to use synchronous inference only in latency oriented scenarios. Models with static input shapes are supported. Unlike [demos](https://docs.openvino.ai/2023.3/omz_demos.html) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.3/openvino_sample_sync_benchmark.html)

## Requirements

| Options                     | Values                                                                                               |
| ----------------------------| -----------------------------------------------------------------------------------------------------|
| Validated Models            | [alexnet](https://docs.openvino.ai/2023.3/omz_models_model_alexnet.html),                            |
|                             | [googlenet-v1](https://docs.openvino.ai/2023.3/omz_models_model_googlenet_v1.html),                  |
|                             | [yolo-v3-tf](https://docs.openvino.ai/2023.3/omz_models_model_yolo_v3_tf.html),                      |
|                             | [face-detection-0200](https://docs.openvino.ai/2023.3/omz_models_model_face_detection_0200.html)     |
| Model Format                | OpenVINO™ toolkit Intermediate Representation                                                        |
|                             | (\*.xml + \*.bin), ONNX (\*.onnx)                                                                    |
| Supported devices           | [All](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)  |
| Other language realization  | [C++](https://docs.openvino.ai/2023.3/openvino_sample_sync_benchmark.html)                           |

The following Python API is used in the application:

| Feature                   | API                                             | Description                                  |
| --------------------------| ------------------------------------------------|----------------------------------------------|
| OpenVINO Runtime Version  | [openvino.runtime.get_version]                  | Get Openvino API version.                    |
| Basic Infer Flow          | [openvino.runtime.Core],                        | Common API to do inference: compile a model, |
|                           | [openvino.runtime.Core.compile_model],          | configure input tensors.                     |
|                           | [openvino.runtime.InferRequest.get_tensor]      |                                              |
| Synchronous Infer         | [openvino.runtime.InferRequest.infer],          | Do synchronous inference.                    |
| Model Operations          | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
| Tensor Operations         | [openvino.runtime.Tensor.get_shape],            | Get a tensor shape and its data.             |
|                           | [openvino.runtime.Tensor.data]                  |                                              |
