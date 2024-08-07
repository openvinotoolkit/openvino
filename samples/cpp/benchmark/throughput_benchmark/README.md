# Throughput Benchmark C++ Sample

This sample demonstrates how to estimate performance of a model using Asynchronous Inference Request API in throughput mode. Unlike [demos](https://docs.openvino.ai/2024/omz_demos.html) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

The reported results may deviate from what [benchmark_app](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html) reports. One example is model input precision for computer vision tasks. benchmark_app sets ``uint8``, while the sample uses default model precision which is usually ``float32``.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/throughput-benchmark.html)

## Requirements

| Options                     | Values                                                                                                                         |
| ----------------------------| -------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | [yolo-v3-tf](https://docs.openvino.ai/2024/omz_models_model_yolo_v3_tf.html),                                                  |
|                             | [face-detection-](https://docs.openvino.ai/2024/omz_models_model_face_detection_0200.html)                                     |
| Model Format                | OpenVINO™ toolkit Intermediate Representation                                                                                  |
|                             | (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                              |
| Supported devices           | [All](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html)                           |
| Other language realization  | [Python](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/throughput-benchmark.html)                              |

The following C++ API is used in the application:

| Feature                  | API                                          | Description                                  |
| -------------------------| ---------------------------------------------|----------------------------------------------|
| OpenVINO Runtime Version | ``ov::get_openvino_version``                 | Get Openvino API version.                    |
| Basic Infer Flow         | ``ov::Core``, ``ov::Core::compile_model``,   | Common API to do inference: compile a model, |
|                          | ``ov::CompiledModel::create_infer_request``, | create an infer request,                     |
|                          | ``ov::InferRequest::get_tensor``             | configure input tensors.                     |
| Asynchronous Infer       | ``ov::InferRequest::start_async``,           | Do asynchronous inference with callback.     |
|                          | ``ov::InferRequest::set_callback``           |                                              |
| Model Operations         | ``ov::CompiledModel::inputs``                | Get inputs of a model.                       |
| Tensor Operations        | ``ov::Tensor::get_shape``,                   | Get a tensor shape and its data.             |
|                          | ``ov::Tensor::data``                         |                                              |


