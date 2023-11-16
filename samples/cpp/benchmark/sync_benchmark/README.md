# Sync Benchmark C++ Sample

This sample demonstrates how to estimate performance of a model using Synchronous Inference Request API. It makes sense to use synchronous inference only in latency oriented scenarios. Models with static input shapes are supported. Unlike [demos](https://docs.openvino.ai/2023.2/omz_demos.html) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_sync_benchmark.md)

## Requirements

| Options                        | Values                                                                                                  |
| -------------------------------| --------------------------------------------------------------------------------------------------------|
| Validated Models               | [alexnet](https://docs.openvino.ai/nightly/omz_models_model_alexnet.html),                              |
|                                | [googlenet-v1](https://docs.openvino.ai/nightly/omz_models_model_googlenet_v1.html),                    |
|                                | [yolo-v3-tf](https://docs.openvino.ai/nightly/omz_models_model_yolo_v3_tf.html),                        |
|                                | [face-detection-0200](https://docs.openvino.ai/nightly/omz_models_model_face_detection_0200.html)       |
| Model Format                   | OpenVINO™ toolkit Intermediate Representation                                                           |
|                                | (\*.xml + \*.bin), ONNX (\*.onnx)                                                                       |
| Supported devices              | [All](..\..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)       |
| Other language realization     | [Python](..\..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_sample_sync_benchmark.md)  |

The following C++ API is used in the application:

| Feature                  | API                                          | Description                                  |
| -------------------------| ---------------------------------------------|----------------------------------------------|
| OpenVINO Runtime Version | ``ov::get_openvino_version``                 | Get Openvino API version.                    |
| Basic Infer Flow         | ``ov::Core``, ``ov::Core::compile_model``,   | Common API to do inference: compile a model, |
|                          | ``ov::CompiledModel::create_infer_request``, | create an infer request,                     |
|                          | ``ov::InferRequest::get_tensor``             | configure input tensors.                     |
| Synchronous Infer        | ``ov::InferRequest::infer``,                 | Do synchronous inference.                    |
| Model Operations         | ``ov::CompiledModel::inputs``                | Get inputs of a model.                       |
| Tensor Operations        | ``ov::Tensor::get_shape``,                   | Get a tensor shape and its data.             |
|                          | ``ov::Tensor::data``                         |                                              |
