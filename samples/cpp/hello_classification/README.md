# Hello Classification C++ Sample

This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API.

Models with only one input and output are supported.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html)

## Requirements

| Options                     | Values                                                                                                                        |
| ----------------------------| ------------------------------------------------------------------------------------------------------------------------------|
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                               |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                           |
| Other language realization  | [Python, C](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html),         |

The following C++ API is used in the application:

| Feature                   | API                                                            | Description                                                                            |
| --------------------------| ---------------------------------------------------------------|----------------------------------------------------------------------------------------|
| OpenVINO Runtime Version  | ``ov::get_openvino_version``                                   | Get Openvino API version                                                               |
| Basic Infer Flow          | ``ov::Core::read_model``,                                      | Common API to do inference: read and compile a model, create an infer request,         |
|                           | ``ov::Core::compile_model``,                                   | configure input and output tensors                                                     |
|                           | ``ov::CompiledModel::create_infer_request``,                   |                                                                                        |
|                           | ``ov::InferRequest::set_input_tensor``,                        |                                                                                        |
|                           | ``ov::InferRequest::get_output_tensor``                        |                                                                                        |
| Synchronous Infer         | ``ov::InferRequest::infer``                                    | Do synchronous inference                                                               |
| Model Operations          | ``ov::Model::inputs``,                                         | Get inputs and outputs of a model                                                      |
|                           | ``ov::Model::outputs``                                         |                                                                                        |
| Tensor Operations         | ``ov::Tensor::get_shape``                                      | Get a tensor shape                                                                     |
| Preprocessing             | ``ov::preprocess::InputTensorInfo::set_element_type``,         | Set image of the original size as input for a model with other input size. Resize      |
|                           | ``ov::preprocess::InputTensorInfo::set_layout``,               | and layout conversions are performed automatically by the corresponding plugin         |
|                           | ``ov::preprocess::InputTensorInfo::set_spatial_static_shape``, | just before inference.                                                                 |
|                           | ``ov::preprocess::PreProcessSteps::resize``,                   |                                                                                        |
|                           | ``ov::preprocess::InputModelInfo::set_layout``,                |                                                                                        |
|                           | ``ov::preprocess::OutputTensorInfo::set_element_type``,        |                                                                                        |
|                           | ``ov::preprocess::PrePostProcessor::build``                    |                                                                                        |
