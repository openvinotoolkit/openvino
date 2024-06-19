# Image Classification Async C++ Sample 

This sample demonstrates how to do inference of image classification models using Asynchronous Inference Request API. 
 
Models with only one input and output are supported.

In addition to regular images, the sample also supports single-channel ``ubyte`` images as an input for LeNet model.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.3/openvino_sample_image_classification_async.html)

## Requirements

| Options                    | Values                                                                                                                               |
| ---------------------------| -------------------------------------------------------------------------------------------------------------------------------------| 
| Validated Models           | [alexnet](https://docs.openvino.ai/2023.3/omz_models_model_alexnet.html),                                                            |
|                            | [googlenet-v1](https://docs.openvino.ai/2023.3/omz_models_model_googlenet_v1.html)                                                   |
| Model Format               | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                      |
| Supported devices          | [All](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)                                  |
| Other language realization | [Python](https://docs.openvino.ai/2023.3/openvino_sample_image_classification_async.html)                                            |

The following C++ API is used in the application:

| Feature                  | API                                                                   | Description                                                                            |
| -------------------------| ----------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Asynchronous Infer       | ``ov::InferRequest::start_async``, ``ov::InferRequest::set_callback`` | Do asynchronous inference with callback.                                               |
| Model Operations         | ``ov::Output::get_shape``, ``ov::set_batch``                          | Manage the model, operate with its batch size. Set batch size using input image count. |
| Infer Request Operations | ``ov::InferRequest::get_input_tensor``                                | Get an input tensor.                                                                   |
| Tensor Operations        | ``ov::shape_size``, ``ov::Tensor::data``                              | Get a tensor shape size and its data.                                                  |

