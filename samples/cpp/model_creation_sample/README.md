# Model Creation C++ Sample

This sample demonstrates how to execute an synchronous inference using [model](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html) built on the fly which uses weights from LeNet classification model, which is known to work well on digit classification tasks.

You do not need an XML file to create a model. The API of ov::Model allows creating a model on the fly from the source code.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/model-creation.html)

## Requirements

| Options                     | Values                                                                                                                         |
| ----------------------------| -------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | LeNet                                                                                                                          |
| Model Format                | model weights file (\*.bin)                                                                                                    |
| Validated images            | single-channel ``MNIST ubyte`` images                                                                                          |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                            |
| Other language realization  | [Python](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/model-creation.html)                                                  |

The following C++ API is used in the application:

| Feature                   | API                                     | Description                           |
| --------------------------| ----------------------------------------|---------------------------------------|
| OpenVINO Runtime Info     | ``ov::Core::get_versions``              | Get device plugins versions           |
| Shape Operations          | ``ov::Output::get_shape``,              | Operate with shape                    |
|                           | ``ov::Shape::size``,                    |                                       |
|                           | ``ov::shape_size``                      |                                       |
| Tensor Operations         | ``ov::Tensor::get_byte_size``,          | Get tensor byte size and its data     |
|                           | ``ov::Tensor:data``                     |                                       |
| Model Operations          | ``ov::set_batch``                       | Operate with model batch size         |
| Infer Request Operations  | ``ov::InferRequest::get_input_tensor``  | Get a input tensor                    |
| Model creation objects    | ``ov::opset8::Parameter``,              | Used to construct an OpenVINO model   |
|                           | ``ov::Node::output``,                   |                                       |
|                           | ``ov::opset8::Constant``,               |                                       |
|                           | ``ov::opset8::Convolution``,            |                                       |
|                           | ``ov::opset8::Add``,                    |                                       |
|                           | ``ov::opset1::MaxPool``,                |                                       |
|                           | ``ov::opset8::Reshape``,                |                                       |
|                           | ``ov::opset8::MatMul``,                 |                                       |
|                           | ``ov::opset8::Relu``,                   |                                       |
|                           | ``ov::opset8::Softmax``,                |                                       |
|                           | ``ov::descriptor::Tensor::set_names``,  |                                       |
|                           | ``ov::opset8::Result``,                 |                                       |
|                           | ``ov::Model``,                          |                                       |
|                           | ``ov::ParameterVector::vector``         |                                       |

Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).
