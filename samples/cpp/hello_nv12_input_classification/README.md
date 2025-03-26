# Hello NV12 Input Classification C++ Sample

This sample demonstrates how to execute an inference of image classification models with images in NV12 color format using Synchronous Inference Request API.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-nv12-input-classification.html)

## Requirements

| Options                     | Values                                                                                                                          |
| ----------------------------| --------------------------------------------------------------------------------------------------------------------------------|
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                 |
| Validated images            | An uncompressed image in the NV12 color format - \*.yuv                                                                         |
| Supported devices           | [All](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)                             |
| Other language realization  | [C](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-nv12-input-classification.html)                                       |


The following C++ API is used in the application:

| Feature                  | API                                                         | Description                               |
| -------------------------| ------------------------------------------------------------|-------------------------------------------|
| Node Operations          | ``ov::Output::get_any_name``                                | Get a layer name                          |
| Infer Request Operations | ``ov::InferRequest::set_tensor``,                           | Operate with tensors                      |
|                          | ``ov::InferRequest::get_tensor``                            |                                           |
| Preprocessing            | ``ov::preprocess::InputTensorInfo::set_color_format``,      | Change the color format of the input data |
|                          | ``ov::preprocess::PreProcessSteps::convert_element_type``,  |                                           |
|                          | ``ov::preprocess::PreProcessSteps::convert_color``          |                                           |


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/hello-classification.html).

