# Two Simple Demos implemented by OpenVINO C API

The tutorials provide the basic introduction to OpenVINO™ shows how to do inference with an image classification model.

- [Hello Classification C Sample](../../../../samples/c/hello_classification/README.md) Inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs and supports Unicode paths).
- [Hello NV12 Input Classification C Sample](../../../../samples/c/hello_nv12_input_classification/README.md) Input of any size and layout can be provided to an infer request. The sample transforms the input to the NV12 color format and pre-process it automatically during inference. The sample supports only images as inputs.
