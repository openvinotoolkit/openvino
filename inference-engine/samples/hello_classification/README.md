# Hello Classification C++ Sample

This topic describes how to run the Hello Infer Classification sample application.
The sample is simplified version of [Image Classification Sample Async](./inference-engine/samples/classification_sample_async/README.md)
and developed with support of UNICODE.
It demonstrates how to use the following Inference Engine API in applications:
* Synchronous Infer Request API
* Input auto-resize API. It allows to set image of the original size as input for a network with other input size.
  Resize will be performed automatically by the corresponding plugin just before inference. 

There is also an API introduced to crop a ROI object and set it as input without additional memory re-allocation.
To properly demonstrate this API, it is required to run several networks in pipeline which is out of scope of this sample.
Please refer to [Security Barrier Camera Demo](./demos/security_barrier_camera_demo/README.md), or
[Crossroad Camera Demo](./demos/crossroad_camera_demo/README.md) with an example of using of new crop ROI API.

Refer to [Integrate the Inference Engine New Request API with Your Application](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

You can do inference of an image using a trained AlexNet network on a GPU using the following command:
```sh
./hello_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp GPU
```

## Sample Output

The application outputs top-10 inference results.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
