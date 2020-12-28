# Hello Classification Batch C Sample {#openvino_inference_engine_ie_bridges_c_samples_hello_classification_batch_README}

This topic describes how to run the Hello Classification Batch C sample application.

It demonstrates how to use the following Inference Engine C API in applications:
* Synchronous Infer Request API
* API for setting custom batch size for Executable Network. It allows to perform inference on multiple images in one inference run.
* API for setting custom batch size for Infer Request. It allows to reuse Infer Request with input blobs containing arbitrary number of input images.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](@ref omz_tools_downloader_README) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

You can do inference of an image using a trained AlexNet network on a GPU using the following command:

```sh
./hello_classification_batch_c <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp <path_to_image>/dog.bmp ... <path_to_image>/bird.bmp GPU
```

## Sample Output

The application outputs top-10 inference results for each input image.
