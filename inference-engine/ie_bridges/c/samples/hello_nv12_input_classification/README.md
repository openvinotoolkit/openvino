# Hello NV12 Input Classification C Sample {#openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README}

Inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

Hello NV12 Input Classification C Sample demonstrates how to use the NV12 automatic input pre-processing API of the Inference Engine in your applications:

| Feature    | API  | Description |
|:---     |:--- |:---
| Blob Operations| [ie_blob_make_memory_nv12] | Create a NV12 blob
| Input in N12 color format |[ie_network_set_color_format]| Change the color format of the input data
Basic Inference Engine API is covered by [Hello Classification C sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | AlexNet (image classification network)
| Model Format                     | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)
| Supported devices                | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [C++](../../../../samples/hello_nv12_input_classification/README.md) |


This topic describes how to run the Hello NV12 Input Classification sample application.
The sample demonstrates how to use the new NV12 automatic input pre-processing API of the Inference Engine in your applications.
Refer to [Integrate the Inference Engine New Request API with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads specified network and an
image in the NV12 color format to an Inference Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the
application outputs data to the standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:
 - you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
 - you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

The sample accepts an uncompressed image in the NV12 color format. To run the sample, you need to
convert your BGR/RGB image to NV12. To do this, you can use one of the widely available tools such
as FFmpeg\* or GStreamer\*. The following command shows how to convert an ordinary image into an
uncompressed NV12 image using FFmpeg:
```sh
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

> **NOTES**:
>
> * Because the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> * By default, this sample expects that network input has BGR channels order. If you trained your
>   model to work with RGB order, you need to reconvert your model using the Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
> * Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> * The sample accepts models in ONNX format (.onnx) that do not require preprocessing.


You can perform inference on an NV12 image using a trained AlexNet network on a CPU with the following command:
```sh
./hello_nv12_input_classification_c <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.yuv 640x480 CPU
```

## Sample Output

The application outputs top-10 inference results.

[//]: # (TODO: insert output)

## See Also
* [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
* [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
* [Model Downloader](@ref omz_tools_downloader_README)
* [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[ie_network_set_color_format]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#ga85f3251f1f7b08507c297e73baa58969
[ie_blob_make_memory_nv12]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Blob.html#ga0a2d97b0d40a53c01ead771f82ae7f4a
