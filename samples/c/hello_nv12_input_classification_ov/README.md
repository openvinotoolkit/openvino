# Hello NV12 Input Classification C Sample for OpenVINO™ 2.0 C-API

This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

## How It Works

At startup, the sample application reads command-line parameters, loads specified network and an
image in the `NV12` color format to OpenVINO Runtime. Then, the sample creates a synchronous inference request object. When inference is done, the
application outputs data to the standard output stream.

For more information, refer to the ["Integrate OpenVINO Runtime with Your Application" Guide](../../../docs/OV_Runtime_UG/integrate_with_your_application.md).

## Building

To build the sample, use the instructions available in the **Build the Sample Applications** section in [OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md).

## Running

To run the sample, you need to specify a model and an image:

- You may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
- You may use images from the media files collection, available online in the [test data](https://storage.openvinotoolkit.org/data/test_data).

The sample accepts an uncompressed image in the `NV12` color format. To run the sample, you need to
convert your `BGR`/`RGB` image to `NV12`. To do this, you can use one of the widely available tools such
as FFmpeg or GStreamer. The following command shows how to convert an ordinary image into an
uncompressed `NV12` image, using FFmpeg:

```sh
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

> **NOTES**:
>
> - Since the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding `NV12` logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> - By default, this sample expects that network input has the `BGR` channel order. If you trained your
>   model to work with `RGB`, you need to reconvert your model, using Model Optimizer 
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/Additional_Optimizations.md).
> - Before running the sample with a trained model, make sure the model is converted to the OpenVINO Intermediate Representation (\*.xml + \*.bin) by using [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in the ONNX format (\*.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model, using [Model Downloader](@ref omz_tools_downloader):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the OpenVINO IR or ONNX format, it must be converted with Model Converter:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Do inference of `NV12` image, using the `alexnet` model on `CPU`, for example:

```
<path_to_sample>/hello_nv12_input_classification_ov_c <path_to_model>/alexnet.xml <path_to_image>/cat.yuv 300x300 CPU
```

## Sample Output

The application outputs top-10 inference results.

```
Top 10 results:

Image <path_to_image>/cat.yuv

classid probability
------- -----------
876       0.125426
435       0.120252
285       0.068099
282       0.056738
281       0.032151
36       0.027748
94       0.027691
999       0.026507
335       0.021384
186       0.017978

This sample is an API example. For any performance measurements, use the dedicated `benchmark_app` tool.
```