# Hello NV12 Input Classification C Sample for OpenVINO 2.0 C-API

This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads specified network and an
image in the NV12 color format to an Inference Engine plugin. Then, the sample creates a synchronous inference request object. When inference is done, the
application outputs data to the standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
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
> - Because the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> - By default, this sample expects that network input has BGR channels order. If you trained your
>   model to work with RGB order, you need to reconvert your model using the Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Perform inference of NV12 image using `alexnet` model on a `CPU`, for example:

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

This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```