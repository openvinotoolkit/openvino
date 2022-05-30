# Hello NV12 Input Classification C Sample {#openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README}

This sample demonstrates how to execute an inference of image classification networks such as AlexNet with images in `NV12` color format using Synchronous Inference Request API.

Hello NV12 Input Classification C Sample demonstrates how to use the NV12 automatic input pre-processing API of the Inference Engine in your applications:

| Feature    | API  | Description |
|:---     |:--- |:---
| Blob Operations| [ie_blob_make_memory_nv12] | Create a `NV12` blob.
| Input in `N12` color format |[ie_network_set_color_format]| Change the color format of the input data.
Basic Inference Engine API is described in [Hello Classification C sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [alexnet](@ref omz_models_model_alexnet)
| Model Format                     | OpenVINO Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | An uncompressed image in the `NV12` color format - \*.yuv
| Supported devices                | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [C++](../../../samples/cpp/hello_nv12_input_classification/README.md) |

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads specified network and an
image in the `NV12` color format to the Inference Engine plugin. Then, the sample creates a synchronous inference request object. When inference is done, the
application outputs data to the standard output stream.

For more information, refer to the explicit description of
each sample **Integration Step** in the [Integrate OpenVINO Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md) guide.

## Building

To build the sample, use the instructions available in the **Build the Sample Applications** section in [OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md).

## Running

To run the sample, you need to specify a model and an image:

- you may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded by using the [Model Downloader](@ref omz_tools_downloader).
- you may use images from the media files collection, available online in the [test data storage](https://storage.openvinotoolkit.org/data/test_data).

The sample accepts an uncompressed image in the `NV12` color format. To run the sample, you need to
convert your `BGR`/`RGB` image to `NV12`. To do this, you may use one of the widely available tools such
as FFmpeg or GStreamer. The following command shows how to convert an ordinary image into an
uncompressed `NV12` image, using FFmpeg:

```sh
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

> **NOTES**:
> - Since the sample reads raw image files, a correct image size along with the image path 
>   should be provided. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image, the corresponding `NV12` logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> - By default, this sample expects that network input has `BGR` order of channels. If you trained your
>   model to work with the `RGB` order, you need to reconvert your model, using Model Optimizer 
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to the **When to Reverse Input Channels** section of
>   [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
> - Before running the sample with a trained model, make sure that the model is converted to the OpenVINO Intermediate Representation (OpenVINO IR) format (\*.xml + \*.bin) by using [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in the ONNX format (.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model, using [Model Downloader](@ref omz_tools_downloader):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the OpenVINO IR or ONNX format, it must be converted with Model Converter:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Perform inference of the `NV12` image, using the `alexnet` model on a `CPU`, for example:

```
<path_to_sample>/hello_nv12_input_classification_c <path_to_model>/alexnet.xml <path_to_image>/cat.yuv 300x300 CPU
```

## Sample Output

The application outputs top-10 inference results.

```
Top 10 results:

Image ./cat.yuv

classid probability
------- -----------
435       0.091733
876       0.081725
999       0.069305
587       0.043726
666       0.038957
419       0.032892
285       0.030309
700       0.029941
696       0.021628
855       0.020339

This sample is an API example. Use the dedicated `benchmark_app` tool for any performance measurements.
```

## See Also

- [Integrate the OpenVINO into Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [OpenVINO Toolkit Test Data Storage](https://storage.openvinotoolkit.org/data/test_data).

[ie_network_set_color_format]:https://docs.openvino.ai/latest/ie_c_api/group__Network.html#ga85f3251f1f7b08507c297e73baa58969
[ie_blob_make_memory_nv12]:https://docs.openvino.ai/latest/ie_c_api/group__Blob.html#ga0a2d97b0d40a53c01ead771f82ae7f4a