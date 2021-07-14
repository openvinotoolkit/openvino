# Hello NV12 Input Classification C++ Sample {#openvino_inference_engine_samples_hello_nv12_input_classification_README}

This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API and [input reshape feature](../../../docs/IE_DG/ShapeInference.md).

Hello NV12 Input Classification C++ Sample demonstrates how to use the NV12 automatic input pre-processing API of the Inference Engine in your applications:

| Feature    | API  | Description |
|:---        |:---  |:---
|Inference Engine Core Operations| `InferenceEngine::Core::GetMetric` | Gets general runtime metric for dedicated hardware
| Blob Operations | `InferenceEngine::NV12Blob` | Create NV12Blob to hold the NV12 input data
| Input in N12 color format | `InferenceEngine::PreProcessInfo::setColorFormat` | Change the color format of the input data
| Model Input Reshape | `InferenceEngine::CNNNetwork::getInputShapes`, `InferenceEngine::CNNNetwork::reshape`, `InferenceEngine::CNNNetwork::getBatchSize` | Set the batch size equal to the number of input images

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [alexnet](@ref omz_models_model_alexnet)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | An uncompressed image in the NV12 color format - \*.yuv
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [C](../../ie_bridges/c/samples/hello_nv12_input_classification/README.md) |

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads specified network and an
image in the NV12 color format to an Inference Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream. You can place labels in .labels file near the model to get pretty output.

You can see the explicit description of each sample step at [Integration Steps](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.


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
> - Because the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> - By default, this sample expects that network input has BGR channels order. If you trained your
>   model to work with RGB order, you need to reconvert your model using the Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader_README):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Perform inference of NV12 image using `alexnet` model on a `CPU`, for example:

```
<path_to_sample>/hello_nv12_input_classification <path_to_model>/alexnet.xml <path_to_image>/cat.yuv 300x300 CPU
```

## Sample Output

The application outputs top-10 inference results.

```
[ INFO ] Files were added: 1
[ INFO ]     ./cat.yuv
Batch size is 1

Top 10 results:

Image ./cat.yuv

classid probability
------- -----------
435     0.0917327  
876     0.0817254  
999     0.0693054  
587     0.0437265  
666     0.0389570  
419     0.0328923  
285     0.0303094  
700     0.0299405  
696     0.0216280  
855     0.0203389  

This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
