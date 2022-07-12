# Hello Classification C Sample for OpenVINO™ 2.0 C-API

This sample demonstrates how to execute an inference of image classification networks like AlexNet and GoogLeNet, using Synchronous Inference Request API and input auto-resize feature.

## How It Works

At startup, the sample application reads command-line parameters, loads specified network and an image to the OpenVINO plugin.
Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream.

## Building

To build the sample, use the instructions available in the **Build the Sample Applications** section in [OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md).

## Running

To run the sample, you need to specify a model and an image:

- You may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded using [Model Downloader](@ref omz_tools_downloader).
- You may use images from the media files collection, available online in the [test data](https://storage.openvinotoolkit.org/data/test_data).

> **NOTES**:
>
> - By default, OpenVINO Toolkit samples and demos expect input with the `BGR` channel order. If you trained your model to work with `RGB`, you need to manually rearrange the default channel order in the sample or demo application, or reconvert your model, using Model Optimizer with `--reverse_input_channels` argument specified. For more information about the argument, refer to the **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/Additional_Optimizations.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the OpenVINO Intermediate Representation (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
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

3. Do inference of the `car.bmp` image, using the `alexnet` model on `GPU`, for example:

```
<path_to_sample>/hello_classification_c <path_to_model>/alexnet.xml <path_to_image>/car.bmp GPU
```

## Sample Output

The application outputs top-10 inference results.

```
Top 10 results:

Image /opt/intel/openvino/samples/scripts/car.png

classid probability
------- -----------
656       0.666479
654       0.112940
581       0.068487
874       0.033385
436       0.026132
817       0.016731
675       0.010980
511       0.010592
569       0.008178
717       0.006336

This sample is an API example, for any performance measurements, use the dedicated `benchmark_app` tool.
```
