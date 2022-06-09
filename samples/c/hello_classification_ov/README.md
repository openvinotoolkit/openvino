# Hello Classification C Sample for OpenVINO 2.0 C-API

This sample demonstrates how to execute an inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API and input auto-resize feature.

## How It Works

Upon the start-up, the sample application reads command line parameters, loads specified network and an image to the OpenVINO plugin.
Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Perform inference of `car.bmp` using `alexnet` model on a `GPU`, for example:

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

This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```
