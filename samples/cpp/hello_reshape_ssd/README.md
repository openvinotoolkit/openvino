# Hello Reshape SSD C++ Sample {#openvino_inference_engine_samples_hello_reshape_ssd_README}

This sample demonstrates how to do synchronous inference of object detection models, using [input reshape feature](../../../docs/OV_Runtime_UG/ShapeInference.md).
Models with only 1 input and output are supported.

The following C++ API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| Node operations | `ov::Node::get_type_info`, `ngraph::op::DetectionOutput::get_type_info_static`, `ov::Output::get_any_name`, `ov::Output::get_shape` | Get a node info |
| Model Operations | `ov::Model::get_ops`, `ov::Model::reshape` | Get model nodes, reshape input |
| Tensor Operations | `ov::Tensor::data` | Get a tensor data |
| Preprocessing | `ov::preprocess::PreProcessSteps::convert_element_type`, `ov::preprocess::PreProcessSteps::convert_layout` | Model input preprocessing |

Basic OpenVINO™ Runtime API is described in [Hello Classification C++ sample](../hello_classification/README.md).

| Options | Values |
| :--- | :--- |
| Validated Models | [person-detection-retail-0013](@ref omz_models_model_person_detection_retail_0013) |
| Model Format | OpenVINO Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Supported devices | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [Python](../../../samples/python/hello_reshape_ssd/README.md) |

## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network and image to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image and output data to the standard output stream.

For more information, refer to the explicit description of [Integration Steps](../../../docs/OV_Runtime_UG/integrate_with_your_application.md).

## Building

To build the sample, use the instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO Toolkit Samples.

## Running

Before running the sample, specify the model and the image:

- you may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded by using the [Model Downloader](@ref omz_tools_downloader).
- you may use images from the media files collection, available online in [test-data](https://storage.openvinotoolkit.org/data/test_data) storage.

To run the sample, use the following script:

```
hello_reshape_ssd <path_to_model> <path_to_image> <device>
```

> **NOTES**:
> - By default, OpenVINO Toolkit Samples and Demos expect input with `BGR` order of channels. If you trained your model to work with `RGB` order, you need to manually rearrange the default order of channels in the sample or demo application, or reconvert your model, using Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

### Example

1. Install openvino-dev Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model, using:

```
omz_downloader --name person-detection-retail-0013
```

3. `person-detection-retail-0013` does not need to be converted, because it is already in an appropriate format, so you can skip this step. If you want to use another model that is not in the IR or ONNX format, you can convert it, using the Model Converter:

```
omz_converter --name <model_name>
```

4. Perform inference of `person_detection.bmp`, using `person-detection-retail-0013` model on a `GPU`, for example:

```
hello_reshape_ssd person-detection-retail-0013.xml person_detection.bmp GPU
```

## Sample Output

The application renders an image with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>
[ INFO ]
[ INFO ] Loading model files: \models\person-detection-retail-0013.xml
[ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 320, 544}
[ INFO ]     outputs
[ INFO ]         output name: detection_out
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1, 200, 7}
Reshape network to the image size = [960x1699]
[ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 960, 1699}
[ INFO ]     outputs
[ INFO ]         output name: detection_out
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1, 200, 7}
[0,1] element, prob = 0.716309,    (852,187)-(983,520)
The resulting image was saved in the file: hello_reshape_ssd_output.bmp

This sample is an API example, for any performance measurements use the dedicated benchmark_app tool
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)