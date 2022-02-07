# Hello Reshape SSD C++ Sample {#openvino_inference_engine_samples_hello_reshape_ssd_README}

This sample demonstrates how to execute an inference of object detection networks like SSD-VGG using Synchronous Inference Request API, [input reshape feature](../../../docs/OV_Runtime_UG/ShapeInference.md).

Hello Reshape SSD C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
| Model Operations | `ov::Core::read_model`, `ov::Core::compile_model` |  Managing of model
| Model Input Reshape | `ov::Model::reshape`| Resize model to match image sizes and given batch
| Tensor Operations | `ov::Tensor::get_element_type`, `ov::Tensor::get_shape`, `ov::Tensor::data` | Work with storing inputs, outputs of the model, weights and biases of the layers

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [person-detection-retail-0013](@ref omz_models_model_person_detection_retail_0013)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | The sample uses OpenCV\* to [read input image](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) (\*.bmp, \*.png)
| Supported devices                | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [Python](../../../samples/python/hello_reshape_ssd/README.md) |

## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network and image to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image and output data to the standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/OV_Runtime_UG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

You can use the following command to do inference on CPU of an image using a trained SSD network:

```
<path_to_sample>/hello_reshape_ssd <path_to_model> <path_to_image> <device> <batch>
```

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader):
```
python <path_to_omz_tools>/downloader.py --name person-detection-retail-0013
```

2. `person-detection-retail-0013` model does not need to be converted, because it is already in necessary format, so you can skip this step. If you want to use a other model that is not in the Inference Engine IR or ONNX format, you can convert it using the model converter script:

```
python <path_to_omz_tools>/converter.py --name <model_name>
```

3. Perform inference of `person_detection.png` using `person-detection-retail-0013` model on a `GPU`, for example:

```
<path_to_sample>/hello_reshape_ssd <path_to_model>/person-detection-retail-0013.xml <path_to_image>/person_detection.png GPU 1
```

## Sample Output

The application renders an image with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

```
[ INFO ] Loading model files: C:\temp\models\public\ssd_mobilenet_v1_fpn_coco\FP16\ssd_mobilenet_v1_fpn_coco.xml
[ INFO ] model name: ssd_mobilenet_v1_fpn_coco
[ INFO ]     inputs
[ INFO ]         input name: image_tensor
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 640, 640}
[ INFO ]     outputs
[ INFO ]         output name: DetectionOutput
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1, 100, 7}
Reshape network to the image size = [512x512] with batch = 1
[ INFO ] model name: ssd_mobilenet_v1_fpn_coco
[ INFO ]     inputs
[ INFO ]         input name: image_tensor
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 512, 512}
[ INFO ]     outputs
[ INFO ]         output name: DetectionOutput
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1, 100, 7}
[0,18] element, prob = 0.781129    (109,52)-(342,441) batch id = 0
The resulting image was saved in the file: hello_reshape_ssd_batch_0.bmp
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/OV_Runtime_UG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
