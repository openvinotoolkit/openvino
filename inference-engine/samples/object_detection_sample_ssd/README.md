# Object Detection SSD C++ Sample {#openvino_inference_engine_samples_object_detection_sample_ssd_README}

This sample demonstrates how to execute an inference of object detection networks like SSD-VGG using Synchronous Inference Request API.

Object Detection SSD C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Inference Engine Version| `InferenceEngine::GetInferenceEngineVersion` | Get Inference Engine API version
|Available Devices|`InferenceEngine::Core::GetAvailableDevices`| Get version information of the devices for inference
|Custom Extension Kernels|`InferenceEngine::Core::AddExtension`, `InferenceEngine::Core::SetConfig`| Load extension library and config to the device
| Network Operations | `InferenceEngine::CNNNetwork::getBatchSize`, `InferenceEngine::CNNNetwork::getFunction` |  Managing of network, operate with its batch size.
|nGraph Functions|`ngraph::Function::get_ops`, `ngraph::Node::get_friendly_name`, `ngraph::Node::get_type_info`| Go thru network nGraph

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [person-detection-retail-0013](@ref omz_models_model_person_detection_retail_0013)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | The sample uses OpenCV\* to [read input image](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) (\*.bmp, \*.png)
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [C](../../ie_bridges/c/samples/object_detection_sample_ssd/README.md), [Python](../../ie_bridges/python/sample/object_detection_sample_ssd/README.md) |

## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network and image to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image and output data to the standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

Running the application with the `-h` option yields the following usage message:

```
./object_detection_sample_ssd -h
InferenceEngine:
    API version ............ <version>
    Build .................. <build>
    Description ....... API

object_detection_sample_ssd [OPTION]
Options:

    -h                      Print a usage message.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -i "<path>"             Required. Path to an image.
      -l "<absolute_path>"  Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
      -c "<absolute_path>"  Required for GPU, MYRIAD, HDDL custom kernels. Absolute path to the .xml config file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified.

    Available target devices: <devices>

```

Running the application with the empty list of options yields the usage message given above and an error message.

> **NOTES**:
>
> - By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader_README):
```
python <path_to_omz_tools>/downloader.py --name person-detection-retail-0013
```

2. `person-detection-retail-0013` model does not need to be converted, because it is already in necessary format, so you can skip this step. If you want to use a other model that is not in the Inference Engine IR or ONNX format, you can convert it using the model converter script:

```
python <path_to_omz_tools>/converter.py --name <model_name>
```

3. For example, to do inference on a CPU with the OpenVINO&trade; toolkit person detection SSD models, run one of the following commands:

- with one image and [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model

```
<path_to_sample>/object_detection_sample_ssd -m <path_to_model>/person-detection-retail-0013.xml -i <path_to_image>/person_detection.png -d CPU
```

- with one image and [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0002_description_person_detection_retail_0002.html) model

```
<path_to_sample>/object_detection_sample_ssd -m <path_to_model>/person-detection-retail-0002.xml -i <path_to_image>/person_detection.png -d GPU
```

## Sample Output

The application outputs an image (`out_0.bmp`) with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

```
object_detection_sample_ssd -m person-detection-retail-0013\FP16\person-detection-retail-0013.xml -i person_detection.png
[ INFO ] InferenceEngine:
        API version ............ <version>
        Build .................. <build>
        Description ....... API
Parsing input parameters
[ INFO ] Files were added: 1
[ INFO ]     person_detection.png
[ INFO ] Loading Inference Engine
[ INFO ] Device info:
        CPU
        MKLDNNPlugin version ......... <version>
        Build ........... <build>
[ INFO ] Loading network files:
        person-detection-retail-0013\FP16\person-detection-retail-0013.xml
[ INFO ] Preparing input blobs
[ INFO ] Batch size is 1
[ INFO ] Preparing output blobs
[ INFO ] Loading model to the device
[ INFO ] Create infer request
[ WARNING ] Image is resized from (1699, 960) to (544, 320)
[ INFO ] Batch size is 1
[ INFO ] Start inference
[ INFO ] Processing output blobs
[0,1] element, prob = 0.99909    (370,201)-(634,762) batch id : 0 WILL BE PRINTED!
[1,1] element, prob = 0.997386    (836,192)-(999,663) batch id : 0 WILL BE PRINTED!
[2,1] element, prob = 0.314753    (192,2)-(265,172) batch id : 0
...
[ INFO ] Image out_0.bmp created!
[ INFO ] Execution successful

[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
