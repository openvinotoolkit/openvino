# Fake SSD (C++) {#openvino_inference_engine_samples_object_detection_sample_ssd_README}

This topic demonstrates how to run the Fake sample application, which does inference using object detection
networks like SSD-VGG on Intel® Processors and Intel® HD Graphics.

> **NOTE:** This topic describes usage of C++ implementation of the Fake SSD Sample. For the Python* implementation, refer to [Fake SSD Sample](../../ie_bridges/python/sample/object_detection_sample_ssd/README.md).

| Feature    | API  | Description |
|:---     |:--- |:---
|Inference Engine Version| [`InferenceEngine::GetInferenceEngineVersion`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#aed95a031158de9bc4118bab301c3bfb6) | Get Inference Engine API version
|Available Devices|[`InferenceEngine::Core::GetAvailableDevices`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#a2173cee0e7f2522ffbc55c97d6e05ac5)| Get version information of the devices for inference
|Custom Extension Kernels|[`InferenceEngine::Core::AddExtension`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#a74c5ae4572fe9ed590fd6c0c39c68a80), [`InferenceEngine::Core::SetConfig`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#a74c5ae4572fe9ed590fd6c0c39c68a80)| Load extension library and config to the device
| Network Operations | [`InferenceEngine::CNNNetwork::setBatchSize`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#a74c5ae4572fe9ed590fd6c0c39c68a80), [`InferenceEngine::CNNNetwork::getBatchSize`](https://docs.openvinotoolkit.org/latest/namespaceInferenceEngine.html#a74c5ae4572fe9ed590fd6c0c39c68a80) |  Managing of network, operate with its batch size. Setting batch size using input image count.

The basic Inference Engine API is covered by the [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated images                 | The sample uses OpenCV\* to [read input image](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) (\*.jpg, \*.bmp, \*.png, \*.tif)
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language implementation    | [Python](../../ie_bridges/python/sample/style_transfer_sample/README.md) |

## How It Works

At startup the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates an output image and outputs data to the standard output stream.

## Supported Models

This sample application uses one of each of the following model types:

**Detection**
* `person-vehicle-bike-detection-crossroad-0078`([details](https://github.com/baychub/open_model_zoo/blob/master/models/intel/person-vehicle-bike-detection-crossroad-0078/description/person-vehicle-bike-detection-crossroad-0078.md)) is a primary detection network for finding the persons (and other objects if needed)
* `person-vehicle-bike-detection-crossroad-1016`([details](https://github.com/baychub/open_model_zoo/blob/master/models/intel/person-vehicle-bike-detection-crossroad-0078/description/person-vehicle-bike-detection-crossroad-1016.md)) is a primary detection network for finding the persons (and other objects if needed)

**Recognition** 
* `person-attributes-recognition-crossroad-0230`([details](https://github.com/baychub/open_model_zoo/blob/master/models/intel/person-attributes-recognition-crossroad-0230/description/person-attributes-recognition-crossroad-0230.md)) is executed on top of the results from the first network and reports a person's attributes like gender, has hat, or has long-sleeved clothes.

**Reidentification**
* `person-reidentification-retail-0031`([details](https://github.com/baychub/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0031/description/person-reidentification-retail-0031.md)) is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.
* `person-reidentification-retail-0287`([details](https://github.com/baychub/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/description/person-reidentification-retail-0287.md)) is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Preparing to Run

### Installing dependencies

This sample application has dependencies that may not be installed on your system. Run the following commands to install them:
```sh
cd $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/samples/python/fake_ssd_sample
pip3 install requirements.txt
```

### Downloading the models

This sample application can run with several different models. The command below downloads all of them to a structure under the ~/models directory (which you can change).

```sh
python3 $INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/downloader/downloader.py --list models.lst
```

You can also download the models individually using the --name option of the [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/blob/develop/tools/downloader/README.md).

### Converting the models

Some of the models above may need to be converted with Model Optimizer for use with OpenVINO™. The command below converts all the models that need converting.

```sh
python3 $INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/downloader/converter.py --list models.lst -o ~/models
```

You can also convert the previously downloaded models individually using the --name option of the [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/blob/develop/tools/downloader/README.md), specifying the same base directory (~/models in the command above).


### Selecting sample media

To run the sample successfully, find a still image with the following attributes:
* Aspect ratio 3:2 (width:height)
* JPEG, BMP, PNG, TIFF, or JP2
* Contains vehicles and pedestrians

### Building the sample application

If you have not yet compiled the sample applications, instructions are in the [Building the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section of the Inference Engine Samples guide.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./fake_ssd_sample -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

fake_ssd_sample [OPTION]
Options:

    -h                      Print a usage message.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -i "<path>"             Required. Path to an .bmp image.
      -l "<absolute_path>"  Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
      -c "<absolute_path>"  Required for GPU custom kernels. Absolute path to the .xml file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

For example, to do inference on a CPU with the OpenVINO&trade; toolkit person detection SSD models, run one of the following commands:

```sh
./fake_ssd_sample -m <path_to_model>/person-detection-retail-0013.xml -i <path_to_image>/inputImage.bmp -d CPU
```
or
```sh
./fake_ssd_sample -m <path_to_model>/person-detection-retail-0002.xml -i <path_to_image>/inputImage.jpg -d CPU
```

## Sample Output

The application outputs an image (`out_0.bmp`) with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

![](https://docs.openvinotoolkit.org/2018_R5/person-detection-retail-0002.png)


## See Also
* [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader_README)
* [Integrating with your application](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html)
