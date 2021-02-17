# Object Detection C Sample SSD {#openvino_inference_engine_ie_bridges_c_samples_object_detection_sample_ssd_README}

This topic demonstrates how to run the Object Detection C sample application, which does inference using object detection
networks like SSD-VGG on Intel® Processors and Intel® HD Graphics.

> **NOTE:** This topic describes usage of C implementation of the Object Detection Sample SSD. For the C++* implementation, refer to [Object Detection C++* Sample SSD](../../../../samples/object_detection_sample_ssd/README.md) and for the Python* implementation, refer to [Object Detection Python* Sample SSD](../../../python/sample/object_detection_sample_ssd/README.md).

## How It Works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates output images and outputs data to the standard output stream.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

> **NOTE**: This sample uses `ie_network_reshape()` to set the batch size. While supported by SSD networks, reshape may not work with arbitrary topologies. See [Shape Inference Guide](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_ShapeInference.html) for more info.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```sh
./object_detection_sample_ssd_c -h
[ INFO ] InferenceEngine:
<version><number>
[ INFO ] Parsing input parameters

object_detection_sample_ssd_c [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to one or more .bmp images.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"  Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
      -c "<absolute_path>"  Required for GPU custom kernels. Absolute path to the .xml file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified
    -g                      Path to the configuration file. Default value: "config".
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

For example, to do inference on a CPU with the OpenVINO&trade; toolkit person detection SSD models, run one of the following commands:

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage.bmp -m <path_to_model>person-detection-retail-0013.xml -d CPU
```

or

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage1.bmp <path_to_image>/inputImage2.bmp ... -m <path_to_model>person-detection-retail-0013.xml -d CPU
```

or

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage.jpg -m <path_to_model>person-detection-retail-0002.xml -d CPU
```

## Sample Output

The application outputs several images (`out_0.bmp`, `out_1.bmp`, ... ) with detected objects enclosed in rectangles. It outputs the list of
classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.


## See Also
* [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader_README)
