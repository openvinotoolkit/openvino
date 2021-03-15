# Object Detection C Sample SSD {#openvino_inference_engine_ie_bridges_c_samples_object_detection_sample_ssd_README}

Inference of object detection networks like SSD-VGG using Asynchronous Inference Request API and [input reshape feature](../../../../../docs/IE_DG/ShapeInference.md).

Object Detection C sample SSD application demonstrates how to use the following Inference Engine C API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Asynchronous Infer |[ie_infer_request_infer_async][ie_infer_request_wait]| Do Asynchronous inference
|Inference Engine Version| [ie_c_api_version] | Get Inference Engine API version
|Available Devices| [ie_core_get_versions] | Get version information of the devices for inference
|Custom Extension Kernels|[ie_core_add_extension] [ie_core_set_config]| Load extension library and config to the device
|Network Operations|[ie_network_get_inputs_number] [ie_network_get_input_dims] [ie_network_get_input_shapes] [ie_network_get_outputs_number] [ie_network_get_output_dims]| Managing of network
|Blob Operations|[ie_blob_get_buffer]| Work with memory container for storing inputs, outputs of the network, weights and biases of the layers
|Input Reshape|[ie_network_reshape]| Set the batch size equal to the number of input images

Basic Inference Engine API is covered by [Hello Classification C sample](../hello_classification/README.md).

> **NOTE**: This sample uses `ie_network_reshape()` to set the batch size. While supported by SSD networks, reshape may not work with arbitrary topologies. See [Shape Inference Guide](../../../../../docs/IE_DG/ShapeInference.md) for more info.

| Options  | Values |
|:---                              |:---
| Validated Models                 | Person detection SSD (object detection network)
| Model Format                     | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)
| Supported devices                | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [C++](../../../../samples/object_detection_sample_ssd/README.md), [Python](../../../python/sample/object_detection_sample_ssd/README.md) |


## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network and image(s) to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image(s) and output data to the standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:
 - you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
 - you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

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
      -l "<absolute_path>"  Required for device plugin custom layers. Absolute path to a shared library with the kernels implementations.
          And
      -c "<absolute_path>"  Required for GPU, MYRIAD, HDDL custom kernels. Absolute path to the .xml file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified
    -g                      Path to the configuration file. Default value: "config".
```

Running the application with the empty list of options yields the usage message given above and an error message.

> **NOTES**:
>
> * By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
>
> * Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> * The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

For example, to do inference on a CPU with the OpenVINO&trade; toolkit person detection SSD models, run one of the following commands:
- with one image and [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage.bmp -m <path_to_model>person-detection-retail-0013.xml -d CPU
```
- with some images and [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage1.bmp <path_to_image>/inputImage2.bmp ... -m <path_to_model>person-detection-retail-0013.xml -d CPU
```

- with [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0002_description_person_detection_retail_0002.html) model

```sh
./object_detection_sample_ssd_c -i <path_to_image>/inputImage.jpg -m <path_to_model>person-detection-retail-0002.xml -d CPU
```

## Sample Output

The application outputs several images (`out_0.bmp`, `out_1.bmp`, ... ) with detected objects enclosed in rectangles. It outputs the list of
classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.

[//]: # (TODO: insert output)

## See Also
* [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
* [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
* [Model Downloader](@ref omz_tools_downloader_README)
* [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[ie_infer_request_infer_async]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__InferRequest.html#gad2351010e292b6faec959a3d5a8fb60e
[ie_infer_request_wait]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__InferRequest.html#ga0c05e63e63c8d9cdd92900e82b0137c9
[ie_c_api_version]:https://docs.openvinotoolkit.org/latest/ie_c_api/ie__c__api_8h.html#a8fe3efe9cc606dcc7bec203102043e68
[ie_core_get_versions]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Core.html#ga2932e188a690393f5d594572ac5d237b
[ie_core_add_extension]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Core.html#gadded2444ba81d2d396516b72c2478f8e
[ie_core_set_config]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Core.html#gaf09d1e77cc264067e4e22ddf99f21ec1
[ie_network_get_inputs_number]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#ga6a3349bca66c4ba8b41a434061fccf52
[ie_network_get_input_dims]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#gac621a654b89d413041cbc2288627f6a5
[ie_network_get_input_shapes]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#ga5409734f25ffbb1379e876217c0bc6f3
[ie_network_get_outputs_number]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#ga869b8c309797f1e09f73ddffd1b57509
[ie_network_get_output_dims]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#ga8de7bf2f626f19eba08a2f043fc1b5d2
[ie_network_reshape]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Network.html#gac4f690afd0c2221f7db2ff9be4aa0637
[ie_blob_get_buffer]:https://docs.openvinotoolkit.org/latest/ie_c_api/group__Blob.html#ga948e0186cea6a393c113d5c399cfcb4c
