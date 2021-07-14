# Style Transfer C++ Sample {#openvino_inference_engine_samples_style_transfer_sample_README}

This sample demonstrates how to execute an inference of style transfer models using Synchronous Inference Request API.

Style Transfer C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Inference Engine Version| `InferenceEngine::GetInferenceEngineVersion` | Get Inference Engine API version
|Available Devices|`InferenceEngine::Core::GetAvailableDevices`| Get version information of the devices for inference
|Custom Extension Kernels|`InferenceEngine::Core::AddExtension`, `InferenceEngine::Core::SetConfig`| Load extension library and config to the device
| Network Operations | `InferenceEngine::CNNNetwork::setBatchSize`, `InferenceEngine::CNNNetwork::getBatchSize` |  Managing of network, operate with its batch size. Setting batch size using input image count.

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [fast-neural-style-mosaic-onnx](@ref omz_models_model_fast_neural_style_mosaic_onnx)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | The sample uses OpenCV\* to [read input image](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) (\*.bmp, \*.png)
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [Python](../../ie_bridges/python/sample/style_transfer_sample/README.md) |

## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network and image(s) to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image(s), logging each step in a standard output stream.

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
[ INFO ] InferenceEngine:
        API version ............<version>
        Build ..................<build>
        Description ....... API
[ INFO ] Parsing input parameters

style_transfer_sample [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Path to a folder with images or paths to image files.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"  Required for CPU plugin custom layers. Absolute path to a shared library with the kernels implementations.
          Or
      -c "<absolute_path>"  Required for GPU, MYRIAD, HDDL custom kernels. Absolute path to the .xml config file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified.
    -mean_val_r,
    -mean_val_g,
    -mean_val_b             Mean values. Required if the model needs mean values for preprocessing and postprocessing.

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
python <path_to_omz_tools>/downloader.py --name fast-neural-style-mosaic-onnx
```

2. `fast-neural-style-mosaic-onnx` model does not need to be converted, because it is already in necessary format, so you can skip this step. If you want to use a other model that is not in the Inference Engine IR or ONNX format, you can convert it using the model converter script:

```
python <path_to_omz_tools>/converter.py --name <model_name>
```

3. Perform inference of `car.bmp` and `cat.jpg` using `fast-neural-style-mosaic-onnx` model on a `GPU`, for example:

```
<path_to_sample>/style_transfer_sample -m <path_to_model>/fast-neural-style-mosaic-onnx.onnx -i <path_to_image>/car.bmp <path_to_image>/cat.jpg -d GPU
```

## Sample Output

The sample application logs each step in a standard output stream and creates an image (`out1.bmp`) or a sequence of images (`out1.bmp`, ..., `out<N>.bmp`) which are redrawn in style of the style transfer model used for the sample.

```
[ INFO ] InferenceEngine:
        IE version ......... 2021.4.0
        Build ........... 2021.4.0-3839-cd81789d294-releases/2021/4
[ INFO ] Parsing input parameters
[ INFO ] Files were added: 2
[ INFO ]     C:\images\car.bmp
[ INFO ]     C:\images\cat.jpg
[ INFO ] Loading Inference Engine
[ INFO ] Device info:
        GPU
        clDNNPlugin version ......... 2021.4.0
        Build ........... 2021.4.0-3839-cd81789d294-releases/2021/4

[ INFO ] Loading network files:
[ INFO ] C:\openvino\deployment_tools\open_model_zoo\tools\downloader\public\fast-neural-style-mosaic-onnx\fast-neural-style-mosaic-onnx.onnx
[ INFO ] Preparing input blobs
[ WARNING ] Image is resized from (749, 637) to (224, 224)
[ WARNING ] Image is resized from (300, 300) to (224, 224)
[ INFO ] Batch size is 2
[ INFO ] Preparing output blobs
[ INFO ] Loading model to the device
[ INFO ] Create infer request
[ INFO ] Start inference
[ INFO ] Output size [N,C,H,W]: 2, 3, 224, 224
[ INFO ] Image out1.bmp created!
[ INFO ] Image out2.bmp created!
[ INFO ] Execution successful

[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
