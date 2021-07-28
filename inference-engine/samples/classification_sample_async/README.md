# Image Classification Async C++ Sample {#openvino_inference_engine_samples_classification_sample_async_README}

This sample demonstrates how to execute an inference of image classification networks like AlexNet and GoogLeNet using Asynchronous Inference Request API.

In addition to regular images, the sample also supports single-channel `ubyte` images as an input for LeNet model.

Image Classification Async C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Inference Engine Version| `InferenceEngine::GetInferenceEngineVersion` | Get Inference Engine API version
|Available Devices|`InferenceEngine::Core::GetAvailableDevices`| Get version information of the devices for inference
| Asynchronous Infer | `InferenceEngine::InferRequest::StartAsync`, `InferenceEngine::InferRequest::SetCompletionCallback` | Do asynchronous inference with callback
|Custom Extension Kernels|`InferenceEngine::Core::AddExtension`, `InferenceEngine::Core::SetConfig`| Load extension library and config to the device
| Network Operations | `InferenceEngine::CNNNetwork::setBatchSize`, `InferenceEngine::CNNNetwork::getBatchSize`, `InferenceEngine::CNNNetwork::getFunction` |  Managing of network, operate with its batch size. Setting batch size using input image count.

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | [alexnet](@ref omz_models_model_alexnet), [googlenet-v1](@ref omz_models_model_googlenet_v1)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | The sample uses OpenCV\* to [read input image](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) (\*.bmp, \*.png), single-channel `ubyte` images.
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [Python](../../ie_bridges/python/sample/classification_sample_async/README.md) |

## How It Works

Upon the start-up, the sample application reads command line parameters and loads specified network and input images (or a
folder with images) to the Inference Engine plugin. The batch size of the network is set according to the number of read images. The batch mode is an independent attribute on the asynchronous mode. Asynchronous mode works efficiently with any batch size.

Then, the sample creates an inference request object and assigns completion callback for it. In scope of the completion callback
handling the inference request is executed again.

After that, the application starts inference for the first infer request and waits of 10th inference request execution being completed. The asynchronous mode might increase the throughput of the pictures.

When inference is done, the application outputs data to the standard output stream. You can place labels in .labels file near the model to get pretty output.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Running the application with the `-h` option yields the following usage message:

```
<path_to_sample>/classification_sample_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <build>
    Description ....... API

classification_sample_async [OPTION]
Options:

    -h                      Print a usage message.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -i "<path>"             Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet and a .bmp file for the other networks.
      -l "<absolute_path>"  Required for CPU plugin custom layers. Absolute path to a shared library with the kernels implementations
          Or
      -c "<absolute_path>"  Required for GPU, MYRIAD, HDDL custom kernels. Absolute path to the .xml config file with the kernels descriptions.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify HETERO plugin. Sample will look for a suitable plugin for device specified.
    -nt "<integer>"         Optional. Number of top results. Default value is 10.

    Available target devices: <devices>

```

Running the application with the empty list of options yields the usage message given above and an error message.

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader_README):
```
python <path_to_omz_tools>/downloader.py --name alexnet
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name alexnet
```

3. Perform inference of `car.bmp` using `alexnet` model on a `GPU`, for example:

```
<path_to_sample>/classification_sample_async -m <path_to_model>/alexnet.xml -i <path_to_image>/car.bmp -d GPU
```

## Sample Output

By default the application outputs top-10 inference results for each infer request.

```
[ INFO ] InferenceEngine:
        IE version ......... 2021.4.0
        Build ........... 2021.4.0-3839-cd81789d294-releases/2021/4
[ INFO ] Parsing input parameters
[ INFO ] Files were added: 1
[ INFO ]     C:\images\car.bmp
[ INFO ] Loading Inference Engine
[ INFO ] Device info:
        GPU
        clDNNPlugin version ......... 2021.4.0
        Build ........... 2021.4.0-3839-cd81789d294-releases/2021/4

[ INFO ] Loading network files:
[ INFO ] C:\openvino\deployment_tools\open_model_zoo\tools\downloader\public\alexnet\FP32\alexnet.xml
[ INFO ] Preparing input blobs
[ WARNING ] Image is resized from (749, 637) to (227, 227)
[ INFO ] Batch size is 1
[ INFO ] Loading model to the device
[ INFO ] Create infer request
[ INFO ] Start inference (10 asynchronous executions)
[ INFO ] Completed 1 async request execution
[ INFO ] Completed 2 async request execution
[ INFO ] Completed 3 async request execution
[ INFO ] Completed 4 async request execution
[ INFO ] Completed 5 async request execution
[ INFO ] Completed 6 async request execution
[ INFO ] Completed 7 async request execution
[ INFO ] Completed 8 async request execution
[ INFO ] Completed 9 async request execution
[ INFO ] Completed 10 async request execution
[ INFO ] Processing output blobs

Top 10 results:

Image C:\images\car.bmp

classid probability
------- -----------
656     0.6645315
654     0.1121185
581     0.0698451
874     0.0334973
436     0.0259718
817     0.0173190
675     0.0109321
511     0.0109075
569     0.0083093
717     0.0063173

[ INFO ] Execution successful

[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
