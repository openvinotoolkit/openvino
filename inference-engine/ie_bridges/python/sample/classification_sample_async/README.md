# Python* Classification Sample Async {#openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README}

This sample demonstrates how to do inference of image classification networks using Asynchronous Inference Request API.  
Models with only 1 input and output are supported.

The following Inference Engine Python API is used in the application:

| Feature                  | API                                                                                                                         | Description                                           |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| Basic Infer Flow         | [IECore], [IECore.read_network], [IECore.load_network]                                                                      | Common API to do inference                            |
| Asynchronous Infer       | [InferRequest.async_infer]                                                                                                  | Do asynchronous inference                             |
| Network Operations       | [IENetwork.input_info], [IENetwork.outputs], [InputInfoPtr.precision], [DataPtr.precision], [InputInfoPtr.input_data.shape] | Managing of network: configure input and output blobs |
| Custom Extension Kernels | [IECore.add_extension], [IECore.set_config]                                                                                 | Load extension library and config to the device       |

| Options                    | Values                                                                                                    |
| :------------------------- | :-------------------------------------------------------------------------------------------------------- |
| Validated Models           | [alexnet](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/alexnet/alexnet.md) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)                                  |
| Supported devices          | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md)                                   |
| Other language realization | [C++](../../../../samples/classification_sample_async)                                                    |

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to Inference Engine plugin, performs synchronous inference and processes output data, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Running

Run the application with the -h option to see the usage message:

```
python3 classification_sample_async.py -h
```

Usage message:

```
usage: classification_sample_async.py [-h] -m MODEL -i INPUT [INPUT ...]
                                      [-l EXTENSION] [-c CONFIG] [-d DEVICE]
                                      [--labels LABELS] [-nt NUMBER_TOP]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml or .onnx file with a trained
                        model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to an image file(s).
  -l EXTENSION, --extension EXTENSION
                        Optional. Required by the CPU Plugin for executing the
                        custom operation on a CPU. Absolute path to a shared
                        library with the kernels implementations.
  -c CONFIG, --config CONFIG
                        Optional. Required by GPU or VPU Plugins for the
                        custom operation kernel. Absolute path to operation
                        description file (.xml).
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, MYRIAD, HDDL or HETERO: is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
  --labels LABELS       Optional. Path to a labels mapping file.
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results.
```

To run the sample, you need specify a model and image:
 - you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
 - you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> * By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
>
> * Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> * The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

You can do inference of an image using a pre-trained model on a GPU using the following command:

```
python3 classification_sample_async.py -m <path_to_model>/alexnet.xml -i <path_to_image>/cat.bmp <path_to_image>/car.bmp -d GPU
```

## Sample Output

The sample application logs each step in a standard output stream and outputs top-10 inference results.

```
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: models\alexnet.xml
[ INFO ] Configuring input and output blobs
[ INFO ] Loading the model to the plugin
[ WARNING ] Image images\cat.jpg is resized from (300, 300) to (227, 227)
[ WARNING ] Image images\car.png is resized from (259, 787) to (227, 227)
[ INFO ] Starting inference in asynchronous mode
[ INFO ] Infer request 0 returned 0
[ INFO ] Image path: images\cat.jpg
[ INFO ] Top 10 results:
[ INFO ] ---------------------
[ INFO ] probability | classid
[ INFO ] ---------------------
[ INFO ] 0.099689044 | 435
[ INFO ] 0.090024225 | 876
[ INFO ] 0.069144860 | 999
[ INFO ] 0.039018910 | 587
[ INFO ] 0.036039289 | 666
[ INFO ] 0.030830737 | 419
[ INFO ] 0.030628702 | 285
[ INFO ] 0.029300876 | 700
[ INFO ] 0.020270746 | 696
[ INFO ] 0.019912610 | 631
[ INFO ]
[ INFO ] Infer request 1 returned 0
[ INFO ] Image path: images\car.png
[ INFO ] Top 10 results:
[ INFO ] ---------------------
[ INFO ] probability | classid
[ INFO ] ---------------------
[ INFO ] 0.756178558 | 479
[ INFO ] 0.075569794 | 511
[ INFO ] 0.073027283 | 436
[ INFO ] 0.046027526 | 817
[ INFO ] 0.030379366 | 656
[ INFO ] 0.005528264 | 661
[ INFO ] 0.003129612 | 581
[ INFO ] 0.002987558 | 468
[ INFO ] 0.002279181 | 717
[ INFO ] 0.001629688 | 627
[ INFO ]
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

* [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
* [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
* [Model Downloader](@ref omz_tools_downloader_README)
* [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[IECore]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html
[IECore.add_extension]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a8a4b671a9928c7c059bd1e76d2333967
[IECore.set_config]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05
[IECore.read_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a0d69c298618fab3a08b855442dca430f
[IENetwork.input_info]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#data_fields
[IENetwork.outputs]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#data_fields
[InputInfoPtr.precision]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InputInfoPtr.html#data_fields
[DataPtr.precision]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1DataPtr.html#data_fields
[IECore.load_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc
[InputInfoPtr.input_data.shape]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InputInfoPtr.html#data_fields
[InferRequest.async_infer]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InferRequest.html#a95ebe0368cdf4d5d64f9fddc8ee1cd0e