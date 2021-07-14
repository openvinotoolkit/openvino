# Hello Classification Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_classification_README}

This sample demonstrates how to do inference of image classification networks using Synchronous Inference Request API.  
Models with only 1 input and output are supported.

The following Inference Engine Python API is used in the application:

| Feature            | API                                                                                                                         | Description                                           |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| Basic Infer Flow   | [IECore], [IECore.read_network], [IECore.load_network]                                                                      | Common API to do inference                            |
| Synchronous Infer  | [ExecutableNetwork.infer]                                                                                                   | Do synchronous inference                              |
| Network Operations | [IENetwork.input_info], [IENetwork.outputs], [InputInfoPtr.precision], [DataPtr.precision], [InputInfoPtr.input_data.shape] | Managing of network: configure input and output blobs |

| Options                    | Values                                                                                                    |
| :------------------------- | :-------------------------------------------------------------------------------------------------------- |
| Validated Models           | [alexnet](@ref omz_models_model_alexnet), [googlenet-v1](@ref omz_models_model_googlenet_v1) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)                                  |
| Supported devices          | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md)                                   |
| Other language realization | [C++](../../../../samples/hello_classification/README.md), [C](../../../c/samples/hello_classification/README.md)             |

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the Inference Engine plugin, performs synchronous inference, and processes output data, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Running

Run the application with the `-h` option to see the usage message:

```
python <path_to_sample>/hello_classification.py -h
```

Usage message:

```
usage: hello_classification.py [-h] -m MODEL -i INPUT [-d DEVICE]
                               [--labels LABELS] [-nt NUMBER_TOP]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml or .onnx file with a trained
                        model.
  -i INPUT, --input INPUT
                        Required. Path to an image file.
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
> - By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

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
python <path_to_sample>/hello_classification.py -m <path_to_model>/alexnet.xml -i <path_to_image>/car.bmp -d GPU
```

## Sample Output

The sample application logs each step in a standard output stream and outputs top-10 inference results.

```
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: c:\openvino\deployment_tools\open_model_zoo\tools\downloader\public\alexnet\FP32\alexnet.xml
[ INFO ] Configuring input and output blobs
[ INFO ] Loading the model to the plugin
[ WARNING ] Image c:\images\car.bmp is resized from (637, 749) to (227, 227)
[ INFO ] Starting inference in synchronous mode
[ INFO ] Image path: c:\images\car.bmp
[ INFO ] Top 10 results:    
[ INFO ] classid probability
[ INFO ] -------------------
[ INFO ] 656     0.6645315
[ INFO ] 654     0.1121185
[ INFO ] 581     0.0698451
[ INFO ] 874     0.0334973
[ INFO ] 436     0.0259718
[ INFO ] 817     0.0173190
[ INFO ] 675     0.0109321
[ INFO ] 511     0.0109075
[ INFO ] 569     0.0083093
[ INFO ] 717     0.0063173
[ INFO ]
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[IECore]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html
[IECore.read_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a0d69c298618fab3a08b855442dca430f
[IENetwork.input_info]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#data_fields
[IENetwork.outputs]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#data_fields
[InputInfoPtr.precision]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InputInfoPtr.html#data_fields
[DataPtr.precision]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1DataPtr.html#data_fields
[IECore.load_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc
[InputInfoPtr.input_data.shape]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InputInfoPtr.html#data_fields
[ExecutableNetwork.infer]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#aea96e8e534c8e23d8b257bad11063519
