# Object Detection SSD Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_object_detection_sample_ssd_README}

This sample demonstrates how to do inference of object detection networks using Synchronous Inference Request API.  
Models with 1 input and 1 or 2 outputs are supported.  
In the last case names of output blobs must be "boxes" and "labels".

The following Inference Engine Python API is used in the application:

| Feature                  | API                                                                                                                         | Description                                           |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| Custom Extension Kernels | [IECore.add_extension], [IECore.set_config]                                                                                 | Load extension library and config to the device       |

Basic Inference Engine API is covered by [Hello Classification Python* Sample](../hello_classification/README.md).

| Options                    | Values                                                                                                                                                                                                                                                                                |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Validated Models           | [mobilenet-ssd](@ref omz_models_model_mobilenet_ssd), [face-detection-0206](@ref omz_models_model_face_detection_0206) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)                                                                                                                                                                                                              |
| Supported devices          | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md)                                                                                                                                                                                                               |
| Other language realization | [C++](../../../../samples/object_detection_sample_ssd/README.md), [C](../../../c/samples/object_detection_sample_ssd/README.md)                                                                                                                                                                           |

## How It Works

On startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the Inference Engine plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Running

Run the application with the `-h` option to see the usage message:

```
python <path_to_sample>/object_detection_sample_ssd.py -h
```

Usage message:

```
usage: object_detection_sample_ssd.py [-h] -m MODEL -i INPUT [-l EXTENSION]
                                      [-c CONFIG] [-d DEVICE]
                                      [--labels LABELS]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml or .onnx file with a trained
                        model.
  -i INPUT, --input INPUT
                        Required. Path to an image file.
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
python <path_to_omz_tools>/downloader.py --name mobilenet-ssd
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name mobilenet-ssd
```

3. Perform inference of `car.bmp` using `mobilenet-ssd` model on a `GPU`, for example:

```
python <path_to_sample>/object_detection_sample_ssd.py -m <path_to_model>/mobilenet-ssd.xml -i <path_to_image>/car.bmp -d GPU
```

## Sample Output

The sample application logs each step in a standard output stream and creates an output image, drawing bounding boxes for inference results with an over 50% confidence.

```
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: c:\openvino\deployment_tools\open_model_zoo\tools\downloader\public\mobilenet-ssd\FP32\mobilenet-ssd.xml        
[ INFO ] Configuring input and output blobs
[ INFO ] Loading the model to the plugin
[ WARNING ] Image c:\images\car.bmp is resized from (637, 749) to (300, 300)
[ INFO ] Starting inference in synchronous mode
[ INFO ] Found: label = 7, confidence = 1.00, coords = (228, 120), (502, 460)
[ INFO ] Found: label = 7, confidence = 0.95, coords = (637, 233), (743, 608)
[ INFO ] Image out.bmp created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

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
[ExecutableNetwork.infer]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#aea96e8e534c8e23d8b257bad11063519