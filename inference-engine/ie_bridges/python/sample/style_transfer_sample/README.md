# Style Transfer Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_style_transfer_sample_README}

This sample demonstrates how to do synchronous inference of style transfer networks using Network Batch Size Feature.  
You can specify multiple images to input, a network batch size will be set equal to their number automatically.  
Models with only 1 input and output are supported.

The following Inference Engine Python API is used in the application:

| Feature                  | API                                                                                                                                                 | Description                                           |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| Network Operations       | [IENetwork.batch_size] | Managing of network: configure input and output blobs |
| Custom Extension Kernels | [IECore.add_extension], [IECore.set_config]                                                                                                         | Load extension library and config to the device       |

Basic Inference Engine API is covered by [Hello Classification Python* Sample](../hello_classification/README.md).

| Options                    | Values                                                                                                                                                                      |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Validated Models           | [fast-neural-style-mosaic-onnx](@ref omz_models_model_fast_neural_style_mosaic_onnx) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)                                                                                                    |
| Supported devices          | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md)                                                                                                     |
| Other language realization | [C++](../../../../samples/style_transfer_sample/README.md)                                                                                                                            |

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image(s) to the Inference Engine plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image(s), logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Running

Run the application with the `-h` option to see the usage message:

```
python <path_to_sample>/style_transfer_sample.py -h
```

Usage message:

```
usage: style_transfer_sample.py [-h] -m MODEL -i INPUT [INPUT ...]
                                [-l EXTENSION] [-c CONFIG] [-d DEVICE]
                                [--original_size] [--mean_val_r MEAN_VAL_R]
                                [--mean_val_g MEAN_VAL_G]
                                [--mean_val_b MEAN_VAL_B]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml or .onnx file with a trained
                        model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
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
  --original_size       Optional. Resize an output image to original image
                        size.
  --mean_val_r MEAN_VAL_R
                        Optional. Mean value of red channel for mean value
                        subtraction in postprocessing.
  --mean_val_g MEAN_VAL_G
                        Optional. Mean value of green channel for mean value
                        subtraction in postprocessing.
  --mean_val_b MEAN_VAL_B
                        Optional. Mean value of blue channel for mean value
                        subtraction in postprocessing.
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
python <path_to_omz_tools>/downloader.py --name fast-neural-style-mosaic-onnx
```

2. `fast-neural-style-mosaic-onnx` model does not need to be converted, because it is already in necessary format, so you can skip this step. If you want to use a other model that is not in the Inference Engine IR or ONNX format, you can convert it using the model converter script:

```
python <path_to_omz_tools>/converter.py --name <model_name>
```

3. Perform inference of `car.bmp` and `cat.jpg` using `fast-neural-style-mosaic-onnx` model on a `GPU`, for example:

```
python <path_to_sample>/style_transfer_sample.py -m <path_to_model>/fast-neural-style-mosaic-onnx.onnx -i <path_to_image>/car.bmp <path_to_image>/cat.jpg -d GPU
```

## Sample Output

The sample application logs each step in a standard output stream and creates an output image (`out_0.bmp`) or a sequence of images (`out_0.bmp`, .., `out_<n>.bmp`) that are redrawn in the style of the style transfer model used.

```
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: c:\openvino\deployment_tools\open_model_zoo\tools\downloader\public\fast-neural-style-mosaic-onnx\fast-neural-style-mosaic-onnx.onnx
[ INFO ] Configuring input and output blobs
[ INFO ] Loading the model to the plugin
[ WARNING ] Image c:\images\car.bmp is resized from (637, 749) to (224, 224)
[ WARNING ] Image c:\images\cat.jpg is resized from (300, 300) to (224, 224)
[ INFO ] Starting inference in synchronous mode
[ INFO ] Image out_0.bmp created!
[ INFO ] Image out_1.bmp created!
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
[IENetwork.batch_size]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e
[IECore.load_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc
[InputInfoPtr.input_data.shape]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InputInfoPtr.html#data_fields
[ExecutableNetwork.infer]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#aea96e8e534c8e23d8b257bad11063519