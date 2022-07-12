# Hello Reshape SSD Python Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README}

This sample demonstrates how to do synchronous inference of object detection models using the [Shape Inference feature](../../../docs/OV_Runtime_UG/ShapeInference.md).  
Models with only 1 input and output are supported.

The following Python API is used in the application:

| Feature          | API                                                                                                                                       | Description       |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| Model Operations | [openvino.runtime.Model.reshape], [openvino.runtime.Model.input], [openvino.runtime.Output.get_any_name], [openvino.runtime.PartialShape] | Managing of model |

Basic OpenVINO™ Runtime API is described in [Hello Classification Python Sample](../hello_classification/README.md).

| Options                    | Values                                                                   |
| :------------------------- | :----------------------------------------------------------------------- |
| Validated Models           | [mobilenet-ssd](@ref omz_models_model_mobilenet_ssd)                     |
| Validated Layout           | NCHW                                                                     |
| Model Format               | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx) |
| Supported devices          | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md)        |
| Other language realization | [C++](../../../samples/cpp/hello_reshape_ssd/README.md)                  |

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to OpenVINO Runtime plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image, logging each step in a standard output stream.

For more information, refer to the ["Integrate OpenVINO Runtime with Your Application" Guide](../../../docs/OV_Runtime_UG/integrate_with_your_application.md).

## Running

Before running the sample, specify a model and an image:

- Use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded by using [Model Downloader](@ref omz_tools_downloader).
- You may use images from the media files collection, available online in the [test data](https://storage.openvinotoolkit.org/data/test_data).

To run the sample, use the following script:

```
python hello_reshape_ssd.py <path_to_model> <path_to_image> <device_name>
```

> **NOTES**:
> - By default, samples and demos in OpenVINO Toolkit expect input with the `BGR` channel order. If you trained your model to work with `RGB` order, you need to manually rearrange the default channel order in the sample or demo application, or reconvert your model, using Model Optimizer with `--reverse_input_channels` argument specified. For more information about the argument, refer to the **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/Additional_Optimizations.md).
>
> - Before running the sample with a trained model, make sure that the model is converted to the OpenVINO Intermediate Representation format (\*.xml + \*.bin) by using [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in the ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model:
```
omz_downloader --name ssdlite_mobilenet_v2
```

3. If a model is not in the OpenVINO IR or ONNX format, it must be converted with Model Converter:

```
omz_converter --name ssdlite_mobilenet_v2
```

4. Do inference of the `banana.jpg` image, using the `ssdlite_mobilenet_v2` model on `GPU`, for example:

```
python hello_reshape_ssd.py ssdlite_mobilenet_v2.xml banana.jpg GPU
```

## Sample Output

The sample application logs each step in a standard output stream and creates an output image, drawing bounding boxes for inference results with an over 50% confidence.

```
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: C:/test_data/models/ssdlite_mobilenet_v2.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] Found: class_id = 52, confidence = 0.98, coords = (21, 98), (276, 210)
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## Additional Resources

- [Integrate the OpenVINO Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [OpenVINO Toolkit Test Data](https://storage.openvinotoolkit.org/data/test_data).

<!-- [openvino.runtime.Model.reshape]:
[openvino.runtime.Model.input]:
[openvino.runtime.Output.get_any_name]:
[openvino.runtime.PartialShape]: -->
