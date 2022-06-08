# Hello Reshape SSD Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README}

This sample demonstrates how to do synchronous inference of object detection models using [Shape Inference feature](../../../docs/OV_Runtime_UG/ShapeInference.md).  
Models with only one input and output are supported. Source code for this example is also available [on GitHub](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_reshape_ssd).

The following Python API is used in the application:

| Feature          | API                                                                                                                                       | Description       |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| Model Operations | [openvino.runtime.Model.reshape], [openvino.runtime.Model.input], [openvino.runtime.Output.get_any_name], [openvino.runtime.PartialShape] | Managing of model |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](../hello_classification/README.md).

| Options                    | Values                                                                   |
| :------------------------- | :----------------------------------------------------------------------- |
| Validated Models           | [ssdlite_mobilenet_v2](@ref omz_models_model_ssdlite_mobilenet_v2)       |
| Model Format               | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx) |
| Supported devices          | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md)        |
| Other language realization | [C++](../../../samples/cpp/hello_reshape_ssd/README.md)                  |

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

```
python hello_reshape_ssd.py <path_to_model> <path_to_image> <device_name>
```

To run the sample, you need specify a model and image:
- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).
>
> - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model:
```
omz_downloader --name ssdlite_mobilenet_v2
```

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

```
omz_converter --name ssdlite_mobilenet_v2
```

4. Perform inference of `banana.jpg` using `ssdlite_mobilenet_v2` model on a `GPU`, for example:

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

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

<!-- [openvino.runtime.Model.reshape]:
[openvino.runtime.Model.input]:
[openvino.runtime.Output.get_any_name]:
[openvino.runtime.PartialShape]: -->
