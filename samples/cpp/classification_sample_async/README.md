# Image Classification Async C++ Sample {#openvino_inference_engine_samples_classification_sample_async_README}

This sample demonstrates how to do inference of image classification models using Asynchronous Inference Request API.  
Models with only 1 input and output are supported.

In addition to regular images, the sample also supports single-channel `ubyte` images as an input for LeNet model.

The following C++ API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| Asynchronous Infer | `ov::InferRequest::start_async`, `ov::InferRequest::set_callback` | Do asynchronous inference with callback. |
| Model Operations | `ov::Output::get_shape`, `ov::set_batch` |  Manage the model, operate with its batch size. Set batch size using input image count. |
| Infer Request Operations | `ov::InferRequest::get_input_tensor` | Get an input tensor. |
| Tensor Operations | `ov::shape_size`, `ov::Tensor::data` | Get a tensor shape size and its data. |

Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options | Values |
| :--- | :--- |
| Validated Models | [alexnet](@ref omz_models_model_alexnet), [googlenet-v1](@ref omz_models_model_googlenet_v1) |
| Model Format | OpenVINO Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Supported devices | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [Python](../../../samples/python/classification_sample_async/README.md) |

## How It Works

At startup, the sample application reads command-line parameters and loads the specified model and input images (or a
folder with images) to the OpenVINO Runtime plugin. The batch size of the model is set according to the number of read images. The batch mode is an independent attribute on the asynchronous mode. Asynchronous mode works efficiently with any batch size.

Then, the sample creates an inference request object and assigns completion callback for it. In scope of the completion callback,
handling the inference request is executed again.

After that, the application starts inference for the first infer request and waits until the 10th inference request execution is completed. The asynchronous mode might increase the throughput of the pictures.

When inference is done, the application outputs data to the standard output stream. You may place labels in `.labels` file near the model to get pretty output.

For more information, refer to the explicit description of
each sample **Integration Step** in the [Integrate OpenVINO Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md) guide.

## Building

To build the sample, use the instructions available in the **Build the Sample Applications** section in [OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md).

## Running

Before running the sample, you need to specify a model and an image:

- you may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded by using the [Model Downloader](@ref omz_tools_downloader).
- you may use images from the media files collection, available online in the [test data storage](https://storage.openvinotoolkit.org/data/test_data).

Run the application with the `-h` option to see the usage instructions:

```
classification_sample_async -h
```

Usage instructions:

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>

classification_sample_async [OPTION]
Options:

    -h                      Print usage instructions.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -i "<path>"             Required. Path to a folder with images or path to image files: a .ubyte file for LeNet and a .bmp file for other models.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify the HETERO plugin. Sample will look for a suitable plugin for the device specified.

Available target devices: <devices>
```

> **NOTES**:
> - By default, samples and demos in OpenVINO Toolkit expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default order of channels in the sample or demo application, or reconvert your model, using Model Optimizer with `--reverse_input_channels` argument specified. For more information about the argument, refer to the **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure that the model is converted to the OpenVINO Intermediate Representation (OpenVINO IR) format (\*.xml + \*.bin) using [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in the ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model, using:

```
omz_downloader --name googlenet-v1
```

3. If a model is not in the OpenVINO IR or ONNX format, it must be converted with Model Converter:

```
omz_converter --name googlenet-v1
```

4. Perform inference of the `dog.bmp` image, using the `googlenet-v1` model on a `GPU`, for example:

```
classification_sample_async -m googlenet-v1.xml -i dog.bmp -d GPU
```

## Sample Output

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>
[ INFO ]
[ INFO ] Parsing input parameters
[ INFO ] Files were added: 1
[ INFO ]     /images/dog.bmp
[ INFO ] Loading model files:
[ INFO ] /models/googlenet-v1.xml
[ INFO ] model name: GoogleNet
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 224, 224}
[ INFO ]     outputs
[ INFO ]         output name: prob
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1000}
[ INFO ] Read input images
[ INFO ] Set batch size 1
[ INFO ] model name: GoogleNet
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: u8
[ INFO ]         input shape: {1, 224, 224, 3}
[ INFO ]     outputs
[ INFO ]         output name: prob
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1000}
[ INFO ] Loading model to the device GPU
[ INFO ] Create infer request
[ INFO ] Start inference (asynchronous executions)
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
[ INFO ] Completed async requests execution

Top 10 results:

Image /images/dog.bmp

classid probability
------- -----------
156     0.8935547
218     0.0608215
215     0.0217133
219     0.0105667
212     0.0018835
217     0.0018730
152     0.0018730
157     0.0015745
154     0.0012817
220     0.0010099
```

## See Also

- [Integrate the OpenVINO Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [OpenVINO Toolkit Test Data Storage](https://storage.openvinotoolkit.org/data/test_data).