# Hello NV12 Input Classification C++ Sample {#openvino_inference_engine_samples_hello_nv12_input_classification_README}

This sample demonstrates how to execute an inference of image classification models with images in `NV12` color format using Synchronous Inference Request API.

The following C++ API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| Node Operations | `ov::Output::get_any_name` | Get a layer name |
| Infer Request Operations | `ov::InferRequest::set_tensor`, `ov::InferRequest::get_tensor` | Operate with tensors |
| Preprocessing | `ov::preprocess::InputTensorInfo::set_color_format`, `ov::preprocess::PreProcessSteps::convert_element_type`, `ov::preprocess::PreProcessSteps::convert_color` | Change the color format of the input data |

Basic OpenVINO™ Runtime API is described in [Hello Classification C++ sample](../hello_classification/README.md).

| Options | Values |
| :--- |:--- |
| Validated Models | [alexnet](@ref omz_models_model_alexnet) |
| Model Format | OpenVINO Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Validated images | An uncompressed image in the `NV12` color format - \*.yuv |
| Supported devices | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [C](../../../samples/c/hello_nv12_input_classification/README.md) |

## How It Works

At startup, the sample application reads command line parameters, loads the specified model and an
image in the `NV12` color format to an OpenVINO Runtime plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream. You can place labels in `.labels` file near the model to get pretty output.

For more information, refer to the explicit description of [Integration Steps](../../../docs/OV_Runtime_UG/integrate_with_your_application.md).

## Building

To build the sample, use the instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO Toolkit Samples.

## Running

Before running the sample, specify the model and the image:

- you may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded by using the [Model Downloader](@ref omz_tools_downloader).
- you may use images from the media files collection, available online in [test-data](https://storage.openvinotoolkit.org/data/test_data) storage.

The sample accepts an uncompressed image in the `NV12` color format. To run the sample, you need to
convert your `BGR`/`RGB` image to `NV12`. To do this, use one of the widely available tools such
as FFmpeg or GStreamer. The following command shows how to convert an ordinary image into an
uncompressed `NV12` image, using FFmpeg:

```sh
ffmpeg -i cat.jpg -pix_fmt nv12 car.yuv
```

To run the sample, use the following script:

```
hello_nv12_input_classification <path_to_model> <path_to_image> <image_size> <device_name>
```


> **NOTES**:
> - Because the sample reads raw image files, provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 `BGR`/`RGB` image, the corresponding `NV12` logical image size is also 640x480, whereas the
>   buffer size is 640x720.
>
> - By default, this sample expects that model input has `BGR` order of channels. If you trained your
>   model to work with `RGB` order, you need to reconvert your model, using Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure that the model is converted to OpenVINO Intermediate Representation format (\*.xml + \*.bin) by using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in the ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install openvino-dev Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model:
```
omz_downloader --name alexnet
```

3. If a model is not in OpenVINO IR or ONNX format, it must be converted with the Model Converter:

```
omz_converter --name alexnet
```

4. Perform inference of the `NV12` image, using the `alexnet` model on a `CPU`, for example:

```
hello_nv12_input_classification alexnet.xml car.yuv 300x300 CPU
```

## Sample Output

The application outputs top-10 inference results.

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>
[ INFO ]
[ INFO ] Loading model files: \models\alexnet.xml
[ INFO ] model name: AlexNet
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 227, 227}
[ INFO ]     outputs
[ INFO ]         output name: prob
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1000}

Top 10 results:

Image \images\car.yuv

classid probability
------- -----------
656     0.6668988
654     0.1125269
581     0.0679280
874     0.0340229
436     0.0257744
817     0.0169367
675     0.0110199
511     0.0106134
569     0.0083373
717     0.0061734
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)