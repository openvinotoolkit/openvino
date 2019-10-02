# Hello NV12 Input Classification C++ Sample

This topic describes how to run the Hello NV12 Input Classification sample application.
The sample is a simplified version of the [Image Classification Sample Async](./inference-engine/samples/classification_sample_async/README.md).
It demonstrates how to use the new NV12 automatic input pre-processing API of the Inference Engine in your applications.
Refer to [Integrate the Inference Engine New Request API with Your Application](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads a network and sets an
image in the NV12 color format to an Inference Engine plugin. When inference is done, the
application outputs data to the standard output stream.

The sample accepts an uncompressed image in the NV12 color format. To run the sample, you need to
convert your BGR/RGB image to NV12. To do this, you can use one of the widely available tools such
as FFmpeg\* or GStreamer\*. The following command shows how to convert an ordinary image into an
uncompressed NV12 image using FFmpeg:
```sh
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

> **NOTE**:
>
> * Because the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> * The sample uses input autoresize API of the Inference Engine to simplify user-side
>   pre-processing.
> * By default, this sample expects that network input has BGR channels order. If you trained your
>   model to work with RGB order, you need to reconvert your model using the Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use public or pre-trained models. To download pre-trained models, use
the OpenVINO&trade; [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader)
or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the
> Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

You can perform inference on an NV12 image using a trained AlexNet network on CPU with the following command:
```sh
./hello_nv12_input_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.yuv 640x480 CPU
```

## Sample Output

The application outputs top-10 inference results.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
