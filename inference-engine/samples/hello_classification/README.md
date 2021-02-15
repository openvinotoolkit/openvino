# Hello Classification C++ Sample {#openvino_inference_engine_samples_hello_classification_README}

This topic describes how to run the Hello Infer Classification sample application.
The sample is simplified version of [Image Classification Sample Async](../classification_sample_async/README.md)
and developed with support of UNICODE.
It demonstrates how to use the following Inference Engine API in applications:
* Synchronous Infer Request API
* Input auto-resize API. It allows to set image of the original size as input for a network with other input size.
  Resize will be performed automatically by the corresponding plugin just before inference. 

There is also an API introduced to crop a ROI object and set it as input without additional memory re-allocation.
To properly demonstrate this API, it is required to run several networks in pipeline which is out of scope of this sample.
Please refer to [Security Barrier Camera Demo](@ref omz_demos_security_barrier_camera_demo_README), or
[Crossroad Camera Demo](@ref omz_demos_crossroad_camera_demo_README) with an example of using of new crop ROI API.

Refer to [Integrate the Inference Engine New Request API with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
 
You can do inference of an image using a trained AlexNet network on a GPU using the following command:
```sh
./hello_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp GPU
```

## Sample Output

The application outputs top-10 inference results.

## See Also
* [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
