# Hello Autoresize Classification Sample {#InferenceEngineHelloAutoresizeClassificationSample}

This topic describes how to run the Hello Autoresize Classification sample application.
The sample is simplified version of [Image Classification Sample](@ref InferenceEngineClassificationSampleApplication). 
It's intended to demonstrate using of new input autoresize API of Inference Engine in applications. Refer to
[Integrate with customer application New Request API](@ref IntegrateIEInAppNewAPI) for details.

There is also new API introduced to crop a ROI object and set it as input without additional memory re-allocation.
To properly demonstrate this new API it's required to run several networks in pipeline which is out of scope of this sample.
Please refer to [Object Detection for SSD Demo app](@ref InferenceEngineObjectDetectionSSDDemoAsyncApplication) or
[Security Barrier Camera Demo](@ref InferenceEngineSecurityBarrierCameraDemoApplication) or
[Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication) with an example of using of new crop ROI API.

## Running

You can do inference on an image using a trained AlexNet network on Intel&reg; Processors using the following command:
```sh
./hello_autoresize_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp CPU
```

### Outputs

The application outputs top-10 inference results. 

## See Also 
* [Using Inference Engine Samples](@ref SamplesOverview)
