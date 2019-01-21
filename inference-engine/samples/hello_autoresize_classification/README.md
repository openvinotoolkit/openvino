# Hello Autoresize Classification Sample

This topic describes how to run the Hello Autoresize Classification sample application.
The sample is simplified version of [Image Classification Sample](./samples/classification_sample/README.md).
It's intended to demonstrate using of new input autoresize API of Inference Engine in applications. Refer to
[Integrate with customer application New Request API](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

There is also new API introduced to crop a ROI object and set it as input without additional memory re-allocation.
To properly demonstrate this new API it's required to run several networks in pipeline which is out of scope of this sample.
Please refer to [Object Detection for SSD Demo app](./samples/object_detection_demo_ssd_async/README.md) or
[Security Barrier Camera Demo](./samples/security_barrier_camera_demo/README.md) or
[Crossroad Camera Demo](./samples/crossroad_camera_demo/README.md) with an example of using of new crop ROI API.

## Running

You can do inference on an image using a trained AlexNet network on Intel&reg; Processors using the following command:
```sh
./hello_autoresize_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp CPU
```

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

### Outputs

The application outputs top-10 inference results. 

## See Also 
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
