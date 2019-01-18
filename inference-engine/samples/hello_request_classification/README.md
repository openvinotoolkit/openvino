# Hello Infer Request Classification Sample

This topic describes how to run the Hello Infer Classification sample application.
The sample is simplified version of [Image Classification Sample](./samples/classification_sample/README.md).
It's intended to demonstrate using of new Infer Request API of Inference Engine in applications. Refer to 
[Integrate with customer application New Request API](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

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
