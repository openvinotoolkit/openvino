# Hello Infer Request Classification Sample

This topic describes how to run the Hello Infer Classification sample application.
The sample is simplified version of [Image Classification Sample](./samples/classification_sample/README.md).
It's intended to demonstrate using of new Infer Request API of Inference Engine in applications. Refer to 
[Integrate with customer application New Request API](./docs/Inference_Engine_Developer_Guide/Integrate_with_customer_application_new_API.md) for details.

## Running

You can do inference on an image using a trained AlexNet network on Intel&reg; Processors using the following command:
```sh
./hello_autoresize_classification <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.bmp CPU
```

### Outputs

The application outputs top-10 inference results. 


## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
