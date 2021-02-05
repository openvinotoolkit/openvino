# Hetero Image Classification C++ Sample Async {#openvino_inference_engine_samples_hetero_classification_sample_async_README}

This sample demonstrates how to run the Image Classification sample application with inference executed in the asynchronous mode with Hetero plugin. This sample uses manual splitting with specified layer being the last layer of the first sub-network.

You can do inference of an image using the following command:
```sh
./hetero_classification_sample_async -m <path_to_model>/alexnet_fp32.xml -i <path_to_image>/cat.bmp -d VPUX,CPU -split_layer=<layer_to_split_on>
```
