# Hello Classification C Sample

This sample demonstrates how to execute an inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API and input auto-resize feature.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\c_sample_hello_classification.md)

## Requirements

| Options                    | Values                                                                                                                                                                      |
| ---------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models           | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`                                                                              |
| Model Format               | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                              |
| Validated images           | The sample uses OpenCV\* to `read input image <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`__ (\*.bmp, \*.png)          |
| Supported devices          | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                                                                                                        |
| Other language realization | :doc:`C++ <openvino_inference_engine_samples_hello_classification_README>`, :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>`  |

Hello Classification C sample application demonstrates how to use the C API from OpenVINO in applications.

| Feature                  | API                                                         | Description                                                                                                                                                                             |
| -------------------------| ------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OpenVINO Runtime Version | ``ov_get_openvino_version``                                 | Get Openvino API version.                                                                                                                                                               |
| Basic Infer Flow         | ``ov_core_create``,                                         | Common API to do inference: read and compile a model, create an infer request, configure input and output tensors                                                                       |
|                          | ``ov_core_read_model``,                                     |                                                                                                                                                                                         |
|                          | ``ov_core_compile_model``,                                  |                                                                                                                                                                                         |
|                          | ``ov_compiled_model_create_infer_request``,                 |                                                                                                                                                                                         |
|                          | ``ov_infer_request_set_input_tensor_by_index``,             |                                                                                                                                                                                         |
|                          | ``ov_infer_request_get_output_tensor_by_index``             |                                                                                                                                                                                         |
| Synchronous Infer        | ``ov_infer_request_infer``                                  | Do synchronous inference                                                                                                                                                                |
| Model Operations         | ``ov_model_const_input``,                                   | Get inputs and outputs of a model                                                                                                                                                       |
|                          | ``ov_model_const_output``                                   |                                                                                                                                                                                         |
| Tensor Operations        | ``ov_tensor_create_from_host_ptr``                          | Create a tensor shape                                                                                                                                                                   |
| Preprocessing            | ``ov_preprocess_prepostprocessor_create``,                  | Set image of the original size as input for a model with other input size. Resize and layout conversions are performed automatically by the corresponding plugin just before inference. |
|                          | ``ov_preprocess_prepostprocessor_get_input_info_by_index``, |                                                                                                                                                                                         |
|                          | ``ov_preprocess_input_info_get_tensor_info``,               |                                                                                                                                                                                         |
|                          | ``ov_preprocess_input_tensor_info_set_from``,               |                                                                                                                                                                                         |
|                          | ``ov_preprocess_input_tensor_info_set_layout``,             |                                                                                                                                                                                         |
|                          | ``ov_preprocess_input_info_get_preprocess_steps``,          |                                                                                                                                                                                         |
|                          | ``ov_preprocess_preprocess_steps_resize``,                  |                                                                                                                                                                                         |
|                          | ``ov_preprocess_input_model_info_set_layout``,              |                                                                                                                                                                                         |
|                          | ``ov_preprocess_output_set_element_type``,                  |                                                                                                                                                                                         | 
|                          | ``ov_preprocess_prepostprocessor_build``                    |                                                                                                                                                                                         |

