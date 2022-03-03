# Inference Pipeline {#openvino_2_0_inference_pipeline}

Usually to inference model with the OpenVINO™ Runtime an user needs to do the following steps in the application pipeline:
- 1. Create Core object
- 2. Read model from the disk
 - 2.1. (Optional) Model preprocessing
- 3. Load the model to the device
- 4. Create an inference request
- 5. Fill input tensors with data
- 6. Start inference
- 7. Process the inference results

Code snippets below cover these steps and show how application code should be changed for migration to OpenVINO™ Runtime 2.0.

## 1. Create Core

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_core

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core

## 2. Read model from the disk

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:read_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model

Read model has the same structure as in the example from [Model Creation](./graph_construction.md) migration guide.

Note, you can combine read and compile model stages into a single call `ov::Core::compile_model(filename, devicename)`.

### 2.1 (Optional) Model preprocessing

When application's input data doesn't perfectly match with model's input format, preprocessing steps may need to be added.
See detailed guide [how to migrate preprocessing in OpenVINO Runtime API 2.0](./preprocessing.md)

## 3. Load the Model to the Device

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:compile_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model

If you need to configure OpenVINO Runtime devices with additional configuration parameters, please, refer to the migration [Configure devices](./configure_devices.md) guide.

## 4. Create an Inference Request

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_infer_request

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request

## 5. Fill input tensors

Inference Engine API fills inputs as `I32` precision (**not** aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]
       
.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]
       
.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

@endsphinxdirective

OpenVINO™ Runtime API 2.0 fills inputs as `I64` precision (aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_v10]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]
       
.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]
       
.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]

@endsphinxdirective

## 6. Start Inference

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:inference

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference

## 7. Process the Inference Results

Inference Engine API processes outputs as `I32` precision (**not** aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]
       
.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]
       
.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

@endsphinxdirective

OpenVINO™ Runtime API 2.0 processes outputs:
- For IR v10 as `I32` precision (**not** aligned with the original model) to match **old** behavior
- For IR v11, ONNX, ov::Model, Paddle as `I64` precision (aligned with the original model) to match **new** behavior

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_v10]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]
       
.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]
       
.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]

@endsphinxdirective
